import torch
import torch.nn as nn
from typing import Optional
from functools import lru_cache
from contextlib import contextmanager

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from app.core.config import get_settings
from app.inference.vram_monitor import VRAMMonitor
from app.inference.sae_registry import SAERegistry, SAEEntry

# Available layers for Gemma-3-4B
AVAILABLE_LAYERS = [9, 17, 22, 29]


class JumpReLUSAE(nn.Module):
    """JumpReLU Sparse Autoencoder for Gemma Scope 2."""

    def __init__(self, d_in: int, d_sae: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.w_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.w_dec = nn.Parameter(torch.zeros(d_sae, d_in))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def encode(self, input_acts: torch.Tensor) -> torch.Tensor:
        pre_acts = input_acts @ self.w_enc + self.b_enc
        mask = pre_acts > self.threshold
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.w_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts = self.encode(x)
        return self.decode(acts)


class ModelManager:
    """Manages loading and caching of Gemma 3 model and multiple Gemma Scope 2 SAEs."""

    _instance: Optional["ModelManager"] = None
    _model: Optional[AutoModelForCausalLM] = None
    _tokenizer: Optional[AutoTokenizer] = None
    _sae: Optional[JumpReLUSAE] = None  # Legacy single SAE (for backwards compatibility)
    _initialized: bool = False
    _activation_cache: dict = {}
    _hooks: list = []

    # Multi-SAE support
    _vram_monitor: Optional[VRAMMonitor] = None
    _sae_registry: Optional[SAERegistry] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize VRAM monitor and SAE registry
            cls._vram_monitor = VRAMMonitor(safety_margin_gb=2.0)
            cls._sae_registry = SAERegistry(
                vram_monitor=cls._vram_monitor,
                max_sae_budget_gb=20.0
            )
        return cls._instance

    @property
    def model(self) -> AutoModelForCausalLM:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._tokenizer

    @property
    def sae(self) -> JumpReLUSAE:
        if self._sae is None:
            raise RuntimeError("SAE not loaded. Call load_model() first.")
        return self._sae

    @property
    def is_loaded(self) -> bool:
        return self._initialized

    def load_model(self, force_reload: bool = False) -> None:
        """Load the Gemma 3 model and Gemma Scope 2 SAE into memory."""
        if self._initialized and not force_reload:
            return

        settings = get_settings()
        device = settings.device
        dtype = torch.float32 if settings.dtype == "float32" else torch.bfloat16

        print(f"Loading tokenizer: {settings.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(settings.model_name)

        print(f"Loading model: {settings.model_name}")
        self._model = AutoModelForCausalLM.from_pretrained(
            settings.model_name,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self._model.eval()

        # Build the SAE path
        sae_subdir = f"{settings.sae_type}/layer_{settings.sae_layer}_width_{settings.sae_width}_l0_{settings.sae_l0}"
        sae_filename = f"{sae_subdir}/params.safetensors"

        print(f"Loading SAE from: {settings.sae_repo}/{sae_filename}")
        path_to_params = hf_hub_download(
            repo_id=settings.sae_repo,
            filename=sae_filename,
        )

        params = load_file(path_to_params, device=device)
        d_model, d_sae = params["w_enc"].shape

        self._sae = JumpReLUSAE(d_model, d_sae)
        self._sae.load_state_dict(params)
        self._sae.to(device)
        self._sae.eval()

        self._initialized = True
        print("Model and SAE loaded successfully")
        print(f"  Model: {settings.model_name}")
        print(f"  SAE: {sae_subdir}")
        print(f"  SAE width: {d_sae}")

    def unload_model(self) -> None:
        """Unload model from memory to free GPU resources."""
        self._clear_hooks()
        # Clear SAE registry
        if self._sae_registry is not None:
            self._sae_registry.clear_all()
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if self._sae is not None:
            del self._sae
            self._sae = None
        self._initialized = False
        self._activation_cache.clear()
        torch.cuda.empty_cache()
        print("Model and all SAEs unloaded")

    # =========================================================================
    # VRAM and System Status
    # =========================================================================

    def get_vram_status(self) -> dict:
        """Get current VRAM usage status."""
        snapshot = self._vram_monitor.get_snapshot()
        return snapshot.to_dict()

    def get_system_status(self) -> dict:
        """Get complete system status including model and SAEs."""
        settings = get_settings()
        vram = self.get_vram_status()
        sae_status = self._sae_registry.get_status() if self._sae_registry else {}

        return {
            "model_loaded": self._initialized,
            "model_name": settings.model_name if self._initialized else None,
            "vram": vram,
            "saes": sae_status,
            "available_layers": AVAILABLE_LAYERS,
        }

    def preflight_check(self, layers: list[int], width: str = "65k") -> dict:
        """
        Check if requested SAEs can be loaded.

        Returns dict with can_load, bytes_needed, bytes_available, etc.
        """
        settings = get_settings()
        dtype = torch.float32 if settings.dtype == "float32" else torch.bfloat16

        # Calculate bytes needed for new SAEs (not already loaded)
        bytes_needed = 0
        layers_to_load = []
        already_loaded = []

        for layer in layers:
            if layer not in AVAILABLE_LAYERS:
                continue
            if self._sae_registry.is_loaded(layer, width, settings.sae_l0, settings.sae_type):
                already_loaded.append(layer)
            else:
                bytes_needed += self._vram_monitor.estimate_sae_size(width, dtype)
                layers_to_load.append(layer)

        snapshot = self._vram_monitor.get_snapshot()
        bytes_available = snapshot.free_bytes

        can_load = bytes_available >= bytes_needed + self._vram_monitor.safety_margin_bytes

        result = {
            "can_load": can_load,
            "bytes_needed": bytes_needed,
            "bytes_available": bytes_available,
            "layers_to_load": layers_to_load,
            "already_loaded": already_loaded,
            "recommendation": None,
        }

        if not can_load:
            deficit = bytes_needed - bytes_available + self._vram_monitor.safety_margin_bytes
            result["recommendation"] = (
                f"Need {self._vram_monitor.format_bytes(deficit)} more memory. "
                "Consider unloading unused layers or using a smaller SAE width."
            )

        return result

    # =========================================================================
    # Multi-SAE Loading
    # =========================================================================

    def load_sae_for_layer(
        self,
        layer: int,
        width: str = None,
        l0: str = None,
        sae_type: str = None,
    ) -> SAEEntry:
        """
        Load SAE for a specific layer.

        Args:
            layer: Layer index (must be in AVAILABLE_LAYERS)
            width: SAE width (defaults to settings)
            l0: L0 regularization level (defaults to settings)
            sae_type: Hook type (defaults to settings)

        Returns:
            SAEEntry for the loaded SAE

        Raises:
            ValueError: If layer not available
            MemoryError: If insufficient VRAM
        """
        if layer not in AVAILABLE_LAYERS:
            raise ValueError(f"Layer {layer} not available. Choose from {AVAILABLE_LAYERS}")

        settings = get_settings()
        width = width or settings.sae_width
        l0 = l0 or settings.sae_l0
        sae_type = sae_type or settings.sae_type
        device = settings.device
        dtype = torch.float32 if settings.dtype == "float32" else torch.bfloat16

        # Check if already loaded
        existing = self._sae_registry.get(layer, width, l0, sae_type)
        if existing:
            print(f"SAE for layer {layer} already loaded")
            return existing

        # Estimate size and check memory
        estimated_size = self._vram_monitor.estimate_sae_size(width, dtype)

        if not self._vram_monitor.can_allocate(estimated_size):
            # Try to evict LRU SAEs
            freed = self._sae_registry.evict_lru(estimated_size)
            if not self._vram_monitor.can_allocate(estimated_size):
                snapshot = self._vram_monitor.get_snapshot()
                raise MemoryError(
                    f"Insufficient VRAM to load SAE for layer {layer}. "
                    f"Need ~{estimated_size / (1024**3):.2f}GB, "
                    f"have {snapshot.free_gb:.2f}GB free."
                )

        # Build SAE path and download
        sae_subdir = f"{sae_type}/layer_{layer}_width_{width}_l0_{l0}"
        sae_filename = f"{sae_subdir}/params.safetensors"

        print(f"Loading SAE: {settings.sae_repo}/{sae_filename}")
        path_to_params = hf_hub_download(
            repo_id=settings.sae_repo,
            filename=sae_filename,
        )

        params = load_file(path_to_params, device=device)
        d_model, d_sae = params["w_enc"].shape

        sae = JumpReLUSAE(d_model, d_sae)
        sae.load_state_dict(params)
        sae.to(device)
        sae.eval()

        # Calculate actual size
        actual_size = sum(p.numel() * p.element_size() for p in sae.parameters())

        # Register in registry
        entry = self._sae_registry.register(
            sae=sae,
            layer=layer,
            width=width,
            l0=l0,
            sae_type=sae_type,
            size_bytes=actual_size,
        )

        print(f"SAE loaded: {entry.sae_id} ({actual_size / (1024**2):.1f}MB)")
        return entry

    def load_saes_for_layers(self, layers: list[int], width: str = None) -> dict:
        """
        Load SAEs for multiple layers with memory-aware loading.

        Args:
            layers: List of layer indices
            width: SAE width (defaults to settings)

        Returns:
            Dict with loaded, already_loaded, failed, and vram_status
        """
        settings = get_settings()
        width = width or settings.sae_width

        # Pre-flight check
        check = self.preflight_check(layers, width)

        results = {
            "loaded": [],
            "already_loaded": check["already_loaded"],
            "failed": [],
        }

        for layer in check["layers_to_load"]:
            try:
                self.load_sae_for_layer(layer, width)
                results["loaded"].append(layer)
            except MemoryError as e:
                results["failed"].append({"layer": layer, "error": str(e)})
                # Stop trying to load more if we hit memory issues
                break
            except Exception as e:
                results["failed"].append({"layer": layer, "error": str(e)})

        results["vram_status"] = self.get_vram_status()
        return results

    def unload_sae_for_layer(self, layer: int, width: str = None) -> bool:
        """Unload SAE for a specific layer."""
        settings = get_settings()
        width = width or settings.sae_width
        return self._sae_registry.unload(layer, width, settings.sae_l0, settings.sae_type)

    def get_sae_for_layer(self, layer: int) -> Optional[JumpReLUSAE]:
        """Get the loaded SAE for a specific layer."""
        entry = self._sae_registry.get_by_layer(layer)
        return entry.sae if entry else None

    def get_loaded_layers(self) -> list[int]:
        """Get list of currently loaded layer indices."""
        return self._sae_registry.get_loaded_layers()

    # =========================================================================
    # Multi-Layer Activation Capture
    # =========================================================================

    @contextmanager
    def capture_activations_multi(self, layers: list[int]):
        """
        Context manager to capture activations at multiple layers.

        Args:
            layers: List of layer indices to capture

        Yields:
            Dict mapping layer index to captured activations
        """
        self._activation_cache.clear()

        # Initialize cache for each layer
        for layer in layers:
            self._activation_cache[layer] = None

        def make_hook(layer_idx: int):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                self._activation_cache[layer_idx] = hidden_states.detach()
            return hook_fn

        # Register hooks on all target layers
        transformer_layers = self._find_layers()
        for layer in layers:
            if layer >= len(transformer_layers):
                raise ValueError(f"Layer {layer} out of range. Model has {len(transformer_layers)} layers.")
            hook = transformer_layers[layer].register_forward_hook(make_hook(layer))
            self._hooks.append(hook)

        try:
            yield self._activation_cache
        finally:
            self._clear_hooks()

    def get_feature_activations_multi(
        self,
        text: str,
        layers: list[int],
        top_k: int = 50,
    ) -> dict:
        """
        Get top-k SAE feature activations for each token across multiple layers.

        Args:
            text: Input text
            layers: List of layer indices to analyze
            top_k: Number of top features per token

        Returns:
            Dict with 'tokens' and 'layers' (dict mapping layer to token_activations)
        """
        settings = get_settings()
        device = settings.device

        # Ensure SAEs are loaded for all requested layers
        for layer in layers:
            if self.get_sae_for_layer(layer) is None:
                self.load_sae_for_layer(layer)

        # Tokenize
        token_strs = self.tokenize(text)
        input_ids = self.get_token_ids(text).to(device)

        # Run forward pass and capture activations at all layers
        with self.capture_activations_multi(layers) as cache:
            with torch.no_grad():
                self._model(input_ids)

        # Process each layer
        layer_results = {}
        for layer in layers:
            residual = cache[layer]  # [1, seq_len, d_model]
            sae = self.get_sae_for_layer(layer)

            if sae is None:
                continue

            with torch.no_grad():
                sae_features = sae.encode(residual.to(torch.float32))

            sae_features = sae_features.squeeze(0)  # [seq_len, d_sae]
            top_values, top_indices = torch.topk(
                sae_features, k=min(top_k, sae_features.shape[-1]), dim=-1
            )

            token_activations = []
            for i, token in enumerate(token_strs):
                features = []
                for j in range(top_values.shape[-1]):
                    features.append({
                        'id': top_indices[i, j].item(),
                        'activation': top_values[i, j].item(),
                    })
                token_activations.append({
                    'token': token,
                    'position': i,
                    'top_features': features,
                })

            layer_results[layer] = token_activations

        return {
            'tokens': token_strs,
            'layers': layer_results,
        }

    def _clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _find_layers(self):
        """
        Find the transformer layers in the model.

        Handles different Gemma architectures with recursive fallback.
        """
        model = self._model

        # Standard Gemma text model: model.model.layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            print("Found layers at: model.model.layers")
            return model.model.layers

        # Gemma 3 with top-level language_model: model.language_model.model.layers
        if (hasattr(model, 'language_model') and
            hasattr(model.language_model, 'model') and
            hasattr(model.language_model.model, 'layers')):
            print("Found layers at: model.language_model.model.layers")
            return model.language_model.model.layers

        # Gemma 3 multimodal nested: model.model.language_model.model.layers
        if (hasattr(model, 'model') and
            hasattr(model.model, 'language_model') and
            hasattr(model.model.language_model, 'model') and
            hasattr(model.model.language_model.model, 'layers')):
            print("Found layers at: model.model.language_model.model.layers")
            return model.model.language_model.model.layers

        # text_model path: model.model.text_model.layers
        if (hasattr(model, 'model') and
            hasattr(model.model, 'text_model') and
            hasattr(model.model.text_model, 'layers')):
            print("Found layers at: model.model.text_model.layers")
            return model.model.text_model.layers

        # Direct text_model: model.text_model.layers
        if hasattr(model, 'text_model') and hasattr(model.text_model, 'layers'):
            print("Found layers at: model.text_model.layers")
            return model.text_model.layers

        # Recursive fallback to find layers anywhere
        print("Could not find layers automatically. Searching model structure...")
        print(f"  model type: {type(model).__name__}")

        def find_layers_recursive(obj, path="model", depth=0):
            if depth > 5:
                return None
            if hasattr(obj, 'layers') and hasattr(obj.layers, '__len__'):
                try:
                    if len(obj.layers) > 0:
                        print(f"  FOUND layers at: {path}.layers (count: {len(obj.layers)})")
                        return obj.layers
                except:
                    pass
            for attr_name in ['model', 'language_model', 'text_model', 'decoder', 'encoder', 'transformer']:
                if hasattr(obj, attr_name):
                    result = find_layers_recursive(
                        getattr(obj, attr_name),
                        f"{path}.{attr_name}",
                        depth + 1
                    )
                    if result is not None:
                        return result
            return None

        layers = find_layers_recursive(model)
        if layers is not None:
            return layers

        # Debug: print model structure to help diagnose
        print("\nModel structure exploration:")
        if hasattr(model, 'model'):
            print(f"  model.model type: {type(model.model).__name__}")
            print(f"  model.model attrs: {[a for a in dir(model.model) if not a.startswith('_')][:20]}")
        if hasattr(model, 'language_model'):
            print(f"  model.language_model type: {type(model.language_model).__name__}")
            print(f"  model.language_model attrs: {[a for a in dir(model.language_model) if not a.startswith('_')][:20]}")

        raise AttributeError(
            f"Cannot find transformer layers in model of type {type(model).__name__}. "
            "Please inspect the model structure and update _find_layers()."
        )

    def _get_target_layer(self):
        """Get the target layer module."""
        settings = get_settings()
        layer_idx = settings.sae_layer

        layers = self._find_layers()
        if layer_idx >= len(layers):
            raise ValueError(f"Layer {layer_idx} out of range. Model has {len(layers)} layers.")
        return layers[layer_idx]

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text and return token strings."""
        tokens = self._tokenizer.encode(text, return_tensors="pt")
        token_strs = [self._tokenizer.decode([t]) for t in tokens[0]]
        return token_strs

    def get_token_ids(self, text: str) -> torch.Tensor:
        """Get token IDs for text."""
        return self._tokenizer.encode(text, return_tensors="pt")

    def get_sae_width(self) -> int:
        """Get the SAE feature dimension."""
        return self._sae.d_sae

    def get_hook_point(self) -> str:
        """Get the SAE hook point description."""
        settings = get_settings()
        return f"{settings.sae_type}/layer_{settings.sae_layer}"

    @contextmanager
    def capture_activations(self):
        """Context manager to capture activations at the SAE hook point."""
        self._activation_cache.clear()

        def hook_fn(module, input, output):
            # For Gemma 3, residual stream is the output of the layer
            # output is typically (hidden_states, ...) tuple or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self._activation_cache['residual'] = hidden_states.detach()

        # Register hook on target layer
        target_layer = self._get_target_layer()
        hook = target_layer.register_forward_hook(hook_fn)
        self._hooks.append(hook)

        try:
            yield self._activation_cache
        finally:
            self._clear_hooks()

    def encode_with_sae(self, activations: torch.Tensor) -> torch.Tensor:
        """Encode activations using the SAE."""
        return self.sae.encode(activations.to(torch.float32))

    def decode_with_sae(self, features: torch.Tensor) -> torch.Tensor:
        """Decode SAE features back to activation space."""
        return self.sae.decode(features)

    def get_feature_activations(self, text: str, top_k: int = 50) -> dict:
        """
        Get top-k SAE feature activations for each token in the text.

        Returns:
            dict with 'tokens', 'activations' (top-k features per token)
        """
        settings = get_settings()
        device = settings.device

        # Tokenize
        token_strs = self.tokenize(text)
        input_ids = self.get_token_ids(text).to(device)

        # Run forward pass and capture activations
        with self.capture_activations() as cache:
            with torch.no_grad():
                self._model(input_ids)

        residual = cache['residual']  # [1, seq_len, d_model]

        # Encode with SAE
        with torch.no_grad():
            sae_features = self.encode_with_sae(residual)  # [1, seq_len, d_sae]

        # Get top-k features per token
        sae_features = sae_features.squeeze(0)  # [seq_len, d_sae]
        top_values, top_indices = torch.topk(sae_features, k=min(top_k, sae_features.shape[-1]), dim=-1)

        # Convert to list format
        token_activations = []
        for i, token in enumerate(token_strs):
            features = []
            for j in range(top_values.shape[-1]):
                features.append({
                    'id': top_indices[i, j].item(),
                    'activation': top_values[i, j].item(),
                })
            token_activations.append({
                'token': token,
                'position': i,
                'top_features': features,
            })

        return {
            'tokens': token_strs,
            'token_activations': token_activations,
        }

    def generate_with_steering(
        self,
        prompt: str,
        steering_features: list[dict],  # [{'feature_id': int, 'coefficient': float}]
        max_new_tokens: int = 100,
        temperature: float = 0.7,
    ) -> str:
        """Generate text with SAE feature steering."""
        settings = get_settings()
        device = settings.device

        # Apply chat template for Gemma model
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize the formatted prompt
        input_ids = self._tokenizer.encode(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=True
        ).to(device)

        # Create steering vector from features
        steering_vector = torch.zeros(self._sae.d_in, device=device, dtype=self._model.dtype)

        for feature in steering_features:
            feature_id = feature['feature_id']
            coefficient = feature['coefficient']
            # Get the decoder vector for this feature
            decoder_vector = self._sae.w_dec[feature_id]
            steering_vector += coefficient * decoder_vector

        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                modified = hidden_states + steering_vector.unsqueeze(0).unsqueeze(0)
                return (modified,) + output[1:]
            else:
                return output + steering_vector.unsqueeze(0).unsqueeze(0)

        # Register steering hook
        target_layer = self._get_target_layer()
        hook = target_layer.register_forward_hook(steering_hook)

        # Get stop token IDs (EOS and end_of_turn for Gemma chat)
        stop_token_ids = [self._tokenizer.eos_token_id]
        end_of_turn_ids = self._tokenizer.encode("<end_of_turn>", add_special_tokens=False)
        if end_of_turn_ids:
            stop_token_ids.append(end_of_turn_ids[0])

        try:
            with torch.no_grad():
                outputs = self._model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                    eos_token_id=stop_token_ids,
                )

            # Decode only the newly generated tokens (skip the input prompt)
            input_length = input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            generated_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return generated_text
        finally:
            hook.remove()


@lru_cache
def get_model_manager() -> ModelManager:
    """Get the singleton model manager instance."""
    return ModelManager()
