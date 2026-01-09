import os
import time
from pathlib import Path
from typing import Optional
from functools import lru_cache
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Configure HuggingFace Hub BEFORE importing it
# These environment variables control connection timeouts and behavior
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")  # 60s connection timeout
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")  # Disable hf_transfer (can cause issues on Windows)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ.get("HF_HOME", ""))  # Use default cache

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from safetensors.torch import load_file
from app.core.config import get_settings, get_runtime_config
from app.inference.vram_monitor import VRAMMonitor
from app.inference.sae_registry import SAERegistry, SAEEntry


# Download timeout in seconds (2 minutes per attempt, with retries)
SAE_DOWNLOAD_TIMEOUT = 120


def _is_local_path(path: str) -> bool:
    """
    Check if the given path is a local filesystem path (not a HuggingFace model ID).

    Returns True if:
    - Path contains directory separators (/ or \\)
    - Path exists as a directory on the filesystem
    - Path starts with a drive letter (Windows) or / (Unix)
    """
    # Normalize path separators for cross-platform compatibility
    normalized = path.replace("\\", "/")

    # Check for obvious local path patterns
    # Windows: D:/models/... or D:\models\...
    # Unix: /home/user/models/...
    has_path_sep = "/" in normalized or "\\" in path
    starts_with_drive = len(path) >= 2 and path[1] == ":"  # Windows drive letter
    starts_with_slash = path.startswith("/")  # Unix absolute path

    # If it looks like a path, verify the directory exists
    if has_path_sep or starts_with_drive or starts_with_slash:
        try:
            return Path(path).exists() and Path(path).is_dir()
        except (OSError, ValueError):
            # Invalid path characters or other issues
            return False

    # Otherwise, assume it's a HuggingFace model ID (e.g., "google/gemma-3-4b-it")
    return False


def _download_with_timeout(
    repo_id: str,
    filename: str,
    timeout: int = SAE_DOWNLOAD_TIMEOUT,
    force_download: bool = False,
    max_retries: int = 3,
) -> str:
    """Download a file from HuggingFace with timeout and retry logic.

    Uses retry logic to handle transient network issues common on Windows.
    Each retry uses a fresh thread to avoid connection reuse issues.
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            print(f"Download attempt {attempt + 1}/{max_retries} for {filename}...")

            # Use a fresh executor for each attempt to avoid connection reuse issues
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    hf_hub_download,
                    repo_id=repo_id,
                    filename=filename,
                    force_download=force_download,
                    resume_download=True,  # Resume partial downloads
                )
                try:
                    result = future.result(timeout=timeout)
                    print(f"Download successful: {filename}")
                    return result
                except FuturesTimeoutError:
                    last_error = TimeoutError(
                        f"Download timed out after {timeout}s on attempt {attempt + 1}"
                    )
                    print(f"Download timeout on attempt {attempt + 1}, will retry...")
                    # Cancel the future (though this may not stop the underlying download)
                    future.cancel()
        except Exception as e:
            last_error = e
            print(f"Download error on attempt {attempt + 1}: {e}")

            # Don't retry on certain errors
            error_str = str(e).lower()
            if "not found" in error_str or "404" in error_str:
                raise  # File doesn't exist, no point retrying
            if "unauthorized" in error_str or "403" in error_str:
                raise  # Auth error, no point retrying

        # Wait before retry (exponential backoff)
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt  # 1s, 2s, 4s
            print(f"Waiting {wait_time}s before retry...")
            time.sleep(wait_time)

    # All retries exhausted
    raise TimeoutError(
        f"SAE download failed after {max_retries} attempts. "
        f"Last error: {last_error}. "
        f"The file may be large or your connection unstable. "
        f"Try downloading manually: curl -L https://huggingface.co/{repo_id}/resolve/main/{filename} -o sae.safetensors"
    )


def get_available_layers() -> list[int]:
    """Get available SAE layers for the current model configuration."""
    runtime_config = get_runtime_config()
    return runtime_config.get_available_layers()


# For backwards compatibility - this will be deprecated
# Use get_available_layers() instead
AVAILABLE_LAYERS = [9, 17, 22, 29]  # Default for 4B, updated dynamically


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
    _d_model: int = 4096  # Model hidden dimension (updated from runtime_config or when model loads)

    # Multi-SAE support
    _vram_monitor: Optional[VRAMMonitor] = None
    _sae_registry: Optional[SAERegistry] = None
    _strict_memory_mode: bool = True  # Only keep 1 SAE loaded at a time

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize VRAM monitor and SAE registry
            # Use larger safety margin for bigger models
            cls._vram_monitor = VRAMMonitor(safety_margin_gb=3.0)
            cls._sae_registry = SAERegistry(
                vram_monitor=cls._vram_monitor,
                max_sae_budget_gb=20.0
            )
            # Strict memory mode: only keep 1 SAE loaded at a time
            # Useful for 12B+ models where VRAM is very tight
            cls._strict_memory_mode = True
            # Initialize d_model from runtime config (updated when model actually loads)
            runtime_config = get_runtime_config()
            cls._d_model = runtime_config.hidden_size
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
        """Get the default SAE (first available layer).

        For multi-layer usage, prefer get_sae_for_layer() instead.
        """
        if self._sae is not None:
            return self._sae

        # Try to get from registry (first loaded layer)
        runtime_config = get_runtime_config()
        default_layer = runtime_config.get_default_layer()
        entry = self._sae_registry.get_by_layer(default_layer)
        if entry:
            return entry.sae

        raise RuntimeError(
            "No SAE loaded. Use load_sae_for_layer() to load an SAE first, "
            "or run an analysis which loads SAEs automatically."
        )

    @property
    def is_loaded(self) -> bool:
        return self._initialized

    def load_model(self, force_reload: bool = False) -> None:
        """Load the Gemma 3 model into memory.

        SAEs are loaded on-demand via load_sae_for_layer() to minimize VRAM usage.
        This is especially important for larger models (12B, 27B) where VRAM is tight.

        Supports both HuggingFace model IDs (e.g., "google/gemma-3-4b-it") and
        local paths (e.g., "D:/models/gemma-3-4b-it"). Local paths are detected
        automatically and loaded with local_files_only=True to prevent network calls.
        """
        if self._initialized and not force_reload:
            return

        settings = get_settings()
        runtime_config = get_runtime_config()
        device = settings.device
        dtype = torch.float32 if settings.dtype == "float32" else torch.bfloat16

        # Use runtime config for model name (allows dynamic changes)
        model_name = runtime_config.model_name

        # Check if this is a local path - if so, use local_files_only to prevent
        # any network calls that cause socket hangups and caching errors
        is_local = _is_local_path(model_name)
        if is_local:
            print(f"Detected local model path: {model_name}")
            print("Using local_files_only=True to prevent network calls")

        print(f"Loading tokenizer: {model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=is_local,
            trust_remote_code=True,
        )

        print(f"Loading model: {model_name}")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
            local_files_only=is_local,
        )
        self._model.eval()

        # Clear any existing SAEs from previous model
        if self._sae_registry is not None:
            self._sae_registry.clear_all()
        self._sae = None  # Clear legacy SAE reference

        # Get model's hidden dimension for accurate SAE size estimation
        self._d_model = self._get_model_hidden_size()

        # Ensure runtime config is updated to match the loaded model
        # This handles cases where model was loaded from a local path
        # that differs from what was in the config
        if runtime_config.model_name != model_name:
            runtime_config.update(model_name=model_name)
        else:
            # Force update model config even if name matches
            # (ensures layers/hidden_size are correct after restart)
            runtime_config._update_model_config()

        self._initialized = True
        print("Model loaded successfully (SAEs loaded on-demand)")
        print(f"  Model: {model_name}")
        print(f"  Hidden dim: {self._d_model}")
        print(f"  Available layers: {runtime_config.get_available_layers()}")

        # Show VRAM status after model load
        vram = self._vram_monitor.get_snapshot()
        print(f"  VRAM: {vram.allocated_gb:.1f}GB allocated, {vram.free_gb:.1f}GB free")

    def _get_model_hidden_size(self) -> int:
        """Get the hidden dimension of the loaded model."""
        runtime_config = get_runtime_config()
        if self._model is None:
            return runtime_config.hidden_size  # Use config-based estimate

        config = self._model.config

        # For Gemma3ForConditionalGeneration, the text config is nested
        # Try to get the text_config first (for vision-language models)
        text_config = getattr(config, 'text_config', config)

        # Try common attribute names for hidden size
        for attr in ['hidden_size', 'd_model', 'n_embd', 'dim']:
            if hasattr(text_config, attr):
                value = getattr(text_config, attr)
                print(f"  Found {attr}={value} in text_config")
                return value

        # Also check the main config
        for attr in ['hidden_size', 'd_model', 'n_embd', 'dim']:
            if hasattr(config, attr):
                value = getattr(config, attr)
                print(f"  Found {attr}={value} in config")
                return value

        print(f"  Warning: Could not find hidden_size, using config-based value: {runtime_config.hidden_size}")
        print(f"  Config type: {type(config).__name__}")
        print(f"  Config attrs: {[a for a in dir(config) if not a.startswith('_')][:20]}")
        return runtime_config.hidden_size

    def _estimate_sae_size(self, width: str, dtype: torch.dtype = torch.float32) -> int:
        """
        Estimate SAE memory footprint based on width and actual model dimension.

        SAE parameters:
        - w_enc: (d_model, d_sae) - encoder weights
        - w_dec: (d_sae, d_model) - decoder weights
        - threshold: (d_sae,) - threshold values
        - b_enc: (d_sae,) - encoder bias
        - b_dec: (d_model,) - decoder bias
        """
        d_model = self._d_model

        width_map = {
            "16k": 16384,
            "65k": 65536,
            "262k": 262144,
            "1m": 1048576,
        }
        d_sae = width_map.get(width, 65536)

        bytes_per_param = 4 if dtype == torch.float32 else 2

        # Total parameters
        num_params = (
            d_model * d_sae +  # w_enc
            d_sae * d_model +  # w_dec
            d_sae +            # threshold
            d_sae +            # b_enc
            d_model            # b_dec
        )

        return num_params * bytes_per_param

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
        runtime_config = get_runtime_config()
        vram = self.get_vram_status()
        sae_status = self._sae_registry.get_status() if self._sae_registry else {
            "loaded_count": 0,
            "loaded_layers": [],
            "total_size_mb": 0,
            "total_size_gb": 0,
            "max_budget_gb": 0,
            "entries": [],
        }

        return {
            "model_loaded": self._initialized,
            "model_name": runtime_config.model_name if self._initialized else None,
            "model_size": runtime_config.model_size,
            "vram": vram,
            "saes": sae_status,
            "available_layers": get_available_layers(),
        }

    def preflight_check(self, layers: list[int], width: str = "65k") -> dict:
        """
        Check if requested SAEs can be loaded.

        Returns dict with can_load, bytes_needed, bytes_available, etc.
        """
        settings = get_settings()
        runtime_config = get_runtime_config()
        dtype = torch.float32 if settings.dtype == "float32" else torch.bfloat16
        available_layers = get_available_layers()

        # Use actual model hidden size for accurate SAE size estimation
        d_model = self._d_model if self._initialized else runtime_config.hidden_size

        # Calculate bytes needed for new SAEs (not already loaded)
        bytes_needed = 0
        layers_to_load = []
        already_loaded = []

        for layer in layers:
            if layer not in available_layers:
                continue
            if self._sae_registry.is_loaded(layer, width, runtime_config.sae_l0, runtime_config.sae_type):
                already_loaded.append(layer)
            else:
                bytes_needed += self._vram_monitor.estimate_sae_size(width, dtype, d_model)
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
            layer: Layer index (must be in available layers for current model)
            width: SAE width (defaults to runtime config)
            l0: L0 regularization level (defaults to runtime config)
            sae_type: Hook type (defaults to runtime config)

        Returns:
            SAEEntry for the loaded SAE

        Raises:
            ValueError: If layer not available
            MemoryError: If insufficient VRAM
        """
        available_layers = get_available_layers()
        if layer not in available_layers:
            raise ValueError(f"Layer {layer} not available. Choose from {available_layers}")

        settings = get_settings()
        runtime_config = get_runtime_config()
        width = width or runtime_config.sae_width
        l0 = l0 or runtime_config.sae_l0
        sae_type = sae_type or runtime_config.sae_type
        device = settings.device
        dtype = torch.float32 if settings.dtype == "float32" else torch.bfloat16

        # Check if already loaded
        existing = self._sae_registry.get(layer, width, l0, sae_type)
        if existing:
            print(f"SAE for layer {layer} already loaded")
            return existing

        # Estimate size using the actual model's hidden dimension
        estimated_size = self._estimate_sae_size(width, dtype)

        # In strict memory mode, clear ALL other SAEs before loading a new one
        # This is essential for large models (12B+) where only 1 SAE fits in VRAM
        if self._strict_memory_mode and self._sae_registry.count() > 0:
            print(f"Strict memory mode: clearing {self._sae_registry.count()} existing SAE(s)...")
            self._sae_registry.clear_all()
            torch.cuda.empty_cache()

        # Try to free memory if needed
        snapshot = self._vram_monitor.get_snapshot()
        print(f"VRAM before SAE load: {snapshot.allocated_gb:.2f}GB allocated, {snapshot.free_gb:.2f}GB free")

        if not self._vram_monitor.can_allocate(estimated_size):
            # Try evicting any remaining SAEs
            if self._sae_registry.count() > 0:
                print(f"Evicting {self._sae_registry.count()} SAE(s) to free memory...")
                self._sae_registry.clear_all()
            torch.cuda.empty_cache()
            snapshot = self._vram_monitor.get_snapshot()
            print(f"After cache clear: {snapshot.free_gb:.2f}GB free")

            # Check again - if still not enough VRAM, fail early with clear message
            if not self._vram_monitor.can_allocate(estimated_size):
                needed_gb = estimated_size / (1024**3)
                available_gb = snapshot.free_gb
                raise MemoryError(
                    f"Insufficient VRAM to load SAE for layer {layer}. "
                    f"Need ~{needed_gb:.2f}GB but only {available_gb:.2f}GB available. "
                    f"The model is using {snapshot.allocated_gb:.1f}GB. "
                    f"Try using a smaller model or a smaller SAE width (e.g., 16k instead of {width})."
                )

        # Build SAE path and download using runtime config
        sae_subdir = f"{sae_type}/layer_{layer}_width_{width}_l0_{l0}"
        sae_filename = f"{sae_subdir}/params.safetensors"

        print(f"Loading SAE: {runtime_config.sae_repo}/{sae_filename}")

        # Check cache first to avoid lock contention on Windows
        path_to_params = try_to_load_from_cache(
            repo_id=runtime_config.sae_repo,
            filename=sae_filename,
        )

        if path_to_params is None:
            # Not cached, need to download (with timeout to prevent hangs)
            print(f"SAE not in cache, downloading (timeout: {SAE_DOWNLOAD_TIMEOUT}s)...")
            path_to_params = _download_with_timeout(
                repo_id=runtime_config.sae_repo,
                filename=sae_filename,
            )
        else:
            print(f"SAE found in cache: {path_to_params}")

        # Load SAE to CPU first to avoid OOM crash, then transfer to GPU
        # This allows proper error handling if GPU transfer fails
        try:
            params = load_file(path_to_params, device="cpu")
            d_model, d_sae = params["w_enc"].shape
        except Exception as e:
            # Cache file may be corrupted - try force re-download
            print(f"Cached SAE file corrupted, re-downloading: {e}")
            path_to_params = _download_with_timeout(
                repo_id=runtime_config.sae_repo,
                filename=sae_filename,
                force_download=True,
            )
            try:
                params = load_file(path_to_params, device="cpu")
                d_model, d_sae = params["w_enc"].shape
            except Exception as e2:
                raise RuntimeError(f"Failed to load SAE parameters from {path_to_params}: {e2}")

        sae = JumpReLUSAE(d_model, d_sae)
        sae.load_state_dict(params)

        # Free CPU params now that they're loaded into the module
        del params

        # Transfer to GPU with proper error handling
        try:
            sae.to(device)
        except RuntimeError as e:
            # Clean up
            del sae
            torch.cuda.empty_cache()
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                snapshot = self._vram_monitor.get_snapshot()
                raise MemoryError(
                    f"Out of GPU memory transferring SAE for layer {layer} to device. "
                    f"Available: {snapshot.free_gb:.2f}GB. "
                    f"Try using a smaller model or SAE width (16k instead of {width})."
                ) from e
            raise

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
            width: SAE width (defaults to runtime config)

        Returns:
            Dict with loaded, already_loaded, failed, and vram_status
        """
        runtime_config = get_runtime_config()
        width = width or runtime_config.sae_width

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
        runtime_config = get_runtime_config()
        width = width or runtime_config.sae_width
        return self._sae_registry.unload(layer, width, runtime_config.sae_l0, runtime_config.sae_type)

    def get_sae_for_layer(self, layer: int) -> Optional[JumpReLUSAE]:
        """Get the loaded SAE for a specific layer."""
        entry = self._sae_registry.get_by_layer(layer)
        return entry.sae if entry else None

    def get_loaded_layers(self) -> list[int]:
        """Get list of currently loaded layer indices."""
        return self._sae_registry.get_loaded_layers()

    def get_sae_cache_status(self, layers: list[int]) -> list[dict]:
        """Check which SAE files are cached in HuggingFace local cache.

        This checks the disk cache, not whether SAEs are loaded in VRAM.

        Args:
            layers: List of layer indices to check

        Returns:
            List of dicts with {layer, cached, filename} for each layer
        """
        runtime_config = get_runtime_config()
        results = []

        for layer in layers:
            sae_subdir = f"{runtime_config.sae_type}/layer_{layer}_width_{runtime_config.sae_width}_l0_{runtime_config.sae_l0}"
            sae_filename = f"{sae_subdir}/params.safetensors"

            cached_path = try_to_load_from_cache(
                repo_id=runtime_config.sae_repo,
                filename=sae_filename
            )

            results.append({
                "layer": layer,
                "cached": cached_path is not None,
                "filename": sae_filename,
            })

        return results

    def download_sae_files(self, layers: list[int]) -> dict:
        """Download SAE files to HuggingFace cache without loading to GPU.

        Args:
            layers: List of layer indices to download

        Returns:
            Dict with {downloaded: list[int], already_cached: list[int], failed: list[dict]}
        """
        runtime_config = get_runtime_config()
        downloaded = []
        already_cached = []
        failed = []

        for layer in layers:
            sae_subdir = f"{runtime_config.sae_type}/layer_{layer}_width_{runtime_config.sae_width}_l0_{runtime_config.sae_l0}"
            sae_filename = f"{sae_subdir}/params.safetensors"

            # Check if already cached
            cached_path = try_to_load_from_cache(
                repo_id=runtime_config.sae_repo,
                filename=sae_filename
            )

            if cached_path is not None:
                already_cached.append(layer)
                continue

            # Download the file (with timeout to prevent hangs)
            try:
                print(f"Downloading SAE for layer {layer}: {runtime_config.sae_repo}/{sae_filename}")
                _download_with_timeout(
                    repo_id=runtime_config.sae_repo,
                    filename=sae_filename,
                )
                downloaded.append(layer)
            except Exception as e:
                failed.append({"layer": layer, "error": str(e)})

        return {
            "downloaded": downloaded,
            "already_cached": already_cached,
            "failed": failed,
        }

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

        Uses sequential SAE processing to support limited VRAM scenarios where
        only one SAE can be loaded at a time. The forward pass captures raw
        activations for all layers, then each layer is processed sequentially
        with its SAE loaded on-demand.

        Args:
            text: Input text
            layers: List of layer indices to analyze
            top_k: Number of top features per token

        Returns:
            Dict with 'tokens' and 'layers' (dict mapping layer to token_activations)
        """
        settings = get_settings()
        device = settings.device

        # Tokenize
        token_strs = self.tokenize(text)
        input_ids = self.get_token_ids(text).to(device)

        # Run forward pass ONCE and capture raw activations at all layers
        # This is done before loading any SAEs to minimize peak VRAM usage
        with self.capture_activations_multi(layers) as cache:
            with torch.no_grad():
                self._model(input_ids)

        # Clone and cache raw residuals - these stay in memory while we
        # sequentially load/unload SAEs for each layer
        residual_cache = {}
        for layer in layers:
            if cache[layer] is not None:
                residual_cache[layer] = cache[layer].clone()

        # Process each layer SEQUENTIALLY with immediate unload
        # This ensures only one SAE is in VRAM at a time, allowing analysis
        # even when VRAM is tight (model + 1 SAE must fit)
        runtime_config = get_runtime_config()
        layer_results = {}
        num_layers = len(layers)

        for idx, layer in enumerate(layers):
            if layer not in residual_cache:
                print(f"Warning: No cached residual for layer {layer}, skipping")
                continue

            residual = residual_cache[layer]  # [1, seq_len, d_model]

            # Load SAE for this layer
            try:
                self.load_sae_for_layer(layer)
            except MemoryError as e:
                print(f"Failed to load SAE for layer {layer}: {e}")
                continue

            sae = self.get_sae_for_layer(layer)
            if sae is None:
                print(f"Warning: SAE for layer {layer} not available after load")
                continue

            # Encode with SAE
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

            # Unload SAE after encoding to free VRAM for next layer
            # For multi-layer analysis: always unload to ensure next layer can load
            # For single layer: respect memory_saver_mode setting
            is_last_layer = (idx == num_layers - 1)
            should_unload = (num_layers > 1) or runtime_config.memory_saver_mode

            if should_unload and (not is_last_layer or runtime_config.memory_saver_mode):
                self._sae_registry.unload_layer(layer)
                print(f"Unloaded SAE for layer {layer} (freeing VRAM for next operation)")

        # Clean up residual cache
        del residual_cache
        torch.cuda.empty_cache()

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

    def _get_target_layer(self, layer_idx: int | None = None):
        """
        Get the target layer module for SAE operations.

        Args:
            layer_idx: Specific layer index. If None, uses default layer.

        Returns:
            The transformer layer module at the specified index.
        """
        if layer_idx is None:
            runtime_config = get_runtime_config()
            layer_idx = runtime_config.get_default_layer()

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
        """Get the SAE feature dimension from config (doesn't require SAE to be loaded)."""
        runtime_config = get_runtime_config()
        width_map = {
            "16k": 16384,
            "65k": 65536,
            "262k": 262144,
            "1m": 1048576,
        }
        return width_map.get(runtime_config.sae_width, 65536)

    def get_hook_point(self) -> str:
        """Get the SAE hook point description."""
        runtime_config = get_runtime_config()
        return f"{runtime_config.sae_type}/layer_{runtime_config.get_default_layer()}"

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
