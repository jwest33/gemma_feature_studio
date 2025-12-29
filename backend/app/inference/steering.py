"""
Steering module with norm preservation.

Implements feature steering with three normalization modes:
- none: Raw steering without normalization
- preserve_norm: Rescale output to maintain original activation norm
- clamp: Allow bounded norm changes within a factor
"""

import torch
from typing import Generator

from app.inference.model_loader import get_model_manager
from app.schemas.analysis import SteeringFeature, NormalizationMode
from app.core.config import get_settings, get_runtime_config


def generate_with_steering(
    prompt: str,
    steering_features: list[SteeringFeature],
    max_tokens: int = 128,
    temperature: float = 0.7,
    normalization: NormalizationMode = NormalizationMode.preserve_norm,
    norm_clamp_factor: float = 1.5,
    unit_normalize: bool = False,
) -> str:
    """
    Generate text with steering applied.

    Args:
        prompt: Input prompt
        steering_features: Features to steer
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        normalization: Post-steering normalization mode
        norm_clamp_factor: For clamp mode, max allowed norm change ratio
        unit_normalize: Normalize decoder vectors to unit norm before applying

    Returns:
        Generated text
    """
    manager = get_model_manager()

    if not manager.is_loaded:
        manager.load_model()

    # Convert SteeringFeature objects to dict format
    features = [
        {
            'feature_id': sf.feature_id,
            'coefficient': sf.coefficient,
            'layer': sf.layer,
        }
        for sf in steering_features
    ]

    return manager.generate_with_steering(
        prompt=prompt,
        steering_features=features,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )


def create_steering_hook(
    sae,
    features: list[SteeringFeature],
    normalization: NormalizationMode,
    norm_clamp_factor: float,
    unit_normalize: bool,
):
    """
    Create a forward hook that applies steering with norm preservation.

    Args:
        sae: The SAE object with decoder weights (w_dec)
        features: List of SteeringFeature objects
        normalization: Normalization mode
        norm_clamp_factor: Clamp factor for bounded norm changes
        unit_normalize: Whether to unit-normalize decoder vectors

    Returns:
        A forward hook function
    """
    def steering_hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Handle different tensor shapes
        # Shape could be (batch, seq_len, hidden) or (batch, hidden)
        if hidden_states.dim() == 2:
            # Single token case: (batch, hidden)
            original_norm = torch.norm(hidden_states, dim=-1, keepdim=True)

            for sf in features:
                feat_idx = sf.feature_id
                coeff = sf.coefficient
                decoder_vec = sae.w_dec[feat_idx].to(hidden_states.dtype)

                # Optionally normalize decoder vector to unit norm
                if unit_normalize:
                    decoder_vec = decoder_vec / (torch.norm(decoder_vec) + 1e-8)

                # Apply steering: output += coeff * original_norm * decoder_vec
                hidden_states = hidden_states + coeff * original_norm * decoder_vec

            # Apply post-steering normalization
            if normalization == NormalizationMode.preserve_norm:
                new_norm = torch.norm(hidden_states, dim=-1, keepdim=True)
                hidden_states = hidden_states * (original_norm / (new_norm + 1e-8))
            elif normalization == NormalizationMode.clamp:
                new_norm = torch.norm(hidden_states, dim=-1, keepdim=True)
                norm_ratio = new_norm / (original_norm + 1e-8)
                scale = torch.ones_like(norm_ratio)
                scale = torch.where(norm_ratio > norm_clamp_factor, norm_clamp_factor / norm_ratio, scale)
                scale = torch.where(norm_ratio < 1.0 / norm_clamp_factor, (1.0 / norm_clamp_factor) / norm_ratio, scale)
                hidden_states = hidden_states * scale
        else:
            # Sequence case: (batch, seq_len, hidden)
            # Apply to all positions except BOS token (position 0)
            original_norms = torch.norm(hidden_states[:, 1:], dim=-1, keepdim=True)

            for sf in features:
                feat_idx = sf.feature_id
                coeff = sf.coefficient
                decoder_vec = sae.w_dec[feat_idx].to(hidden_states.dtype)

                # Optionally normalize decoder vector to unit norm
                if unit_normalize:
                    decoder_vec = decoder_vec / (torch.norm(decoder_vec) + 1e-8)

                # Apply steering to all positions except BOS
                hidden_states[:, 1:] = hidden_states[:, 1:] + coeff * original_norms * decoder_vec

            # Apply post-steering normalization
            if normalization == NormalizationMode.preserve_norm:
                new_norms = torch.norm(hidden_states[:, 1:], dim=-1, keepdim=True)
                hidden_states[:, 1:] = hidden_states[:, 1:] * (original_norms / (new_norms + 1e-8))
            elif normalization == NormalizationMode.clamp:
                new_norms = torch.norm(hidden_states[:, 1:], dim=-1, keepdim=True)
                norm_ratios = new_norms / (original_norms + 1e-8)
                scale = torch.ones_like(norm_ratios)
                scale = torch.where(norm_ratios > norm_clamp_factor, norm_clamp_factor / norm_ratios, scale)
                scale = torch.where(norm_ratios < 1.0 / norm_clamp_factor, (1.0 / norm_clamp_factor) / norm_ratios, scale)
                hidden_states[:, 1:] = hidden_states[:, 1:] * scale

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        else:
            return hidden_states

    return steering_hook


def generate_streaming(
    prompt: str,
    steering_features: list[SteeringFeature],
    max_tokens: int = 128,
    temperature: float = 0.7,
    normalization: NormalizationMode = NormalizationMode.preserve_norm,
    norm_clamp_factor: float = 1.5,
    unit_normalize: bool = False,
) -> Generator[str, None, None]:
    """
    Generate text with streaming output and norm preservation.

    Args:
        prompt: Input prompt
        steering_features: Features to steer
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        normalization: Post-steering normalization mode
        norm_clamp_factor: For clamp mode, max allowed norm change ratio
        unit_normalize: Normalize decoder vectors to unit norm before applying

    Yields:
        Individual tokens as they are generated
    """
    settings = get_settings()
    runtime_config = get_runtime_config()
    manager = get_model_manager()

    if not manager.is_loaded:
        manager.load_model()

    model = manager.model
    tokenizer = manager.tokenizer
    device = settings.device

    # Load SAE for the default layer if not already loaded
    # This is needed for steering which requires an SAE
    default_layer = runtime_config.get_default_layer()
    try:
        manager.load_sae_for_layer(default_layer)
    except MemoryError as e:
        # If we can't load SAE due to memory, proceed without steering
        print(f"Warning: Could not load SAE for steering: {e}")

    # Get SAE (may be None if loading failed)
    try:
        sae = manager.sae
    except RuntimeError:
        sae = None  # Proceed without SAE/steering if not available

    # Apply chat template for Gemma model
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize the formatted prompt
    input_ids = tokenizer.encode(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=True
    ).to(device)
    current_tokens = input_ids

    # Get stop token IDs (EOS and end_of_turn for Gemma chat)
    stop_token_ids = {tokenizer.eos_token_id}

    # Add end_of_turn token if it exists (Gemma chat models)
    end_of_turn_token = "<end_of_turn>"
    end_of_turn_ids = tokenizer.encode(end_of_turn_token, add_special_tokens=False)
    if end_of_turn_ids:
        stop_token_ids.add(end_of_turn_ids[0])

    # Set up steering hooks if features provided
    hooks = []
    if steering_features:
        target_layer = manager._get_target_layer()

        # Create steering hook with norm preservation
        hook_fn = create_steering_hook(
            sae=sae,
            features=steering_features,
            normalization=normalization,
            norm_clamp_factor=norm_clamp_factor,
            unit_normalize=unit_normalize,
        )

        hook = target_layer.register_forward_hook(hook_fn)
        hooks.append(hook)

    try:
        # Generate token by token
        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = model(current_tokens)
                logits = outputs.logits

            # Get next token
            next_token_logits = logits[0, -1, :]

            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax().unsqueeze(0)

            # Check for stop tokens (EOS or end_of_turn)
            if next_token.item() in stop_token_ids:
                break

            # Decode and yield the new token
            token_str = tokenizer.decode(next_token)
            yield token_str

            # Append to context
            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0)], dim=1)

    finally:
        # Clean up hooks
        for hook in hooks:
            hook.remove()


def generate_comparison(
    prompt: str,
    steering_features: list[SteeringFeature],
    max_tokens: int = 128,
    temperature: float = 0.7,
    normalization: NormalizationMode = NormalizationMode.preserve_norm,
    norm_clamp_factor: float = 1.5,
    unit_normalize: bool = False,
) -> tuple[str, str]:
    """
    Generate both baseline and steered outputs for comparison.

    Args:
        prompt: Input prompt
        steering_features: Features to steer
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        normalization: Post-steering normalization mode
        norm_clamp_factor: For clamp mode, max allowed norm change ratio
        unit_normalize: Normalize decoder vectors to unit norm before applying

    Returns:
        Tuple of (baseline_output, steered_output)
    """
    # Generate baseline (no steering)
    baseline_tokens = []
    for token in generate_streaming(
        prompt=prompt,
        steering_features=[],
        max_tokens=max_tokens,
        temperature=temperature,
        normalization=NormalizationMode.none,
        norm_clamp_factor=norm_clamp_factor,
        unit_normalize=False,
    ):
        baseline_tokens.append(token)
    baseline = "".join(baseline_tokens)

    # Generate steered output
    steered_tokens = []
    for token in generate_streaming(
        prompt=prompt,
        steering_features=steering_features,
        max_tokens=max_tokens,
        temperature=temperature,
        normalization=normalization,
        norm_clamp_factor=norm_clamp_factor,
        unit_normalize=unit_normalize,
    ):
        steered_tokens.append(token)
    steered = "".join(steered_tokens)

    return baseline, steered
