import torch
from typing import Generator

from app.inference.model_loader import get_model_manager
from app.schemas.analysis import SteeringFeature
from app.core.config import get_settings


def generate_with_steering(
    prompt: str,
    steering_features: list[SteeringFeature],
    max_tokens: int = 128,
    temperature: float = 0.7,
) -> str:
    """
    Generate text with steering applied.

    Args:
        prompt: Input prompt
        steering_features: Features to steer
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated text
    """
    manager = get_model_manager()

    if not manager.is_loaded:
        manager.load_model()

    # Convert SteeringFeature objects to dict format
    features = [
        {'feature_id': sf.feature_id, 'coefficient': sf.coefficient}
        for sf in steering_features
    ]

    return manager.generate_with_steering(
        prompt=prompt,
        steering_features=features,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )


def generate_streaming(
    prompt: str,
    steering_features: list[SteeringFeature],
    max_tokens: int = 128,
    temperature: float = 0.7,
) -> Generator[str, None, None]:
    """
    Generate text with streaming output.

    Args:
        prompt: Input prompt
        steering_features: Features to steer
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Yields:
        Individual tokens as they are generated
    """
    settings = get_settings()
    manager = get_model_manager()

    if not manager.is_loaded:
        manager.load_model()

    model = manager.model
    tokenizer = manager.tokenizer
    sae = manager.sae
    device = settings.device

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    current_tokens = input_ids

    # Create steering vector if features provided
    steering_vector = None
    if steering_features:
        steering_vector = torch.zeros(
            sae.d_in,
            device=device,
            dtype=model.dtype
        )
        for sf in steering_features:
            steering_vector += sf.coefficient * sae.w_dec[sf.feature_id]

    # Set up steering hook
    hooks = []
    if steering_vector is not None:
        target_layer = manager._get_target_layer()

        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                modified = hidden_states + steering_vector.unsqueeze(0).unsqueeze(0)
                return (modified,) + output[1:]
            else:
                return output + steering_vector.unsqueeze(0).unsqueeze(0)

        hook = target_layer.register_forward_hook(steering_hook)
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

            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
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
) -> tuple[str, str]:
    """
    Generate both baseline and steered outputs for comparison.

    Args:
        prompt: Input prompt
        steering_features: Features to steer
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Tuple of (baseline_output, steered_output)
    """
    # Generate baseline (no steering)
    baseline = generate_with_steering(
        prompt=prompt,
        steering_features=[],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Generate steered output
    steered = generate_with_steering(
        prompt=prompt,
        steering_features=steering_features,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return baseline, steered
