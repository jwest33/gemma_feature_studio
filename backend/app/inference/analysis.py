import torch

from app.inference.model_loader import get_model_manager
from app.schemas.analysis import (
    FeatureActivation,
    TokenActivations,
    LayerActivations,
    AnalyzeResponse,
)
from app.core.config import get_settings, get_runtime_config


def analyze_prompt(
    prompt: str,
    top_k: int = 50,
    include_bos: bool = False,
) -> AnalyzeResponse:
    """
    Analyze a prompt and extract SAE feature activations.

    Args:
        prompt: The input text to analyze
        top_k: Number of top features to return per token
        include_bos: Whether to include the BOS token in analysis

    Returns:
        AnalyzeResponse with token-level feature activations
    """
    settings = get_settings()
    manager = get_model_manager()

    if not manager.is_loaded:
        manager.load_model()

    # Use the model manager's built-in feature activation extraction
    result = manager.get_feature_activations(prompt, top_k=top_k)

    token_strs = result['tokens']
    raw_token_activations = result['token_activations']

    # Determine start position (skip BOS if requested)
    start_pos = 0 if include_bos else 1

    # Build token activations in the expected schema format
    token_activations = []
    for data in raw_token_activations[start_pos:]:
        top_features = [
            FeatureActivation(
                id=f['id'],
                activation=f['activation'],
            )
            for f in data['top_features']
            if f['activation'] > 0  # Only include active features
        ]

        token_activations.append(
            TokenActivations(
                token=data['token'],
                position=data['position'] - start_pos,  # Adjust position
                top_features=top_features,
            )
        )

    # Build layer activations
    runtime_config = get_runtime_config()
    default_layer = runtime_config.get_default_layer()
    layer_activations = LayerActivations(
        layer=default_layer,
        hook_point=manager.get_hook_point(),
        token_activations=token_activations,
    )

    # Build final token list (excluding BOS if requested)
    output_tokens = token_strs[start_pos:]

    # Construct SAE ID string
    sae_id = f"{runtime_config.sae_type}/layer_{default_layer}_width_{runtime_config.sae_width}_l0_{runtime_config.sae_l0}"

    return AnalyzeResponse(
        prompt=prompt,
        tokens=output_tokens,
        layers=[layer_activations],
        model_name=settings.model_name,
        sae_id=sae_id,
    )


def get_feature_decoder_vector(feature_id: int) -> torch.Tensor:
    """
    Get the decoder vector for a specific feature.
    This vector can be used for steering.

    Args:
        feature_id: The feature index

    Returns:
        The decoder vector of shape (d_model,)
    """
    manager = get_model_manager()
    if not manager.is_loaded:
        manager.load_model()

    sae = manager.sae
    return sae.w_dec[feature_id]
