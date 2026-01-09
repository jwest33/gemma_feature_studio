from enum import Enum
from pydantic import BaseModel, Field


class FeatureActivation(BaseModel):
    id: int = Field(..., description="Feature index in the SAE")
    activation: float = Field(..., description="Activation strength")


class TokenActivations(BaseModel):
    token: str = Field(..., description="The token string")
    position: int = Field(..., description="Position in the sequence (0-indexed)")
    top_features: list[FeatureActivation] = Field(
        ..., description="Top-K active features for this token"
    )


class LayerActivations(BaseModel):
    layer: int = Field(..., description="Transformer layer index")
    hook_point: str = Field(..., description="Hook point name")
    token_activations: list[TokenActivations] = Field(
        ..., description="Activations per token"
    )


class AnalyzeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input text to analyze")
    top_k: int = Field(default=50, ge=1, le=500, description="Number of top features per token")
    include_bos: bool = Field(default=False, description="Include BOS token in analysis")


class AnalyzeResponse(BaseModel):
    prompt: str = Field(..., description="Original prompt")
    tokens: list[str] = Field(..., description="Tokenized prompt")
    layers: list[LayerActivations] = Field(..., description="Activations per layer")
    model_name: str = Field(..., description="Model used for analysis")
    sae_id: str = Field(..., description="SAE configuration used")


class SteeringFeature(BaseModel):
    feature_id: int = Field(..., ge=0, description="Feature index to steer")
    coefficient: float = Field(..., ge=-5.0, le=5.0, description="Steering strength")
    layer: int | None = Field(default=None, description="Optional layer override for this feature")


class NormalizationMode(str, Enum):
    """Normalization mode for steering."""
    none = "none"
    preserve_norm = "preserve_norm"
    clamp = "clamp"


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input prompt")
    steering: list[SteeringFeature] = Field(
        default=[], description="Features to steer during generation"
    )
    max_tokens: int = Field(default=64, ge=1, le=256)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    include_baseline: bool = Field(
        default=True, description="Also generate without steering for comparison"
    )
    normalization: NormalizationMode = Field(
        default=NormalizationMode.preserve_norm,
        description="Post-steering normalization mode: none, preserve_norm, or clamp"
    )
    norm_clamp_factor: float = Field(
        default=1.5, ge=1.0, le=3.0,
        description="For clamp mode, max allowed norm change ratio"
    )
    unit_normalize: bool = Field(
        default=False,
        description="Normalize decoder vectors to unit norm before applying"
    )


class GenerateResponse(BaseModel):
    prompt: str
    baseline_output: str | None = None
    steered_output: str
    steering_config: list[SteeringFeature]
    normalization: NormalizationMode = NormalizationMode.preserve_norm
    unit_normalize: bool = False


# ============================================================================
# Multi-Layer Analysis Schemas
# ============================================================================

class MultiLayerAnalyzeRequest(BaseModel):
    """Request for multi-layer SAE analysis."""
    prompt: str = Field(..., min_length=1, description="Input text to analyze")
    layers: list[int] = Field(
        default=[17],
        description="Layer indices to analyze (available: 9, 17, 22, 29 for Gemma-3-4B)"
    )
    top_k: int = Field(default=50, ge=1, le=500, description="Number of top features per token")
    include_bos: bool = Field(default=False, description="Include BOS token in analysis")


class MultiLayerAnalyzeResponse(BaseModel):
    """Response for multi-layer analysis."""
    prompt: str = Field(..., description="Original prompt")
    tokens: list[str] = Field(..., description="Tokenized prompt")
    layers: list[LayerActivations] = Field(..., description="Activations per analyzed layer")
    model_name: str = Field(..., description="Model used for analysis")
    sae_width: str = Field(..., description="SAE width used")
    analyzed_layers: list[int] = Field(..., description="Layer indices that were analyzed")


# ============================================================================
# Batch Analysis Schemas
# ============================================================================

class BatchAnalyzeRequest(BaseModel):
    """Request for batch prompt analysis."""
    prompts: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of prompts to analyze"
    )
    layers: list[int] = Field(
        default=[17],
        description="Layer indices to analyze"
    )
    top_k: int = Field(default=50, ge=1, le=500, description="Number of top features per token")
    include_bos: bool = Field(default=False, description="Include BOS token in analysis")


class PromptAnalysisResult(BaseModel):
    """Analysis result for a single prompt within a batch."""
    id: str = Field(..., description="Unique ID for this prompt result")
    prompt: str = Field(..., description="Original prompt")
    tokens: list[str] = Field(..., description="Tokenized prompt")
    layers: list[LayerActivations] = Field(..., description="Activations per layer")
    status: str = Field(default="success", description="'success' or 'error'")
    error: str | None = Field(default=None, description="Error message if failed")


class BatchAnalyzeResponse(BaseModel):
    """Response for batch prompt analysis."""
    results: list[PromptAnalysisResult] = Field(..., description="Analysis results per prompt")
    model_name: str = Field(..., description="Model used for analysis")
    sae_width: str = Field(..., description="SAE width used")
    analyzed_layers: list[int] = Field(..., description="Layer indices analyzed")
    total_prompts: int = Field(..., description="Total number of prompts processed")
    successful: int = Field(..., description="Number of successful analyses")
    failed: int = Field(..., description="Number of failed analyses")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")


# ============================================================================
# VRAM and SAE Management Schemas
# ============================================================================

class VRAMStatus(BaseModel):
    """Current VRAM usage status."""
    total_gb: float = Field(..., description="Total VRAM in GB")
    allocated_gb: float = Field(..., description="Allocated VRAM in GB")
    reserved_gb: float = Field(..., description="Reserved VRAM in GB")
    free_gb: float = Field(..., description="Free VRAM in GB")
    pressure: str = Field(..., description="Memory pressure level: low, moderate, high, critical")


class SAEStatus(BaseModel):
    """Status of a loaded SAE."""
    layer: int = Field(..., description="Layer index")
    width: str = Field(..., description="SAE width (16k, 65k, 262k, 1m)")
    l0: str = Field(..., description="L0 regularization level")
    sae_type: str = Field(..., description="SAE hook type")
    sae_id: str = Field(..., description="Full SAE identifier")
    size_mb: float = Field(..., description="Size in MB")
    loaded_at: float = Field(..., description="Unix timestamp when loaded")
    last_used: float = Field(..., description="Unix timestamp when last used")


class SAERegistryStatus(BaseModel):
    """Status of all loaded SAEs."""
    loaded_count: int = Field(..., description="Number of loaded SAEs")
    loaded_layers: list[int] = Field(..., description="List of loaded layer indices")
    total_size_mb: float = Field(..., description="Total size of all SAEs in MB")
    total_size_gb: float = Field(..., description="Total size of all SAEs in GB")
    max_budget_gb: float = Field(..., description="Maximum allowed SAE memory budget")
    entries: list[SAEStatus] = Field(default=[], description="Details of each loaded SAE")


class SystemStatus(BaseModel):
    """Combined system status including model, VRAM, and SAEs."""
    model_loaded: bool = Field(..., description="Whether the main model is loaded")
    model_name: str | None = Field(default=None, description="Name of loaded model")
    vram: VRAMStatus = Field(..., description="VRAM status")
    saes: SAERegistryStatus = Field(..., description="SAE registry status")
    available_layers: list[int] = Field(
        default=[9, 17, 22, 29],
        description="Available layer indices for analysis"
    )


class PreflightRequest(BaseModel):
    """Request to check if SAEs can be loaded."""
    layers: list[int] = Field(..., description="Layers to check")
    width: str = Field(default="65k", description="SAE width")


class PreflightResponse(BaseModel):
    """Response for preflight check."""
    can_load: bool = Field(..., description="Whether all requested SAEs can be loaded")
    layers_to_load: list[int] = Field(..., description="Layers that need to be loaded")
    already_loaded: list[int] = Field(..., description="Layers already loaded")
    bytes_needed: int = Field(..., description="Total bytes needed for new SAEs")
    bytes_available: int = Field(..., description="Currently available bytes")
    recommendation: str | None = Field(default=None, description="Recommendation if cannot load")


class LoadSAERequest(BaseModel):
    """Request to load SAEs for specific layers."""
    layers: list[int] = Field(..., description="Layers to load SAEs for")
    width: str = Field(default="65k", description="SAE width")


class LoadSAEResponse(BaseModel):
    """Response after loading SAEs."""
    loaded: list[int] = Field(..., description="Layers successfully loaded")
    already_loaded: list[int] = Field(..., description="Layers that were already loaded")
    failed: list[dict] = Field(default=[], description="Layers that failed to load with errors")
    vram_status: VRAMStatus = Field(..., description="Current VRAM status after loading")


class UnloadSAERequest(BaseModel):
    """Request to unload SAE for a specific layer."""
    layer: int = Field(..., description="Layer to unload")
    width: str = Field(default="65k", description="SAE width")


class UnloadSAEResponse(BaseModel):
    """Response after unloading SAE."""
    success: bool = Field(..., description="Whether unload was successful")
    layer: int = Field(..., description="Layer that was unloaded")
    vram_status: VRAMStatus = Field(..., description="Current VRAM status after unloading")


# ============================================================================
# Neuronpedia Feature Info Schemas
# ============================================================================

class NeuronpediaActivation(BaseModel):
    """A single activation example from Neuronpedia."""
    tokens: list[str] = Field(default=[], description="Tokens in the activation context")
    values: list[float] = Field(default=[], description="Activation values per token")
    maxValue: float = Field(default=0.0, description="Maximum activation value")
    maxTokenIndex: int = Field(default=0, description="Index of max activation token")


class NeuronpediaExplanation(BaseModel):
    """Auto-interpretation explanation from Neuronpedia."""
    description: str = Field(default="", description="Human-readable explanation")
    explanationType: str = Field(default="", description="Type of explanation method")
    typeName: str | None = Field(default=None, description="Friendly type name")
    explanationModelName: str | None = Field(default=None, description="Model used for explanation")
    score: float | None = Field(default=None, description="Confidence score")


class NeuronpediaFeatureRequest(BaseModel):
    """Request to get feature info from Neuronpedia."""
    feature_id: int = Field(..., ge=0, description="Feature index in the SAE")
    layer: int = Field(..., description="Layer index")


class NeuronpediaFeatureResponse(BaseModel):
    """Response containing Neuronpedia feature information."""
    modelId: str = Field(..., description="Neuronpedia model identifier")
    layer: str = Field(..., description="Neuronpedia layer/SAE identifier")
    index: int = Field(..., description="Feature index")
    description: str | None = Field(default=None, description="Feature description/explanation")
    explanations: list[NeuronpediaExplanation] = Field(
        default=[], description="Auto-interpretation explanations"
    )
    activations: list[NeuronpediaActivation] = Field(
        default=[], description="Example activations"
    )
    neuronpedia_url: str = Field(..., description="URL to view on Neuronpedia")
    hasData: bool = Field(default=True, description="Whether feature data exists")


# ============================================================================
# Model Configuration Schemas
# ============================================================================

class ConfigureModelRequest(BaseModel):
    """Request to configure model and SAE settings."""
    model_name: str = Field(..., description="HuggingFace model ID or local path")
    model_size: str = Field(default="4b", description="Model size (270m, 1b, 4b, 12b, 27b)")
    sae_repo: str = Field(..., description="HuggingFace SAE repository ID")
    sae_width: str = Field(default="65k", description="SAE width (16k, 65k, 262k, 1m)")
    sae_l0: str = Field(default="medium", description="L0 regularization level")
    sae_type: str = Field(default="resid_post", description="SAE hook type")
    memory_saver_mode: bool | None = Field(
        default=None,
        description="If true, unload SAEs immediately after encoding to save VRAM. "
                    "SAEs will be reloaded automatically when needed for steering."
    )


class ConfigureModelResponse(BaseModel):
    """Response after configuring model settings."""
    status: str = Field(..., description="Configuration status")
    message: str = Field(..., description="Status message")
    config: dict = Field(..., description="Current configuration")
    requires_reload: bool = Field(
        default=False,
        description="Whether model needs to be reloaded for changes to take effect"
    )


# ============================================================================
# SAE Cache Status Schemas
# ============================================================================

class SAECacheStatusRequest(BaseModel):
    """Request to check SAE disk cache status."""
    layers: list[int] = Field(..., description="Layers to check cache status for")


class SAELayerCacheStatus(BaseModel):
    """Cache status for a single SAE layer."""
    layer: int = Field(..., description="Layer index")
    cached: bool = Field(..., description="Whether the SAE is cached on disk")
    filename: str = Field(..., description="SAE filename in the repo")


class SAECacheStatusResponse(BaseModel):
    """Response with SAE disk cache status."""
    layers: list[SAELayerCacheStatus] = Field(..., description="Cache status per layer")
    all_cached: bool = Field(..., description="Whether all requested layers are cached")
    uncached_count: int = Field(..., description="Number of uncached layers")
    sae_repo: str = Field(..., description="SAE repository being checked")


class SAEDownloadRequest(BaseModel):
    """Request to download SAE files to local cache."""
    layers: list[int] = Field(..., description="Layers to download SAEs for")


class SAEDownloadFailure(BaseModel):
    """Details of a failed SAE download."""
    layer: int = Field(..., description="Layer that failed to download")
    error: str = Field(..., description="Error message")


class SAEDownloadResponse(BaseModel):
    """Response after downloading SAE files."""
    downloaded: list[int] = Field(..., description="Layers successfully downloaded")
    already_cached: list[int] = Field(..., description="Layers that were already cached")
    failed: list[SAEDownloadFailure] = Field(default=[], description="Layers that failed to download")
    sae_repo: str = Field(..., description="SAE repository")
