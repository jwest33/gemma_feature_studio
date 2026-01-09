import logging
import traceback
import time
import uuid
import asyncio

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse
import json
import httpx

logger = logging.getLogger(__name__)

from app.schemas.analysis import (
    AnalyzeRequest,
    AnalyzeResponse,
    GenerateRequest,
    GenerateResponse,
    NormalizationMode,
    # Multi-layer analysis
    MultiLayerAnalyzeRequest,
    MultiLayerAnalyzeResponse,
    # Batch analysis
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    PromptAnalysisResult,
    # VRAM and SAE management
    VRAMStatus,
    SystemStatus,
    SAERegistryStatus,
    PreflightRequest,
    PreflightResponse,
    LoadSAERequest,
    LoadSAEResponse,
    UnloadSAERequest,
    UnloadSAEResponse,
    # Data types
    LayerActivations,
    TokenActivations,
    FeatureActivation,
    # Neuronpedia
    NeuronpediaFeatureRequest,
    NeuronpediaFeatureResponse,
    NeuronpediaActivation,
    NeuronpediaExplanation,
    # Model configuration
    ConfigureModelRequest,
    ConfigureModelResponse,
    # SAE cache status
    SAECacheStatusRequest,
    SAECacheStatusResponse,
    SAELayerCacheStatus,
    SAEDownloadRequest,
    SAEDownloadResponse,
    SAEDownloadFailure,
)
from app.inference.analysis import analyze_prompt
from app.inference.steering import (
    generate_with_steering,
    generate_streaming,
    generate_comparison,
)
from app.inference.model_loader import get_model_manager, get_available_layers
from app.core.config import get_settings, get_runtime_config

router = APIRouter(prefix="/api", tags=["features"])


@router.get("/health")
async def health_check():
    """Check API health and model status."""
    manager = get_model_manager()
    return {
        "status": "healthy",
        "model_loaded": manager.is_loaded,
    }


@router.post("/load")
async def load_model():
    """Explicitly load the model and SAE."""
    manager = get_model_manager()
    try:
        # Run blocking model load in thread pool to not block event loop
        await asyncio.to_thread(manager.load_model)
        return {"status": "loaded", "message": "Model and SAE loaded successfully"}
    except Exception as e:
        print(f"Failed to load model: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unload")
async def unload_model():
    """Unload the model to free GPU memory."""
    manager = get_model_manager()
    manager.unload_model()
    return {"status": "unloaded", "message": "Model unloaded from memory"}


# =============================================================================
# Model Configuration Endpoints
# =============================================================================

@router.post("/config", response_model=ConfigureModelResponse)
async def configure_model(request: ConfigureModelRequest):
    """
    Configure model and SAE settings at runtime.

    This updates the runtime configuration without reloading the model.
    If the model name changes, the model will need to be reloaded.
    SAE changes take effect on the next SAE load.
    """
    runtime_config = get_runtime_config()
    manager = get_model_manager()

    # Check if model was previously loaded
    was_loaded = manager.is_loaded

    # Update runtime config with explicit model size
    requires_reload = runtime_config.update(
        model_name=request.model_name,
        model_size=request.model_size,
        sae_repo=request.sae_repo,
        sae_width=request.sae_width,
        sae_l0=request.sae_l0,
        sae_type=request.sae_type,
        memory_saver_mode=request.memory_saver_mode,
    )

    # If model name changed and model was loaded, unload it
    if requires_reload and was_loaded:
        manager.unload_model()
        message = "Configuration updated. Model unloaded due to model change - will reload on next analysis."
    elif requires_reload:
        message = "Configuration updated. New model will be used on next load."
    else:
        # Clear SAE registry if SAE config changed (width, l0, type, or repo)
        if manager._sae_registry:
            manager._sae_registry.clear_all()
        message = "Configuration updated. SAE cache cleared - new SAEs will be loaded on next analysis."

    return ConfigureModelResponse(
        status="configured",
        message=message,
        config=runtime_config.to_dict(),
        requires_reload=requires_reload,
    )


@router.get("/config")
async def get_config():
    """Get current runtime configuration."""
    runtime_config = get_runtime_config()
    manager = get_model_manager()

    return {
        "config": runtime_config.to_dict(),
        "model_loaded": manager.is_loaded,
        "loaded_layers": manager.get_loaded_layers() if manager.is_loaded else [],
    }


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Analyze a prompt and extract SAE feature activations.

    Returns per-token feature activations with top-K active features.
    """
    try:
        # Run blocking analysis in thread pool to not block event loop
        result = await asyncio.to_thread(
            analyze_prompt,
            prompt=request.prompt,
            top_k=request.top_k,
            include_bos=request.include_bos,
        )
        return result
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text with optional steering and norm preservation.

    If include_baseline is True, generates both baseline and steered outputs.
    Supports three normalization modes:
    - none: Raw steering without normalization
    - preserve_norm: Rescale output to maintain original activation norm
    - clamp: Allow bounded norm changes within a factor
    """
    try:
        if request.include_baseline and request.steering:
            baseline, steered = generate_comparison(
                prompt=request.prompt,
                steering_features=request.steering,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                normalization=request.normalization,
                norm_clamp_factor=request.norm_clamp_factor,
                unit_normalize=request.unit_normalize,
            )
            return GenerateResponse(
                prompt=request.prompt,
                baseline_output=baseline,
                steered_output=steered,
                steering_config=request.steering,
                normalization=request.normalization,
                unit_normalize=request.unit_normalize,
            )
        else:
            output = generate_with_steering(
                prompt=request.prompt,
                steering_features=request.steering,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                normalization=request.normalization,
                norm_clamp_factor=request.norm_clamp_factor,
                unit_normalize=request.unit_normalize,
            )
            return GenerateResponse(
                prompt=request.prompt,
                baseline_output=None,
                steered_output=output,
                steering_config=request.steering,
                normalization=request.normalization,
                unit_normalize=request.unit_normalize,
            )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """
    Generate text with streaming output via SSE and norm preservation.

    Each event contains a single token as it is generated.
    Supports three normalization modes for steering.
    """

    async def event_generator():
        try:
            for token in generate_streaming(
                prompt=request.prompt,
                steering_features=request.steering,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                normalization=request.normalization,
                norm_clamp_factor=request.norm_clamp_factor,
                unit_normalize=request.unit_normalize,
            ):
                yield {
                    "event": "token",
                    "data": json.dumps({"token": token}),
                }
                # Yield control to event loop to flush SSE response
                await asyncio.sleep(0)
            yield {
                "event": "done",
                "data": json.dumps({"status": "complete"}),
            }
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            logger.error(traceback.format_exc())
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}),
            }

    return EventSourceResponse(event_generator())


@router.get("/model/info")
async def model_info():
    """Get information about the loaded model and SAE."""
    manager = get_model_manager()

    if not manager.is_loaded:
        return {
            "loaded": False,
            "message": "Model not loaded. Call POST /api/load first.",
        }

    return {
        "loaded": True,
        "sae_width": manager.get_sae_width(),
        "hook_point": manager.get_hook_point(),
    }


# =============================================================================
# VRAM and System Status Endpoints
# =============================================================================

@router.get("/vram/status")
async def vram_status():
    """Get current VRAM usage status."""
    manager = get_model_manager()
    return manager.get_vram_status()


@router.get("/system/status")
async def system_status():
    """Get complete system status including model, VRAM, and loaded SAEs."""
    manager = get_model_manager()
    return manager.get_system_status()


@router.post("/vram/preflight", response_model=PreflightResponse)
async def preflight_check(request: PreflightRequest):
    """Check if SAEs for requested layers can be loaded."""
    manager = get_model_manager()
    result = manager.preflight_check(request.layers, request.width)
    return PreflightResponse(**result)


# =============================================================================
# SAE Management Endpoints
# =============================================================================

@router.get("/sae/loaded")
async def get_loaded_saes():
    """Get list of currently loaded SAEs."""
    manager = get_model_manager()
    return {
        "layers": manager.get_loaded_layers(),
        "available_layers": get_available_layers(),
        "registry": manager._sae_registry.get_status() if manager._sae_registry else {},
    }


@router.post("/sae/load", response_model=LoadSAEResponse)
async def load_saes(request: LoadSAERequest):
    """Load SAEs for specified layers."""
    manager = get_model_manager()

    # Ensure model is loaded first
    if not manager.is_loaded:
        try:
            # Run blocking model load in thread pool to not block event loop
            await asyncio.to_thread(manager.load_model)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    try:
        # Run blocking SAE load in thread pool to not block event loop
        result = await asyncio.to_thread(manager.load_saes_for_layers, request.layers, request.width)
        return LoadSAEResponse(
            loaded=result["loaded"],
            already_loaded=result["already_loaded"],
            failed=result["failed"],
            vram_status=VRAMStatus(**result["vram_status"]),
        )
    except Exception as e:
        logger.error(f"Failed to load SAEs: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sae/unload", response_model=UnloadSAEResponse)
async def unload_sae(request: UnloadSAERequest):
    """Unload SAE for a specific layer."""
    manager = get_model_manager()
    success = manager.unload_sae_for_layer(request.layer, request.width)
    return UnloadSAEResponse(
        success=success,
        layer=request.layer,
        vram_status=VRAMStatus(**manager.get_vram_status()),
    )


@router.post("/sae/cache/status", response_model=SAECacheStatusResponse)
async def get_sae_cache_status(request: SAECacheStatusRequest):
    """Check if SAE files are cached locally in HuggingFace cache.

    This checks disk cache, not VRAM. Use /sae/loaded to check VRAM status.
    """
    manager = get_model_manager()
    runtime_config = get_runtime_config()

    try:
        cache_status = manager.get_sae_cache_status(request.layers)
        uncached = [s for s in cache_status if not s["cached"]]

        return SAECacheStatusResponse(
            layers=[SAELayerCacheStatus(**s) for s in cache_status],
            all_cached=len(uncached) == 0,
            uncached_count=len(uncached),
            sae_repo=runtime_config.sae_repo,
        )
    except Exception as e:
        logger.error(f"Failed to check SAE cache status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sae/download", response_model=SAEDownloadResponse)
async def download_saes(request: SAEDownloadRequest):
    """Download SAE files to local HuggingFace cache without loading to GPU.

    Use this to pre-download SAEs before analysis.
    """
    manager = get_model_manager()
    runtime_config = get_runtime_config()

    try:
        # Run blocking download in thread pool to not block event loop
        result = await asyncio.to_thread(manager.download_sae_files, request.layers)

        return SAEDownloadResponse(
            downloaded=result["downloaded"],
            already_cached=result["already_cached"],
            failed=[SAEDownloadFailure(**f) for f in result["failed"]],
            sae_repo=runtime_config.sae_repo,
        )
    except Exception as e:
        logger.error(f"Failed to download SAEs: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Multi-Layer Analysis Endpoints
# =============================================================================

@router.post("/analyze/multi-layer", response_model=MultiLayerAnalyzeResponse)
async def analyze_multi_layer(request: MultiLayerAnalyzeRequest):
    """
    Analyze a prompt across multiple SAE layers.

    Returns feature activations for each token at each selected layer.
    """
    manager = get_model_manager()
    runtime_config = get_runtime_config()

    # Ensure model is loaded
    if not manager.is_loaded:
        try:
            # Run blocking model load in thread pool to not block event loop
            await asyncio.to_thread(manager.load_model)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    try:
        # Run blocking analysis in thread pool to not block event loop
        result = await asyncio.to_thread(
            manager.get_feature_activations_multi,
            text=request.prompt,
            layers=request.layers,
            top_k=request.top_k,
        )

        # Determine start position (skip BOS if requested)
        start_pos = 0 if request.include_bos else 1
        output_tokens = result['tokens'][start_pos:]

        # Build layer activations list
        layer_activations_list = []
        for layer_idx in request.layers:
            if layer_idx not in result['layers']:
                continue

            raw_token_activations = result['layers'][layer_idx]

            token_activations = []
            for data in raw_token_activations[start_pos:]:
                top_features = [
                    FeatureActivation(id=f['id'], activation=f['activation'])
                    for f in data['top_features']
                    if f['activation'] > 0
                ]
                token_activations.append(TokenActivations(
                    token=data['token'],
                    position=data['position'] - start_pos,
                    top_features=top_features,
                ))

            layer_activations_list.append(LayerActivations(
                layer=layer_idx,
                hook_point=f"{runtime_config.sae_type}/layer_{layer_idx}",
                token_activations=token_activations,
            ))

        return MultiLayerAnalyzeResponse(
            prompt=request.prompt,
            tokens=output_tokens,
            layers=layer_activations_list,
            model_name=runtime_config.model_name,
            sae_width=runtime_config.sae_width,
            analyzed_layers=[la.layer for la in layer_activations_list],
        )

    except Exception as e:
        logger.error(f"Multi-layer analysis failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Batch Analysis Endpoints
# =============================================================================

@router.post("/analyze/batch", response_model=BatchAnalyzeResponse)
async def analyze_batch(request: BatchAnalyzeRequest):
    """
    Analyze multiple prompts and extract SAE feature activations.

    Processes prompts sequentially to manage GPU memory.
    """
    manager = get_model_manager()
    runtime_config = get_runtime_config()
    start_time = time.time()

    # Ensure model is loaded
    if not manager.is_loaded:
        try:
            # Run blocking model load in thread pool to not block event loop
            await asyncio.to_thread(manager.load_model)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    results = []
    successful = 0
    failed = 0

    for prompt in request.prompts:
        prompt_id = str(uuid.uuid4())
        try:
            # Get multi-layer activations
            result = manager.get_feature_activations_multi(
                text=prompt,
                layers=request.layers,
                top_k=request.top_k,
            )

            start_pos = 0 if request.include_bos else 1
            output_tokens = result['tokens'][start_pos:]

            # Build layer activations
            layer_activations_list = []
            for layer_idx in request.layers:
                if layer_idx not in result['layers']:
                    continue

                raw_token_activations = result['layers'][layer_idx]
                token_activations = []
                for data in raw_token_activations[start_pos:]:
                    top_features = [
                        FeatureActivation(id=f['id'], activation=f['activation'])
                        for f in data['top_features']
                        if f['activation'] > 0
                    ]
                    token_activations.append(TokenActivations(
                        token=data['token'],
                        position=data['position'] - start_pos,
                        top_features=top_features,
                    ))

                layer_activations_list.append(LayerActivations(
                    layer=layer_idx,
                    hook_point=f"{runtime_config.sae_type}/layer_{layer_idx}",
                    token_activations=token_activations,
                ))

            results.append(PromptAnalysisResult(
                id=prompt_id,
                prompt=prompt,
                tokens=output_tokens,
                layers=layer_activations_list,
                status="success",
            ))
            successful += 1

        except Exception as e:
            logger.error(f"Batch analysis failed for prompt: {e}")
            results.append(PromptAnalysisResult(
                id=prompt_id,
                prompt=prompt,
                tokens=[],
                layers=[],
                status="error",
                error=str(e),
            ))
            failed += 1

    processing_time = (time.time() - start_time) * 1000

    return BatchAnalyzeResponse(
        results=results,
        model_name=runtime_config.model_name,
        sae_width=runtime_config.sae_width,
        analyzed_layers=request.layers,
        total_prompts=len(request.prompts),
        successful=successful,
        failed=failed,
        processing_time_ms=processing_time,
    )


@router.post("/analyze/batch/stream")
async def analyze_batch_stream(request: BatchAnalyzeRequest):
    """
    Analyze multiple prompts with streaming progress updates via SSE.
    """
    manager = get_model_manager()
    runtime_config = get_runtime_config()

    # Ensure model is loaded
    if not manager.is_loaded:
        try:
            # Run blocking model load in thread pool to not block event loop
            await asyncio.to_thread(manager.load_model)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    async def event_generator():
        total = len(request.prompts)
        completed = 0

        for idx, prompt in enumerate(request.prompts):
            # Emit progress event
            yield {
                "event": "progress",
                "data": json.dumps({
                    "type": "progress",
                    "prompt_index": idx,
                    "total": total,
                    "completed": completed,
                }),
            }
            await asyncio.sleep(0)  # Flush SSE

            prompt_id = str(uuid.uuid4())
            try:
                # Get multi-layer activations
                result = manager.get_feature_activations_multi(
                    text=prompt,
                    layers=request.layers,
                    top_k=request.top_k,
                )

                start_pos = 0 if request.include_bos else 1
                output_tokens = result['tokens'][start_pos:]

                # Build layer activations
                layer_activations_list = []
                for layer_idx in request.layers:
                    if layer_idx not in result['layers']:
                        continue

                    raw_token_activations = result['layers'][layer_idx]
                    token_activations = []
                    for data in raw_token_activations[start_pos:]:
                        top_features = [
                            {"id": f['id'], "activation": f['activation']}
                            for f in data['top_features']
                            if f['activation'] > 0
                        ]
                        token_activations.append({
                            "token": data['token'],
                            "position": data['position'] - start_pos,
                            "top_features": top_features,
                        })

                    layer_activations_list.append({
                        "layer": layer_idx,
                        "hook_point": f"{runtime_config.sae_type}/layer_{layer_idx}",
                        "token_activations": token_activations,
                    })

                prompt_result = {
                    "id": prompt_id,
                    "prompt": prompt,
                    "tokens": output_tokens,
                    "layers": layer_activations_list,
                    "status": "success",
                    "error": None,
                }

            except Exception as e:
                logger.error(f"Batch stream analysis failed for prompt: {e}")
                prompt_result = {
                    "id": prompt_id,
                    "prompt": prompt,
                    "tokens": [],
                    "layers": [],
                    "status": "error",
                    "error": str(e),
                }

            completed += 1
            yield {
                "event": "result",
                "data": json.dumps({
                    "type": "result",
                    "prompt_index": idx,
                    "prompt_id": prompt_id,
                    "total": total,
                    "completed": completed,
                    "result": prompt_result,
                }),
            }
            await asyncio.sleep(0)  # Flush SSE

        yield {
            "event": "complete",
            "data": json.dumps({
                "type": "complete",
                "total": total,
                "completed": completed,
            }),
        }

    return EventSourceResponse(event_generator())


# =============================================================================
# Neuronpedia Feature Lookup Endpoints
# =============================================================================

def build_neuronpedia_layer_id(layer: int, sae_type: str, width: str) -> str:
    """
    Build the Neuronpedia layer/SAE identifier.

    Neuronpedia uses format like: "9-gemmascope-res-65k"
    Adjust this based on how Neuronpedia names Gemma Scope SAEs.
    """
    # Map our sae_type to Neuronpedia naming
    type_map = {
        "resid_post": "res",
        "attn_out": "att",
        "mlp_out": "mlp",
        "transcoder": "tc",
    }
    short_type = type_map.get(sae_type, "res")

    # Gemma Scope 2 SAEs on Neuronpedia use format: layer-gemmascope-type-width
    return f"{layer}-gemmascope-2-{short_type}-{width}"


@router.post("/neuronpedia/feature", response_model=NeuronpediaFeatureResponse)
async def get_neuronpedia_feature(request: NeuronpediaFeatureRequest):
    """
    Fetch feature information from Neuronpedia.

    Returns feature explanations and example activations from Neuronpedia's database.
    """
    settings = get_settings()
    runtime_config = get_runtime_config()

    if not settings.neuronpedia_api_key:
        raise HTTPException(
            status_code=500,
            detail="Neuronpedia API key not configured. Add NEURONPEDIA_API_KEY to .env file."
        )

    # Build the Neuronpedia identifiers - use runtime config for model-specific values
    model_id = runtime_config.neuronpedia_model_id
    layer_id = build_neuronpedia_layer_id(request.layer, runtime_config.sae_type, runtime_config.sae_width)

    # Construct the API URL
    api_url = f"{settings.neuronpedia_base_url}/feature/{model_id}/{layer_id}/{request.feature_id}"
    neuronpedia_url = f"https://www.neuronpedia.org/{model_id}/{layer_id}/{request.feature_id}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                api_url,
                headers={
                    "x-api-key": settings.neuronpedia_api_key,
                    "Accept": "application/json",
                }
            )

            if response.status_code == 404:
                # Feature not found in Neuronpedia
                return NeuronpediaFeatureResponse(
                    modelId=model_id,
                    layer=layer_id,
                    index=request.feature_id,
                    description=None,
                    explanations=[],
                    activations=[],
                    neuronpedia_url=neuronpedia_url,
                    hasData=False,
                )

            response.raise_for_status()
            data = response.json()

            # Parse explanations
            explanations = []
            if "explanations" in data and data["explanations"]:
                for exp in data["explanations"]:
                    explanations.append(NeuronpediaExplanation(
                        description=exp.get("description", ""),
                        explanationType=exp.get("explanationType", ""),
                        typeName=exp.get("typeName"),
                        explanationModelName=exp.get("explanationModelName"),
                        score=exp.get("score"),
                    ))

            # Parse activations
            activations = []
            if "activations" in data and data["activations"]:
                for act in data["activations"]:
                    activations.append(NeuronpediaActivation(
                        tokens=act.get("tokens", []),
                        values=act.get("values", []),
                        maxValue=act.get("maxValue", 0.0),
                        maxTokenIndex=act.get("maxTokenIndex", 0),
                    ))

            # Get the primary description (first explanation)
            description = None
            if explanations:
                description = explanations[0].description

            return NeuronpediaFeatureResponse(
                modelId=model_id,
                layer=layer_id,
                index=request.feature_id,
                description=description,
                explanations=explanations,
                activations=activations,
                neuronpedia_url=neuronpedia_url,
                hasData=True,
            )

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Neuronpedia API request timed out")
    except httpx.HTTPStatusError as e:
        logger.error(f"Neuronpedia API error: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Neuronpedia API error: {e}")
    except Exception as e:
        logger.error(f"Failed to fetch from Neuronpedia: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to fetch feature info: {str(e)}")
