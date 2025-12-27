import logging
import traceback
import time
import uuid

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse
import json

logger = logging.getLogger(__name__)

from app.schemas.analysis import (
    AnalyzeRequest,
    AnalyzeResponse,
    GenerateRequest,
    GenerateResponse,
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
)
from app.inference.analysis import analyze_prompt
from app.inference.steering import (
    generate_with_steering,
    generate_streaming,
    generate_comparison,
)
from app.inference.model_loader import get_model_manager, AVAILABLE_LAYERS
from app.core.config import get_settings

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
        manager.load_model()
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


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Analyze a prompt and extract SAE feature activations.

    Returns per-token feature activations with top-K active features.
    """
    try:
        result = analyze_prompt(
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
    Generate text with optional steering.

    If include_baseline is True, generates both baseline and steered outputs.
    """
    try:
        if request.include_baseline and request.steering:
            baseline, steered = generate_comparison(
                prompt=request.prompt,
                steering_features=request.steering,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            return GenerateResponse(
                prompt=request.prompt,
                baseline_output=baseline,
                steered_output=steered,
                steering_config=request.steering,
            )
        else:
            output = generate_with_steering(
                prompt=request.prompt,
                steering_features=request.steering,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            return GenerateResponse(
                prompt=request.prompt,
                baseline_output=None,
                steered_output=output,
                steering_config=request.steering,
            )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """
    Generate text with streaming output via SSE.

    Each event contains a single token as it is generated.
    """

    async def event_generator():
        try:
            for token in generate_streaming(
                prompt=request.prompt,
                steering_features=request.steering,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            ):
                yield {
                    "event": "token",
                    "data": json.dumps({"token": token}),
                }
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
        "available_layers": AVAILABLE_LAYERS,
        "registry": manager._sae_registry.get_status() if manager._sae_registry else {},
    }


@router.post("/sae/load", response_model=LoadSAEResponse)
async def load_saes(request: LoadSAERequest):
    """Load SAEs for specified layers."""
    manager = get_model_manager()

    # Ensure model is loaded first
    if not manager.is_loaded:
        try:
            manager.load_model()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    try:
        result = manager.load_saes_for_layers(request.layers, request.width)
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
    settings = get_settings()

    # Ensure model is loaded
    if not manager.is_loaded:
        try:
            manager.load_model()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    try:
        # Get multi-layer activations
        result = manager.get_feature_activations_multi(
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
                hook_point=f"{settings.sae_type}/layer_{layer_idx}",
                token_activations=token_activations,
            ))

        return MultiLayerAnalyzeResponse(
            prompt=request.prompt,
            tokens=output_tokens,
            layers=layer_activations_list,
            model_name=settings.model_name,
            sae_width=settings.sae_width,
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
    settings = get_settings()
    start_time = time.time()

    # Ensure model is loaded
    if not manager.is_loaded:
        try:
            manager.load_model()
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
                    hook_point=f"{settings.sae_type}/layer_{layer_idx}",
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
        model_name=settings.model_name,
        sae_width=settings.sae_width,
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
    settings = get_settings()

    # Ensure model is loaded
    if not manager.is_loaded:
        try:
            manager.load_model()
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
                        "hook_point": f"{settings.sae_type}/layer_{layer_idx}",
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

        yield {
            "event": "complete",
            "data": json.dumps({
                "type": "complete",
                "total": total,
                "completed": completed,
            }),
        }

    return EventSourceResponse(event_generator())
