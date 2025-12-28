"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import {
  useFlowStore,
  usePanelConfig,
  useSteeringFeatures,
  useSteeringNormalization,
  useSteeringClampFactor,
  useSteeringUnitNormalize,
} from "@/state/flowStore";
import { LayerSelector, VRAMMonitor, PromptManager, ModelSelector } from "@/components/controls";
import { FlowVisualization, FeatureInspector, ResizablePanel } from "@/components/flow";
import { GlobalSteeringPanel } from "@/components/GlobalSteeringPanel";
import { analyzeMultiLayer, loadSAEs, generateTextStream, configureModel, getConfig } from "@/lib/api";
import { SAE_PRESETS } from "@/types/flow";
import type { GenerateRequest } from "@/types/analysis";

export default function Home() {
  const {
    prompts,
    selectedLayers,
    isAnalyzing,
    analysisProgress,
    analysisError,
    setAnalyzing,
    setAnalysisProgress,
    setAnalysisError,
    updatePromptAnalysis,
    selectOutput,
    modelConfig,
    getSaePreset,
    setModelPath,
  } = useFlowStore();

  const activePrompt = useFlowStore((state) => state.getActivePrompt());
  const panelConfig = usePanelConfig();
  const [isConfiguring, setIsConfiguring] = useState(false);

  // Global steering state
  const steeringFeatures = useSteeringFeatures();
  const steeringNormalization = useSteeringNormalization();
  const steeringClampFactor = useSteeringClampFactor();
  const steeringUnitNormalize = useSteeringUnitNormalize();

  const [generatedOutput, setGeneratedOutput] = useState<string | undefined>();
  const [baselineOutput, setBaselineOutput] = useState<string | undefined>();
  const [isGenerating, setIsGenerating] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Fetch backend config on mount to sync model path from .env
  useEffect(() => {
    getConfig()
      .then((config) => {
        if (config.config.model_name) {
          setModelPath(config.config.model_name);
        }
      })
      .catch((err) => {
        console.error("Failed to fetch backend config:", err);
      });
  }, [setModelPath]);

  // Generate LLM response for a prompt (no steering)
  const generateResponse = useCallback(async (promptText: string) => {
    // Cancel any existing generation
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    abortControllerRef.current = new AbortController();
    setIsGenerating(true);
    setGeneratedOutput("");
    setBaselineOutput(undefined);

    try {
      let output = "";
      for await (const token of generateTextStream(
        { prompt: promptText, steering: [], max_tokens: 256 },
        abortControllerRef.current.signal
      )) {
        output += token;
        setGeneratedOutput(output);
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        console.error("Generation error:", e);
      }
    } finally {
      setIsGenerating(false);
      abortControllerRef.current = null;
    }
  }, []);

  // Generate with steering for comparison (uses global steering state)
  const handleGenerateWithSteering = useCallback(async () => {
    if (!activePrompt) return;

    // Get enabled features with non-zero coefficients
    const enabledFeatures = steeringFeatures.filter(
      (f) => f.enabled && f.coefficient !== 0
    );

    if (enabledFeatures.length === 0) return;

    // Cancel any existing generation
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    abortControllerRef.current = new AbortController();
    setIsGenerating(true);
    setGeneratedOutput("");

    const promptText = activePrompt.text;

    try {
      // Only generate baseline if we don't already have one
      if (!baselineOutput) {
        let baseline = "";
        const baselineRequest: GenerateRequest = {
          prompt: promptText,
          steering: [],
          max_tokens: 256,
          normalization: "none",
        };

        for await (const token of generateTextStream(
          baselineRequest,
          abortControllerRef.current.signal
        )) {
          baseline += token;
          setBaselineOutput(baseline);
        }
      }

      // Generate steered output with all enabled features
      let steered = "";
      const steeredRequest: GenerateRequest = {
        prompt: promptText,
        steering: enabledFeatures.map((f) => ({
          feature_id: f.id,
          coefficient: f.coefficient,
          layer: f.layer,
        })),
        max_tokens: 256,
        normalization: steeringNormalization,
        norm_clamp_factor: steeringClampFactor,
        unit_normalize: steeringUnitNormalize,
      };

      for await (const token of generateTextStream(
        steeredRequest,
        abortControllerRef.current.signal
      )) {
        steered += token;
        setGeneratedOutput(steered);
      }

      // Show output in inspector
      selectOutput();

    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        console.error("Steered generation error:", e);
      }
    } finally {
      setIsGenerating(false);
      abortControllerRef.current = null;
    }
  }, [activePrompt, baselineOutput, steeringFeatures, steeringNormalization, steeringClampFactor, steeringUnitNormalize, selectOutput]);

  // Handle model configuration changes
  const handleConfigChange = useCallback(async () => {
    setIsConfiguring(true);
    try {
      const preset = getSaePreset();
      if (!preset) return;

      await configureModel({
        model_name: modelConfig.modelPath,
        sae_repo: preset.repo,
        sae_width: preset.width,
        sae_l0: preset.l0,
        sae_type: preset.type,
      });
    } catch (e) {
      console.error("Failed to configure model:", e);
      setAnalysisError((e as Error).message);
    } finally {
      setIsConfiguring(false);
    }
  }, [modelConfig, getSaePreset, setAnalysisError]);

  // Analyze all prompts
  const handleAnalyze = useCallback(async () => {
    if (prompts.length === 0 || selectedLayers.length === 0) return;

    setAnalyzing(true);
    setAnalysisError(null);
    setGeneratedOutput(undefined);
    setBaselineOutput(undefined);

    try {
      // First, ensure model config is applied
      const preset = getSaePreset();
      if (preset) {
        await configureModel({
          model_name: modelConfig.modelPath,
          sae_repo: preset.repo,
          sae_width: preset.width,
          sae_l0: preset.l0,
          sae_type: preset.type,
        });
      }

      // Then, ensure SAEs are loaded for selected layers
      await loadSAEs({ layers: selectedLayers });

      // Analyze each prompt
      for (let i = 0; i < prompts.length; i++) {
        setAnalysisProgress(i + 1, prompts.length);
        const prompt = prompts[i];

        const response = await analyzeMultiLayer({
          prompt: prompt.text,
          layers: selectedLayers,
          top_k: 50,
          include_bos: false,
        });

        updatePromptAnalysis(prompt.id, response);
      }

      // Auto-generate response if there's only one prompt
      if (prompts.length === 1) {
        await generateResponse(prompts[0].text);
      }
    } catch (e) {
      setAnalysisError((e as Error).message);
    } finally {
      setAnalyzing(false);
    }
  }, [
    prompts,
    selectedLayers,
    setAnalyzing,
    setAnalysisError,
    setAnalysisProgress,
    updatePromptAnalysis,
    generateResponse,
    modelConfig,
    getSaePreset,
  ]);

  return (
    <main className="h-screen flex flex-col bg-zinc-950">
      {/* Header */}
      <header className="border-b border-zinc-800 shrink-0">
        <div className="px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold text-white">Gemma Feature Studio</h1>
            <p className="text-sm text-zinc-500">
              Multi-Layer SAE Analysis and Feature Steering
            </p>
          </div>
          <VRAMMonitor compact />
        </div>
        {/* Model Configuration */}
        <div className="px-6 pb-4 border-t border-zinc-800/50 pt-3">
          <ModelSelector
            disabled={isAnalyzing || isConfiguring}
            onConfigChange={handleConfigChange}
          />
        </div>
      </header>

      {/* Prompt Input Section */}
      <div className="border-b border-zinc-800 px-6 py-4 shrink-0">
        <PromptManager />
      </div>

      {/* Control Bar - Layers and Analyze */}
      <div className="border-b border-zinc-800 px-6 py-3 shrink-0">
        <div className="flex items-center justify-between gap-6">
          <LayerSelector disabled={isAnalyzing} />

          <div className="flex items-center gap-4">
            {/* Analyze Button */}
            <button
              onClick={handleAnalyze}
              disabled={prompts.length === 0 || selectedLayers.length === 0 || isAnalyzing}
              className={`
                px-6 py-2 rounded-lg font-medium transition-colors
                ${
                  isAnalyzing
                    ? "bg-blue-600/50 text-blue-200 cursor-wait"
                    : prompts.length > 0 && selectedLayers.length > 0
                    ? "bg-blue-600 text-white hover:bg-blue-500"
                    : "bg-zinc-700 text-zinc-400 cursor-not-allowed"
                }
              `}
            >
              {isAnalyzing ? (
                <span className="flex items-center gap-2">
                  <svg
                    className="w-4 h-4 animate-spin"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                    />
                  </svg>
                  {analysisProgress
                    ? `Analyzing ${analysisProgress.current}/${analysisProgress.total}...`
                    : "Analyzing..."}
                </span>
              ) : (
                "Analyze"
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Error Display */}
      {analysisError && (
        <div className="mx-6 mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg shrink-0">
          <div className="flex items-center justify-between">
            <span className="text-red-400 text-sm">{analysisError}</span>
            <button
              onClick={() => setAnalysisError(null)}
              className="text-red-400 hover:text-red-300"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>
        </div>
      )}

      {/* Global Steering Panel */}
      <GlobalSteeringPanel
        onGenerate={handleGenerateWithSteering}
        isGenerating={isGenerating}
        disabled={!activePrompt}
      />

      {/* Main Content Area - Visualization + Inspector Panel */}
      <div className={`flex-1 overflow-hidden min-h-0 flex ${panelConfig.position === "right" ? "flex-row" : "flex-col"}`}>
        {/* Flow Visualization */}
        <div className="flex-1 overflow-hidden min-h-0 min-w-0">
          <FlowVisualization
            generatedOutput={generatedOutput}
            baselineOutput={baselineOutput}
            isGenerating={isGenerating}
          />
        </div>

        {/* Feature Inspector Panel */}
        <ResizablePanel>
          <FeatureInspector
            generatedOutput={generatedOutput}
            baselineOutput={baselineOutput}
          />
        </ResizablePanel>
      </div>
    </main>
  );
}
