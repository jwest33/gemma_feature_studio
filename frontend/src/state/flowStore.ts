/**
 * Zustand store for the flow-based multi-layer visualization
 */

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import type {
  VRAMStatus,
  SystemStatus,
  PromptAnalysis,
  FlowSelection,
  ComparisonSelection,
  MultiLayerAnalyzeResponse,
  LoadSAEResponse,
  AVAILABLE_LAYERS,
  ModelConfig,
  SAEPreset,
} from "@/types/flow";
import { SAE_PRESETS } from "@/types/flow";
import type { LayerActivations, NormalizationMode } from "@/types/analysis";

// ============================================================================
// Global Steering Types
// ============================================================================

export interface GlobalSteeringFeature {
  id: number;
  layer: number;
  coefficient: number;
  enabled: boolean;
  name?: string; // From Neuronpedia description
}

// ============================================================================
// Filter Types
// ============================================================================

export interface FlowFilters {
  minActivation: number;  // Minimum activation strength (0-1 normalized)
  minTokenCount: number;  // Minimum number of tokens activating a feature
}

// ============================================================================
// Panel Types
// ============================================================================

export type PanelPosition = "bottom" | "right";

export interface PanelConfig {
  position: PanelPosition;
  size: number;  // Height when bottom, width when right (in pixels)
}

// ============================================================================
// Store State Interface
// ============================================================================

interface FlowState {
  // Model Configuration
  modelConfig: ModelConfig;

  // Prompts
  prompts: PromptAnalysis[];
  activePromptId: string | null;

  // Layer Selection
  selectedLayers: number[];

  // Filters
  filters: FlowFilters;

  // Inspector Panel
  panelConfig: PanelConfig;

  // Global Steering
  steeringFeatures: GlobalSteeringFeature[];
  steeringPanelExpanded: boolean;
  steeringNormalization: NormalizationMode;
  steeringClampFactor: number;
  steeringUnitNormalize: boolean;

  // System Status
  systemStatus: SystemStatus | null;
  isLoadingStatus: boolean;

  // Analysis State
  isAnalyzing: boolean;
  analysisProgress: { current: number; total: number } | null;
  analysisError: string | null;

  // Selection State
  selection: FlowSelection;
  comparison: ComparisonSelection | null;

  // Actions - Model Configuration
  setModelPath: (path: string) => void;
  setSaePreset: (presetId: string) => void;
  getSaePreset: () => SAEPreset | undefined;

  // Actions - Prompts
  addPrompt: (text: string) => string;
  addPrompts: (texts: string[]) => string[];
  removePrompt: (id: string) => void;
  clearPrompts: () => void;
  setActivePrompt: (id: string | null) => void;
  updatePromptAnalysis: (id: string, response: MultiLayerAnalyzeResponse) => void;

  // Actions - Layers
  toggleLayer: (layer: number) => void;
  setSelectedLayers: (layers: number[]) => void;
  selectAllLayers: () => void;
  clearLayerSelection: () => void;

  // Actions - System Status
  setSystemStatus: (status: SystemStatus | null) => void;
  setLoadingStatus: (loading: boolean) => void;

  // Actions - Analysis
  setAnalyzing: (analyzing: boolean) => void;
  setAnalysisProgress: (current: number, total: number) => void;
  clearAnalysisProgress: () => void;
  setAnalysisError: (error: string | null) => void;

  // Actions - Selection
  selectToken: (tokenIndex: number | null) => void;
  selectFeature: (featureId: number | null, layer: number | null) => void;
  selectOutput: () => void;
  clearSelection: () => void;

  // Actions - Comparison
  togglePromptComparison: (promptId: string) => void;
  setComparisonLayer: (layer: number) => void;
  clearComparison: () => void;

  // Actions - Filters
  setMinActivation: (value: number) => void;
  setMinTokenCount: (value: number) => void;
  resetFilters: () => void;

  // Actions - Global Steering
  addSteeringFeature: (id: number, layer: number, name?: string) => void;
  removeSteeringFeature: (id: number, layer: number) => void;
  updateSteeringFeature: (id: number, layer: number, updates: Partial<Omit<GlobalSteeringFeature, 'id' | 'layer'>>) => void;
  setSteeringFeatures: (features: GlobalSteeringFeature[]) => void;
  clearSteeringFeatures: () => void;
  toggleSteeringPanel: () => void;
  setSteeringPanelExpanded: (expanded: boolean) => void;
  setSteeringNormalization: (mode: NormalizationMode) => void;
  setSteeringClampFactor: (factor: number) => void;
  setSteeringUnitNormalize: (value: boolean) => void;
  hasSteeringFeature: (id: number, layer: number) => boolean;
  getEnabledSteeringFeatures: () => GlobalSteeringFeature[];

  // Actions - Panel
  setPanelPosition: (position: PanelPosition) => void;
  setPanelSize: (size: number) => void;
  togglePanelPosition: () => void;

  // Selectors
  getActivePrompt: () => PromptAnalysis | null;
  getPromptById: (id: string) => PromptAnalysis | undefined;
  getComparisonPrompts: () => PromptAnalysis[];
  getLoadedLayers: () => number[];
  canAnalyze: () => boolean;
}

// ============================================================================
// Available Layers
// ============================================================================

const ALL_AVAILABLE_LAYERS = [9, 17, 22, 29];

// ============================================================================
// Store Implementation
// ============================================================================

export const useFlowStore = create<FlowState>()(
  persist(
    (set, get) => ({
      // Initial State
      modelConfig: {
        modelPath: "google/gemma-3-4b-it",
        saePresetId: "gemmascope-2-res-65k",
      },
      prompts: [],
      activePromptId: null,
      selectedLayers: [17], // Default to layer 17
      filters: {
        minActivation: 0,
        minTokenCount: 1,
      },
      panelConfig: {
        position: "bottom",
        size: 288, // 72 * 4 = h-72 equivalent
      },
      // Global Steering
      steeringFeatures: [],
      steeringPanelExpanded: false,
      steeringNormalization: "preserve_norm" as NormalizationMode,
      steeringClampFactor: 1.5,
      steeringUnitNormalize: false,
      // System Status
      systemStatus: null,
      isLoadingStatus: false,
      isAnalyzing: false,
      analysisProgress: null,
      analysisError: null,
      selection: {
        tokenIndex: null,
        featureId: null,
        layer: null,
        showOutput: false,
      },
      comparison: null,

      // ======================================================================
      // Model Configuration Actions
      // ======================================================================

      setModelPath: (path: string) => {
        set((state) => ({
          modelConfig: { ...state.modelConfig, modelPath: path },
        }));
      },

      setSaePreset: (presetId: string) => {
        set((state) => ({
          modelConfig: { ...state.modelConfig, saePresetId: presetId },
        }));
      },

      getSaePreset: () => {
        const { modelConfig } = get();
        return SAE_PRESETS.find((p) => p.id === modelConfig.saePresetId);
      },

      // ======================================================================
      // Prompt Actions
      // ======================================================================

      addPrompt: (text: string) => {
        const id = crypto.randomUUID();
        const prompt: PromptAnalysis = {
          id,
          text,
          analyzedAt: null,
          tokens: [],
          layerData: new Map(),
        };
        set((state) => ({
          prompts: [...state.prompts, prompt],
          activePromptId: state.activePromptId ?? id,
        }));
        return id;
      },

      addPrompts: (texts: string[]) => {
        const newPrompts = texts.map((text) => ({
          id: crypto.randomUUID(),
          text,
          analyzedAt: null,
          tokens: [] as string[],
          layerData: new Map<number, LayerActivations>(),
        }));
        set((state) => ({
          prompts: [...state.prompts, ...newPrompts],
          activePromptId: state.activePromptId ?? newPrompts[0]?.id ?? null,
        }));
        return newPrompts.map((p) => p.id);
      },

      removePrompt: (id: string) => {
        set((state) => {
          const newPrompts = state.prompts.filter((p) => p.id !== id);
          const newActiveId =
            state.activePromptId === id
              ? newPrompts[0]?.id ?? null
              : state.activePromptId;
          return {
            prompts: newPrompts,
            activePromptId: newActiveId,
            comparison:
              state.comparison?.promptIds.includes(id)
                ? {
                    ...state.comparison,
                    promptIds: state.comparison.promptIds.filter((pid) => pid !== id),
                  }
                : state.comparison,
          };
        });
      },

      clearPrompts: () => {
        set({
          prompts: [],
          activePromptId: null,
          comparison: null,
          selection: { tokenIndex: null, featureId: null, layer: null, showOutput: false },
        });
      },

      setActivePrompt: (id: string | null) => {
        set({ activePromptId: id });
      },

      updatePromptAnalysis: (id: string, response: MultiLayerAnalyzeResponse) => {
        set((state) => {
          const layerData = new Map<number, LayerActivations>();
          for (const layer of response.layers) {
            layerData.set(layer.layer, layer);
          }

          return {
            prompts: state.prompts.map((p) =>
              p.id === id
                ? {
                    ...p,
                    tokens: response.tokens,
                    layerData,
                    analyzedAt: new Date().toISOString(),
                  }
                : p
            ),
          };
        });
      },

      // ======================================================================
      // Layer Actions
      // ======================================================================

      toggleLayer: (layer: number) => {
        set((state) => {
          const isSelected = state.selectedLayers.includes(layer);
          if (isSelected) {
            // Don't allow deselecting all layers
            if (state.selectedLayers.length === 1) return state;
            return {
              selectedLayers: state.selectedLayers.filter((l) => l !== layer),
            };
          } else {
            return {
              selectedLayers: [...state.selectedLayers, layer].sort((a, b) => a - b),
            };
          }
        });
      },

      setSelectedLayers: (layers: number[]) => {
        set({ selectedLayers: layers.sort((a, b) => a - b) });
      },

      selectAllLayers: () => {
        set({ selectedLayers: [...ALL_AVAILABLE_LAYERS] });
      },

      clearLayerSelection: () => {
        set({ selectedLayers: [17] }); // Reset to default
      },

      // ======================================================================
      // System Status Actions
      // ======================================================================

      setSystemStatus: (status: SystemStatus | null) => {
        set({ systemStatus: status });
      },

      setLoadingStatus: (loading: boolean) => {
        set({ isLoadingStatus: loading });
      },

      // ======================================================================
      // Analysis Actions
      // ======================================================================

      setAnalyzing: (analyzing: boolean) => {
        set({ isAnalyzing: analyzing });
        if (!analyzing) {
          set({ analysisProgress: null });
        }
      },

      setAnalysisProgress: (current: number, total: number) => {
        set({ analysisProgress: { current, total } });
      },

      clearAnalysisProgress: () => {
        set({ analysisProgress: null });
      },

      setAnalysisError: (error: string | null) => {
        set({ analysisError: error });
      },

      // ======================================================================
      // Selection Actions
      // ======================================================================

      selectToken: (tokenIndex: number | null) => {
        set({
          selection: {
            tokenIndex,
            featureId: null,
            layer: null,
            showOutput: false,
          },
        });
      },

      selectFeature: (featureId: number | null, layer: number | null) => {
        set({
          selection: {
            tokenIndex: null,
            featureId,
            layer,
            showOutput: false,
          },
        });
      },

      selectOutput: () => {
        set({
          selection: {
            tokenIndex: null,
            featureId: null,
            layer: null,
            showOutput: true,
          },
        });
      },

      clearSelection: () => {
        set({
          selection: { tokenIndex: null, featureId: null, layer: null, showOutput: false },
        });
      },

      // ======================================================================
      // Comparison Actions
      // ======================================================================

      togglePromptComparison: (promptId: string) => {
        set((state) => {
          const current = state.comparison?.promptIds || [];
          const isSelected = current.includes(promptId);

          if (isSelected) {
            const newIds = current.filter((pid) => pid !== promptId);
            return {
              comparison:
                newIds.length >= 2
                  ? { ...state.comparison!, promptIds: newIds }
                  : null,
            };
          } else {
            return {
              comparison: {
                promptIds: [...current, promptId],
                layer: state.comparison?.layer ?? state.selectedLayers[0] ?? 17,
              },
            };
          }
        });
      },

      setComparisonLayer: (layer: number) => {
        set((state) => ({
          comparison: state.comparison
            ? { ...state.comparison, layer }
            : null,
        }));
      },

      clearComparison: () => {
        set({ comparison: null });
      },

      // ======================================================================
      // Filter Actions
      // ======================================================================

      setMinActivation: (value: number) => {
        set((state) => ({
          filters: { ...state.filters, minActivation: value },
        }));
      },

      setMinTokenCount: (value: number) => {
        set((state) => ({
          filters: { ...state.filters, minTokenCount: Math.max(1, Math.round(value)) },
        }));
      },

      resetFilters: () => {
        set({
          filters: { minActivation: 0, minTokenCount: 1 },
        });
      },

      // ======================================================================
      // Global Steering Actions
      // ======================================================================

      addSteeringFeature: (id: number, layer: number, name?: string) => {
        set((state) => {
          // Check if feature already exists
          const exists = state.steeringFeatures.some(
            (f) => f.id === id && f.layer === layer
          );
          if (exists) return state;

          const newFeature: GlobalSteeringFeature = {
            id,
            layer,
            coefficient: 0,
            enabled: true,
            name,
          };

          return {
            steeringFeatures: [...state.steeringFeatures, newFeature],
            steeringPanelExpanded: true, // Auto-expand when adding
          };
        });
      },

      removeSteeringFeature: (id: number, layer: number) => {
        set((state) => ({
          steeringFeatures: state.steeringFeatures.filter(
            (f) => !(f.id === id && f.layer === layer)
          ),
        }));
      },

      updateSteeringFeature: (id: number, layer: number, updates: Partial<Omit<GlobalSteeringFeature, 'id' | 'layer'>>) => {
        set((state) => ({
          steeringFeatures: state.steeringFeatures.map((f) =>
            f.id === id && f.layer === layer ? { ...f, ...updates } : f
          ),
        }));
      },

      setSteeringFeatures: (features: GlobalSteeringFeature[]) => {
        set({ steeringFeatures: features });
      },

      clearSteeringFeatures: () => {
        set({ steeringFeatures: [] });
      },

      toggleSteeringPanel: () => {
        set((state) => ({
          steeringPanelExpanded: !state.steeringPanelExpanded,
        }));
      },

      setSteeringPanelExpanded: (expanded: boolean) => {
        set({ steeringPanelExpanded: expanded });
      },

      setSteeringNormalization: (mode: NormalizationMode) => {
        set({ steeringNormalization: mode });
      },

      setSteeringClampFactor: (factor: number) => {
        set({ steeringClampFactor: Math.max(1.0, Math.min(3.0, factor)) });
      },

      setSteeringUnitNormalize: (value: boolean) => {
        set({ steeringUnitNormalize: value });
      },

      hasSteeringFeature: (id: number, layer: number) => {
        return get().steeringFeatures.some((f) => f.id === id && f.layer === layer);
      },

      getEnabledSteeringFeatures: () => {
        return get().steeringFeatures.filter((f) => f.enabled);
      },

      // ======================================================================
      // Panel Actions
      // ======================================================================

      setPanelPosition: (position: PanelPosition) => {
        set((state) => ({
          panelConfig: {
            ...state.panelConfig,
            position,
            // Reset size to default for new position
            size: position === "bottom" ? 288 : 400,
          },
        }));
      },

      setPanelSize: (size: number) => {
        const position = get().panelConfig.position;
        const maxSize = position === "bottom" ? 600 : 1200;
        set((state) => ({
          panelConfig: {
            ...state.panelConfig,
            size: Math.max(200, Math.min(size, maxSize)),
          },
        }));
      },

      togglePanelPosition: () => {
        const current = get().panelConfig.position;
        const newPosition = current === "bottom" ? "right" : "bottom";
        set({
          panelConfig: {
            position: newPosition,
            size: newPosition === "bottom" ? 288 : 400,
          },
        });
      },

      // ======================================================================
      // Selectors
      // ======================================================================

      getActivePrompt: () => {
        const { prompts, activePromptId } = get();
        if (!activePromptId) return null;
        return prompts.find((p) => p.id === activePromptId) ?? null;
      },

      getPromptById: (id: string) => {
        return get().prompts.find((p) => p.id === id);
      },

      getComparisonPrompts: () => {
        const { prompts, comparison } = get();
        if (!comparison) return [];
        return comparison.promptIds
          .map((id) => prompts.find((p) => p.id === id))
          .filter((p): p is PromptAnalysis => p !== undefined);
      },

      getLoadedLayers: () => {
        const { systemStatus } = get();
        return systemStatus?.saes?.loaded_layers ?? [];
      },

      canAnalyze: () => {
        const { prompts, selectedLayers, isAnalyzing } = get();
        return prompts.length > 0 && selectedLayers.length > 0 && !isAnalyzing;
      },
    }),
    {
      name: "lm-feature-studio-flow",
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        // Only persist these fields
        modelConfig: state.modelConfig,
        selectedLayers: state.selectedLayers,
        filters: state.filters,
        panelConfig: state.panelConfig,
        // Persist steering settings (but not features - they're transient)
        steeringNormalization: state.steeringNormalization,
        steeringClampFactor: state.steeringClampFactor,
        steeringUnitNormalize: state.steeringUnitNormalize,
        // Don't persist prompts, analysis results, or steering features (too large/transient)
      }),
    }
  )
);

// ============================================================================
// Selector Hooks (for convenience)
// ============================================================================

export const useActivePrompt = () => useFlowStore((state) => state.getActivePrompt());
export const useSelectedLayers = () => useFlowStore((state) => state.selectedLayers);
export const useModelConfig = () => useFlowStore((state) => state.modelConfig);
export const useSystemStatus = () => useFlowStore((state) => state.systemStatus);
export const useIsAnalyzing = () => useFlowStore((state) => state.isAnalyzing);
export const useAnalysisProgress = () => useFlowStore((state) => state.analysisProgress);
export const useSelection = () => useFlowStore((state) => state.selection);
export const useFilters = () => useFlowStore((state) => state.filters);
export const usePanelConfig = () => useFlowStore((state) => state.panelConfig);

// Steering selectors
export const useSteeringFeatures = () => useFlowStore((state) => state.steeringFeatures);
export const useSteeringPanelExpanded = () => useFlowStore((state) => state.steeringPanelExpanded);
export const useSteeringNormalization = () => useFlowStore((state) => state.steeringNormalization);
export const useSteeringClampFactor = () => useFlowStore((state) => state.steeringClampFactor);
export const useSteeringUnitNormalize = () => useFlowStore((state) => state.steeringUnitNormalize);
