/**
 * Types for the flow-based multi-layer visualization
 */

import type { LayerActivations, TokenActivations, FeatureActivation } from "./analysis";

// ============================================================================
// Model Size Configuration
// ============================================================================

export type ModelSize = "270m" | "1b" | "4b" | "12b" | "27b";

export interface ModelSizeConfig {
  size: ModelSize;
  layers: number[];
  saeRepo: string;
  modelName: string;
  displayName: string;
}

// Maps model sizes to their available SAE layers and configurations
export const MODEL_SIZE_CONFIGS: Record<ModelSize, ModelSizeConfig> = {
  "270m": {
    size: "270m",
    layers: [5, 9, 12, 15],
    saeRepo: "google/gemma-scope-2-270m-it",
    modelName: "google/gemma-3-270m-it",
    displayName: "Gemma 3 270M",
  },
  "1b": {
    size: "1b",
    layers: [7, 13, 17, 22],
    saeRepo: "google/gemma-scope-2-1b-it",
    modelName: "google/gemma-3-1b-it",
    displayName: "Gemma 3 1B",
  },
  "4b": {
    size: "4b",
    layers: [9, 17, 22, 29],
    saeRepo: "google/gemma-scope-2-4b-it",
    modelName: "google/gemma-3-4b-it",
    displayName: "Gemma 3 4B",
  },
  "12b": {
    size: "12b",
    layers: [12, 24, 31, 41],
    saeRepo: "google/gemma-scope-2-12b-it",
    modelName: "google/gemma-3-12b-it",
    displayName: "Gemma 3 12B",
  },
  "27b": {
    size: "27b",
    layers: [16, 31, 40, 53],
    saeRepo: "google/gemma-scope-2-27b-it",
    modelName: "google/gemma-3-27b-it",
    displayName: "Gemma 3 27B",
  },
};

// Default layers for backwards compatibility (4B model)
export const DEFAULT_AVAILABLE_LAYERS = [9, 17, 22, 29] as const;
export type AvailableLayer = number;

// ============================================================================
// Layer Color Configuration
// ============================================================================

// Color palette for layers - uses position-based coloring
// First layer = purple, second = blue, third = cyan, fourth = green
const LAYER_COLOR_PALETTE = ["purple", "blue", "cyan", "green"] as const;
const LAYER_HEX_PALETTE = ["#a855f7", "#3b82f6", "#06b6d4", "#22c55e"] as const;

/**
 * Get color for a layer based on its position in the available layers array
 */
export function getLayerColorByPosition(layer: number, availableLayers: number[]): string {
  const index = availableLayers.indexOf(layer);
  if (index === -1) return "gray";
  return LAYER_COLOR_PALETTE[index % LAYER_COLOR_PALETTE.length];
}

/**
 * Get hex color for a layer based on its position
 */
export function getLayerHexByPosition(layer: number, availableLayers: number[]): string {
  const index = availableLayers.indexOf(layer);
  if (index === -1) return "#888888";
  return LAYER_HEX_PALETTE[index % LAYER_HEX_PALETTE.length];
}

// Legacy static mappings for backwards compatibility (4B model layers)
export const LAYER_COLORS: Record<number, string> = {
  // 270M layers
  5: "purple", 9: "blue", 12: "cyan", 15: "green",
  // 1B layers
  7: "purple", 13: "blue", 17: "cyan", 22: "green",
  // 4B layers (also includes 9, 17, 22 from above)
  29: "green",
  // 12B layers
  24: "blue", 31: "cyan", 41: "green",
  // 27B layers
  16: "purple", 40: "cyan", 53: "green",
};

export const LAYER_BORDER_CLASSES: Record<number, string> = {
  5: "border-purple-500", 7: "border-purple-500", 9: "border-purple-500", 12: "border-purple-500", 16: "border-purple-500",
  13: "border-blue-500", 17: "border-blue-500", 24: "border-blue-500", 31: "border-blue-500",
  22: "border-cyan-500", 40: "border-cyan-500",
  15: "border-green-500", 29: "border-green-500", 41: "border-green-500", 53: "border-green-500",
};

export const LAYER_BG_CLASSES: Record<number, string> = {
  5: "bg-purple-500/10", 7: "bg-purple-500/10", 9: "bg-purple-500/10", 12: "bg-purple-500/10", 16: "bg-purple-500/10",
  13: "bg-blue-500/10", 17: "bg-blue-500/10", 24: "bg-blue-500/10", 31: "bg-blue-500/10",
  22: "bg-cyan-500/10", 40: "bg-cyan-500/10",
  15: "bg-green-500/10", 29: "bg-green-500/10", 41: "bg-green-500/10", 53: "bg-green-500/10",
};

// Full background classes for selected states (Tailwind JIT requires static classes)
export const LAYER_BG_SELECTED_CLASSES: Record<number, string> = {
  5: "bg-purple-500/20", 7: "bg-purple-500/20", 9: "bg-purple-500/20", 12: "bg-purple-500/20", 16: "bg-purple-500/20",
  13: "bg-blue-500/20", 17: "bg-blue-500/20", 24: "bg-blue-500/20", 31: "bg-blue-500/20",
  22: "bg-cyan-500/20", 40: "bg-cyan-500/20",
  15: "bg-green-500/20", 29: "bg-green-500/20", 41: "bg-green-500/20", 53: "bg-green-500/20",
};

// Solid background classes for checkboxes
export const LAYER_CHECKBOX_CLASSES: Record<number, string> = {
  5: "bg-purple-500 border-purple-500", 7: "bg-purple-500 border-purple-500", 9: "bg-purple-500 border-purple-500", 12: "bg-purple-500 border-purple-500", 16: "bg-purple-500 border-purple-500",
  13: "bg-blue-500 border-blue-500", 17: "bg-blue-500 border-blue-500", 24: "bg-blue-500 border-blue-500", 31: "bg-blue-500 border-blue-500",
  22: "bg-cyan-500 border-cyan-500", 40: "bg-cyan-500 border-cyan-500",
  15: "bg-green-500 border-green-500", 29: "bg-green-500 border-green-500", 41: "bg-green-500 border-green-500", 53: "bg-green-500 border-green-500",
};

// ============================================================================
// VRAM Status Types
// ============================================================================

export interface VRAMStatus {
  total_gb: number;
  allocated_gb: number;
  reserved_gb: number;
  free_gb: number;
  pressure: "low" | "moderate" | "high" | "critical";
}

export interface SAEStatus {
  layer: number;
  width: string;
  l0: string;
  sae_type: string;
  sae_id: string;
  size_mb: number;
  loaded_at: number;
  last_used: number;
}

export interface SAERegistryStatus {
  loaded_count: number;
  loaded_layers: number[];
  total_size_mb: number;
  total_size_gb: number;
  max_budget_gb: number;
  entries: SAEStatus[];
}

export interface SystemStatus {
  model_loaded: boolean;
  model_name: string | null;
  model_size: ModelSize;
  vram: VRAMStatus;
  saes: SAERegistryStatus;
  available_layers: number[];
}

// ============================================================================
// Multi-Layer Analysis Types
// ============================================================================

export interface MultiLayerAnalyzeRequest {
  prompt: string;
  layers: number[];
  top_k?: number;
  include_bos?: boolean;
}

export interface MultiLayerAnalyzeResponse {
  prompt: string;
  tokens: string[];
  layers: LayerActivations[];
  model_name: string;
  sae_width: string;
  analyzed_layers: number[];
}

// ============================================================================
// Batch Analysis Types
// ============================================================================

export interface BatchAnalyzeRequest {
  prompts: string[];
  layers: number[];
  top_k?: number;
  include_bos?: boolean;
}

export interface PromptAnalysisResult {
  id: string;
  prompt: string;
  tokens: string[];
  layers: LayerActivations[];
  status: "success" | "error";
  error?: string | null;
}

export interface BatchAnalyzeResponse {
  results: PromptAnalysisResult[];
  model_name: string;
  sae_width: string;
  analyzed_layers: number[];
  total_prompts: number;
  successful: number;
  failed: number;
  processing_time_ms: number;
}

// ============================================================================
// SAE Loading Types
// ============================================================================

export interface PreflightRequest {
  layers: number[];
  width?: string;
}

export interface PreflightResponse {
  can_load: boolean;
  layers_to_load: number[];
  already_loaded: number[];
  bytes_needed: number;
  bytes_available: number;
  recommendation: string | null;
}

export interface LoadSAERequest {
  layers: number[];
  width?: string;
}

export interface LoadSAEResponse {
  loaded: number[];
  already_loaded: number[];
  failed: Array<{ layer: number; error: string }>;
  vram_status: VRAMStatus;
}

export interface UnloadSAERequest {
  layer: number;
  width?: string;
}

export interface UnloadSAEResponse {
  success: boolean;
  layer: number;
  vram_status: VRAMStatus;
}

// ============================================================================
// SAE Cache Status Types
// ============================================================================

export interface SAELayerCacheStatus {
  layer: number;
  cached: boolean;
  filename: string;
}

export interface SAECacheStatusResponse {
  layers: SAELayerCacheStatus[];
  all_cached: boolean;
  uncached_count: number;
  sae_repo: string;
}

export interface SAEDownloadResponse {
  downloaded: number[];
  already_cached: number[];
  failed: Array<{ layer: number; error: string }>;
  sae_repo: string;
}

// ============================================================================
// Flow Store Types
// ============================================================================

export interface PromptAnalysis {
  id: string;
  text: string;
  analyzedAt: string | null;
  tokens: string[];
  layerData: Map<number, LayerActivations>;
}

export interface FlowSelection {
  tokenIndex: number | null;
  featureId: number | null;
  layer: number | null;
  showOutput: boolean;
}

export interface ComparisonSelection {
  promptIds: string[];
  layer: number;
}

// ============================================================================
// Model Configuration Types
// ============================================================================

export interface SAEPreset {
  id: string;
  label: string;
  repo: string;
  width: string;
  l0: string;
  type: string;
}

// SAE presets for 4B model (default) - other model sizes auto-switch repos
export const SAE_PRESETS: SAEPreset[] = [
  { id: "gemmascope-2-res-16k", label: "GemmaScope 2 Res 16k", repo: "google/gemma-scope-2-4b-it", width: "16k", l0: "medium", type: "resid_post" },
  { id: "gemmascope-2-res-65k", label: "GemmaScope 2 Res 65k", repo: "google/gemma-scope-2-4b-it", width: "65k", l0: "medium", type: "resid_post" },
  { id: "gemmascope-2-res-262k", label: "GemmaScope 2 Res 262k", repo: "google/gemma-scope-2-4b-it", width: "262k", l0: "medium", type: "resid_post" },
  { id: "gemmascope-2-res-1m", label: "GemmaScope 2 Res 1M", repo: "google/gemma-scope-2-4b-it", width: "1m", l0: "medium", type: "resid_post" },
];

/**
 * Get SAE presets for a specific model size
 * The SAE repo is automatically updated based on model size
 */
export function getSaePresetsForModelSize(size: ModelSize): SAEPreset[] {
  const config = MODEL_SIZE_CONFIGS[size];
  if (!config) return SAE_PRESETS;

  return SAE_PRESETS.map(preset => ({
    ...preset,
    repo: config.saeRepo,
  }));
}

export interface ModelConfig {
  modelPath: string;
  modelSize: ModelSize;
  saePresetId: string;
  saeL0: "small" | "medium" | "big";
}

// L0 options for SAE sparsity (smaller = sparser = fewer features activate)
export const SAE_L0_OPTIONS = [
  { id: "small", label: "Small", description: "Sparse - fewer features, more interpretable" },
  { id: "medium", label: "Medium", description: "Balanced sparsity" },
  { id: "big", label: "Big", description: "Dense - more features, better reconstruction" },
] as const;

export type SAEL0 = typeof SAE_L0_OPTIONS[number]["id"];

export interface ConfigureModelRequest {
  model_name: string;
  model_size: ModelSize;
  sae_repo: string;
  sae_width: string;
  sae_l0: string;
  sae_type: string;
}

export interface ConfigureModelResponse {
  status: string;
  message: string;
  config: {
    model_name: string;
    sae_repo: string;
    sae_width: string;
    sae_l0: string;
    sae_type: string;
  };
}

// ============================================================================
// Helper Functions
// ============================================================================

export function getLayerColor(layer: number): string {
  return LAYER_COLORS[layer] || "gray";
}

export function getLayerBorderClass(layer: number): string {
  return LAYER_BORDER_CLASSES[layer] || "border-gray-500";
}

export function getLayerBgClass(layer: number): string {
  return LAYER_BG_CLASSES[layer] || "bg-gray-500/10";
}

export function getLayerBgSelectedClass(layer: number): string {
  return LAYER_BG_SELECTED_CLASSES[layer] || "bg-gray-500/20";
}

export function getLayerCheckboxClass(layer: number): string {
  return LAYER_CHECKBOX_CLASSES[layer] || "bg-gray-500 border-gray-500";
}

export function isValidLayer(layer: number, availableLayers: number[]): boolean {
  return availableLayers.includes(layer);
}

/**
 * Detect model size from model path/name
 */
export function detectModelSize(modelPath: string): ModelSize {
  const lowerPath = modelPath.toLowerCase();

  if (lowerPath.includes("270m")) return "270m";
  if (lowerPath.includes("27b")) return "27b";  // Check before 2b
  if (lowerPath.includes("12b")) return "12b";
  if (lowerPath.includes("4b")) return "4b";
  if (lowerPath.includes("1b")) return "1b";

  return "4b";  // Default
}

/**
 * Get available layers for a model size
 */
export function getLayersForModelSize(size: ModelSize): number[] {
  return MODEL_SIZE_CONFIGS[size]?.layers || DEFAULT_AVAILABLE_LAYERS;
}

/**
 * Get SAE repo for a model size
 */
export function getSaeRepoForModelSize(size: ModelSize, isPT: boolean = false): string {
  const config = MODEL_SIZE_CONFIGS[size];
  if (!config) return "google/gemma-scope-2-4b-it";
  // Note: The backend handles IT vs PT repo selection
  return config.saeRepo;
}
