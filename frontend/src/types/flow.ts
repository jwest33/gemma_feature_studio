/**
 * Types for the flow-based multi-layer visualization
 */

import type { LayerActivations, TokenActivations, FeatureActivation } from "./analysis";

// ============================================================================
// Available Layers Configuration
// ============================================================================

export const AVAILABLE_LAYERS = [9, 17, 22, 29] as const;
export type AvailableLayer = (typeof AVAILABLE_LAYERS)[number];

export const LAYER_COLORS: Record<AvailableLayer, string> = {
  9: "purple",
  17: "blue",
  22: "cyan",
  29: "green",
};

export const LAYER_BORDER_CLASSES: Record<AvailableLayer, string> = {
  9: "border-purple-500",
  17: "border-blue-500",
  22: "border-cyan-500",
  29: "border-green-500",
};

export const LAYER_BG_CLASSES: Record<AvailableLayer, string> = {
  9: "bg-purple-500/10",
  17: "bg-blue-500/10",
  22: "bg-cyan-500/10",
  29: "bg-green-500/10",
};

// Full background classes for selected states (Tailwind JIT requires static classes)
export const LAYER_BG_SELECTED_CLASSES: Record<AvailableLayer, string> = {
  9: "bg-purple-500/20",
  17: "bg-blue-500/20",
  22: "bg-cyan-500/20",
  29: "bg-green-500/20",
};

// Solid background classes for checkboxes
export const LAYER_CHECKBOX_CLASSES: Record<AvailableLayer, string> = {
  9: "bg-purple-500 border-purple-500",
  17: "bg-blue-500 border-blue-500",
  22: "bg-cyan-500 border-cyan-500",
  29: "bg-green-500 border-green-500",
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

export const SAE_PRESETS: SAEPreset[] = [
  { id: "gemmascope-2-res-16k", label: "GemmaScope 2 Res 16k", repo: "google/gemma-scope-2-4b-it", width: "16k", l0: "medium", type: "resid_post" },
  { id: "gemmascope-2-res-65k", label: "GemmaScope 2 Res 65k", repo: "google/gemma-scope-2-4b-it", width: "65k", l0: "medium", type: "resid_post" },
  { id: "gemmascope-2-res-262k", label: "GemmaScope 2 Res 262k", repo: "google/gemma-scope-2-4b-it", width: "262k", l0: "medium", type: "resid_post" },
  { id: "gemmascope-2-res-1m", label: "GemmaScope 2 Res 1M", repo: "google/gemma-scope-2-4b-it", width: "1m", l0: "medium", type: "resid_post" },
];

export interface ModelConfig {
  modelPath: string;
  saePresetId: string;
}

export interface ConfigureModelRequest {
  model_name: string;
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
  return LAYER_COLORS[layer as AvailableLayer] || "gray";
}

export function getLayerBorderClass(layer: number): string {
  return LAYER_BORDER_CLASSES[layer as AvailableLayer] || "border-gray-500";
}

export function getLayerBgClass(layer: number): string {
  return LAYER_BG_CLASSES[layer as AvailableLayer] || "bg-gray-500/10";
}

export function getLayerBgSelectedClass(layer: number): string {
  return LAYER_BG_SELECTED_CLASSES[layer as AvailableLayer] || "bg-gray-500/20";
}

export function getLayerCheckboxClass(layer: number): string {
  return LAYER_CHECKBOX_CLASSES[layer as AvailableLayer] || "bg-gray-500 border-gray-500";
}

export function isValidLayer(layer: number): layer is AvailableLayer {
  return AVAILABLE_LAYERS.includes(layer as AvailableLayer);
}
