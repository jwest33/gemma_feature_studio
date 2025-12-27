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

export function isValidLayer(layer: number): layer is AvailableLayer {
  return AVAILABLE_LAYERS.includes(layer as AvailableLayer);
}
