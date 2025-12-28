export interface FeatureActivation {
  id: number;
  activation: number;
}

export interface TokenActivations {
  token: string;
  position: number;
  top_features: FeatureActivation[];
}

export interface LayerActivations {
  layer: number;
  hook_point: string;
  token_activations: TokenActivations[];
}

export interface AnalyzeResponse {
  prompt: string;
  tokens: string[];
  layers: LayerActivations[];
  model_name: string;
  sae_id: string;
}

export interface AnalyzeRequest {
  prompt: string;
  top_k?: number;
  include_bos?: boolean;
}

export interface SteeringFeature {
  feature_id: number;
  coefficient: number;
  layer?: number | null;
}

export type NormalizationMode = "none" | "preserve_norm" | "clamp";

export interface GenerateRequest {
  prompt: string;
  steering: SteeringFeature[];
  max_tokens?: number;
  temperature?: number;
  include_baseline?: boolean;
  normalization?: NormalizationMode;
  norm_clamp_factor?: number;
  unit_normalize?: boolean;
}

export interface GenerateResponse {
  prompt: string;
  baseline_output: string | null;
  steered_output: string;
  steering_config: SteeringFeature[];
  normalization?: NormalizationMode;
  unit_normalize?: boolean;
}

export interface ModelInfo {
  loaded: boolean;
  sae_width?: number;
  hook_point?: string;
  message?: string;
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
}

// Neuronpedia Types
export interface NeuronpediaActivation {
  tokens: string[];
  values: number[];
  maxValue: number;
  maxTokenIndex: number;
}

export interface NeuronpediaExplanation {
  description: string;
  explanationType: string;
  typeName?: string | null;
  explanationModelName?: string | null;
  score?: number | null;
}

export interface NeuronpediaFeatureRequest {
  feature_id: number;
  layer: number;
}

export interface NeuronpediaFeatureResponse {
  modelId: string;
  layer: string;
  index: number;
  description: string | null;
  explanations: NeuronpediaExplanation[];
  activations: NeuronpediaActivation[];
  neuronpedia_url: string;
  hasData: boolean;
}
