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
}

export interface GenerateRequest {
  prompt: string;
  steering: SteeringFeature[];
  max_tokens?: number;
  temperature?: number;
  include_baseline?: boolean;
}

export interface GenerateResponse {
  prompt: string;
  baseline_output: string | null;
  steered_output: string;
  steering_config: SteeringFeature[];
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
