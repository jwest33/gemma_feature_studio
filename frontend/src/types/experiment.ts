export interface SteeringFeatureState {
  id: number;
  name?: string;
  coefficient: number;
  enabled: boolean;
}

export interface ExperimentRun {
  id: string;
  timestamp: string;
  prompt: string;
  baselineOutput: string;
  steeredOutput: string;
  temperature: number;
  maxTokens: number;
}

export interface SteeringConfig {
  features: SteeringFeatureState[];
  targetLayer?: number;
}

export interface ModelConfig {
  name: string;
  saeRelease: string;
  saeId: string;
}

export interface SteeringExperiment {
  id: string;
  name: string;
  description?: string;
  createdAt: string;
  updatedAt: string;
  model: ModelConfig;
  steering: SteeringConfig;
  runs: ExperimentRun[];
}

export interface ExperimentSummary {
  id: string;
  name: string;
  description?: string;
  createdAt: string;
  updatedAt: string;
  featureCount: number;
  runCount: number;
}

export function createExperiment(
  name: string,
  model?: Partial<ModelConfig>
): SteeringExperiment {
  const now = new Date().toISOString();
  return {
    id: crypto.randomUUID(),
    name,
    createdAt: now,
    updatedAt: now,
    model: {
      name: model?.name ?? "gemma-2-2b",
      saeRelease: model?.saeRelease ?? "gemma-scope-2b-pt-res",
      saeId: model?.saeId ?? "layer_20/width_16k/average_l0_71",
    },
    steering: {
      features: [],
    },
    runs: [],
  };
}

export function createExperimentRun(
  prompt: string,
  baselineOutput: string,
  steeredOutput: string,
  temperature: number = 0.7,
  maxTokens: number = 100
): ExperimentRun {
  return {
    id: crypto.randomUUID(),
    timestamp: new Date().toISOString(),
    prompt,
    baselineOutput,
    steeredOutput,
    temperature,
    maxTokens,
  };
}

export function experimentToSummary(exp: SteeringExperiment): ExperimentSummary {
  return {
    id: exp.id,
    name: exp.name,
    description: exp.description,
    createdAt: exp.createdAt,
    updatedAt: exp.updatedAt,
    featureCount: exp.steering.features.length,
    runCount: exp.runs.length,
  };
}
