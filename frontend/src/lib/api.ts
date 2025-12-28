import type {
  AnalyzeRequest,
  AnalyzeResponse,
  GenerateRequest,
  GenerateResponse,
  ModelInfo,
  HealthStatus,
  NeuronpediaFeatureRequest,
  NeuronpediaFeatureResponse,
} from "@/types/analysis";

import type {
  VRAMStatus,
  SystemStatus,
  MultiLayerAnalyzeRequest,
  MultiLayerAnalyzeResponse,
  BatchAnalyzeRequest,
  BatchAnalyzeResponse,
  PromptAnalysisResult,
  PreflightRequest,
  PreflightResponse,
  LoadSAERequest,
  LoadSAEResponse,
  UnloadSAERequest,
  UnloadSAEResponse,
} from "@/types/flow";

const API_BASE = "/api";

async function fetchJson<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    headers: {
      "Content-Type": "application/json",
    },
    ...options,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `API error: ${response.status}`);
  }

  return response.json();
}

export async function checkHealth(): Promise<HealthStatus> {
  return fetchJson<HealthStatus>("/health");
}

export async function loadModel(): Promise<{ status: string; message: string }> {
  return fetchJson("/load", { method: "POST" });
}

export async function unloadModel(): Promise<{ status: string; message: string }> {
  return fetchJson("/unload", { method: "POST" });
}

export async function getModelInfo(): Promise<ModelInfo> {
  return fetchJson<ModelInfo>("/model/info");
}

export async function analyzePrompt(
  request: AnalyzeRequest
): Promise<AnalyzeResponse> {
  return fetchJson<AnalyzeResponse>("/analyze", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export async function generateText(
  request: GenerateRequest
): Promise<GenerateResponse> {
  return fetchJson<GenerateResponse>("/generate", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export async function* generateTextStream(
  request: GenerateRequest,
  signal?: AbortSignal
): AsyncGenerator<string, void, unknown> {
  const response = await fetch(`${API_BASE}/generate/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("No response body");
  }

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const data = JSON.parse(line.slice(6));
          if (data.token) {
            yield data.token;
          }
        } catch {
          // Ignore parse errors
        }
      }
    }
  }
}

// =============================================================================
// VRAM and System Status
// =============================================================================

export async function getVRAMStatus(): Promise<VRAMStatus> {
  return fetchJson<VRAMStatus>("/vram/status");
}

export async function getSystemStatus(): Promise<SystemStatus> {
  return fetchJson<SystemStatus>("/system/status");
}

export async function preflightCheck(
  request: PreflightRequest
): Promise<PreflightResponse> {
  return fetchJson<PreflightResponse>("/vram/preflight", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

// =============================================================================
// SAE Management
// =============================================================================

export async function getLoadedSAEs(): Promise<{
  layers: number[];
  available_layers: number[];
  registry: Record<string, unknown>;
}> {
  return fetchJson("/sae/loaded");
}

export async function loadSAEs(request: LoadSAERequest): Promise<LoadSAEResponse> {
  return fetchJson<LoadSAEResponse>("/sae/load", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export async function unloadSAE(request: UnloadSAERequest): Promise<UnloadSAEResponse> {
  return fetchJson<UnloadSAEResponse>("/sae/unload", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

// =============================================================================
// Multi-Layer Analysis
// =============================================================================

export async function analyzeMultiLayer(
  request: MultiLayerAnalyzeRequest
): Promise<MultiLayerAnalyzeResponse> {
  return fetchJson<MultiLayerAnalyzeResponse>("/analyze/multi-layer", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

// =============================================================================
// Batch Analysis
// =============================================================================

export async function analyzeBatch(
  request: BatchAnalyzeRequest
): Promise<BatchAnalyzeResponse> {
  return fetchJson<BatchAnalyzeResponse>("/analyze/batch", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export interface BatchStreamEvent {
  type: "progress" | "result" | "complete" | "error";
  prompt_index?: number;
  prompt_id?: string;
  total: number;
  completed: number;
  result?: PromptAnalysisResult;
  error?: string;
}

export async function* analyzeBatchStream(
  request: BatchAnalyzeRequest,
  signal?: AbortSignal
): AsyncGenerator<BatchStreamEvent, void, unknown> {
  const response = await fetch(`${API_BASE}/analyze/batch/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("No response body");
  }

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const data = JSON.parse(line.slice(6)) as BatchStreamEvent;
          yield data;
        } catch {
          // Ignore parse errors
        }
      }
    }
  }
}

// =============================================================================
// Neuronpedia Feature Lookup
// =============================================================================

export async function getNeuronpediaFeature(
  request: NeuronpediaFeatureRequest
): Promise<NeuronpediaFeatureResponse> {
  return fetchJson<NeuronpediaFeatureResponse>("/neuronpedia/feature", {
    method: "POST",
    body: JSON.stringify(request),
  });
}
