"use client";

import { useState, useCallback, useRef } from "react";
import type { SteeringFeature, GenerateRequest } from "@/types/analysis";
import { generateTextStream } from "@/lib/api";

interface StreamingState {
  output: string;
  isStreaming: boolean;
  error: string | null;
}

interface GenerationResult {
  baseline: StreamingState;
  steered: StreamingState;
}

interface UseStreamingGenerationReturn {
  result: GenerationResult;
  generate: (
    prompt: string,
    steering: SteeringFeature[],
    options?: { maxTokens?: number; temperature?: number; includeBaseline?: boolean }
  ) => Promise<void>;
  cancel: () => void;
  reset: () => void;
}

const initialStreamingState: StreamingState = {
  output: "",
  isStreaming: false,
  error: null,
};

export function useStreamingGeneration(): UseStreamingGenerationReturn {
  const [result, setResult] = useState<GenerationResult>({
    baseline: initialStreamingState,
    steered: initialStreamingState,
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  const updateState = useCallback(
    (
      type: "baseline" | "steered",
      updates: Partial<StreamingState>
    ) => {
      setResult((prev) => ({
        ...prev,
        [type]: { ...prev[type], ...updates },
      }));
    },
    []
  );

  const streamGeneration = useCallback(
    async (
      type: "baseline" | "steered",
      request: GenerateRequest,
      signal: AbortSignal
    ) => {
      updateState(type, { isStreaming: true, error: null, output: "" });

      try {
        for await (const token of generateTextStream(request, signal)) {
          if (signal.aborted) break;
          setResult((prev) => ({
            ...prev,
            [type]: {
              ...prev[type],
              output: prev[type].output + token,
            },
          }));
        }
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          updateState(type, {
            error: err instanceof Error ? err.message : "Generation failed",
          });
        }
      } finally {
        updateState(type, { isStreaming: false });
      }
    },
    [updateState]
  );

  const generate = useCallback(
    async (
      prompt: string,
      steering: SteeringFeature[],
      options?: {
        maxTokens?: number;
        temperature?: number;
        includeBaseline?: boolean;
      }
    ) => {
      // Cancel any existing generation
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      const abortController = new AbortController();
      abortControllerRef.current = abortController;

      const includeBaseline = options?.includeBaseline ?? true;
      const baseRequest: GenerateRequest = {
        prompt,
        steering: [],
        max_tokens: options?.maxTokens ?? 100,
        temperature: options?.temperature ?? 0.7,
      };

      const steeredRequest: GenerateRequest = {
        prompt,
        steering,
        max_tokens: options?.maxTokens ?? 100,
        temperature: options?.temperature ?? 0.7,
      };

      // Reset states
      setResult({
        baseline: { ...initialStreamingState, isStreaming: includeBaseline },
        steered: { ...initialStreamingState, isStreaming: true },
      });

      // Run streams in parallel
      const promises: Promise<void>[] = [
        streamGeneration("steered", steeredRequest, abortController.signal),
      ];

      if (includeBaseline) {
        promises.push(
          streamGeneration("baseline", baseRequest, abortController.signal)
        );
      }

      await Promise.all(promises);
    },
    [streamGeneration]
  );

  const cancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  }, []);

  const reset = useCallback(() => {
    cancel();
    setResult({
      baseline: initialStreamingState,
      steered: initialStreamingState,
    });
  }, [cancel]);

  return { result, generate, cancel, reset };
}
