"use client";

import { useState, useCallback } from "react";
import { analyzePrompt } from "@/lib/api";
import type { AnalyzeResponse, AnalyzeRequest } from "@/types/analysis";

interface UseAnalysisResult {
  data: AnalyzeResponse | null;
  isLoading: boolean;
  error: string | null;
  analyze: (request: AnalyzeRequest) => Promise<void>;
  reset: () => void;
}

export function useAnalysis(): UseAnalysisResult {
  const [data, setData] = useState<AnalyzeResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const analyze = useCallback(async (request: AnalyzeRequest) => {
    setIsLoading(true);
    setError(null);

    try {
      const result = await analyzePrompt(request);
      setData(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed");
      setData(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setIsLoading(false);
  }, []);

  return { data, isLoading, error, analyze, reset };
}
