"use client";

import { useState, useEffect } from "react";
import { checkHealth, loadModel, unloadModel, getModelInfo } from "@/lib/api";
import type { ModelInfo } from "@/types/analysis";

export function ModelStatus() {
  const [status, setStatus] = useState<"loading" | "loaded" | "unloaded" | "error">("loading");
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = async () => {
    try {
      const health = await checkHealth();
      if (health.model_loaded) {
        setStatus("loaded");
        const info = await getModelInfo();
        setModelInfo(info);
      } else {
        setStatus("unloaded");
        setModelInfo(null);
      }
      setError(null);
    } catch (e) {
      setStatus("error");
      setError("Cannot connect to backend");
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleLoad = async () => {
    setIsLoading(true);
    setError(null);
    try {
      await loadModel();
      await fetchStatus();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load model");
    } finally {
      setIsLoading(false);
    }
  };

  const handleUnload = async () => {
    setIsLoading(true);
    try {
      await unloadModel();
      await fetchStatus();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to unload model");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div
            className={`w-3 h-3 rounded-full ${
              status === "loaded"
                ? "bg-green-500"
                : status === "unloaded"
                  ? "bg-yellow-500"
                  : status === "error"
                    ? "bg-red-500"
                    : "bg-zinc-500 animate-pulse"
            }`}
          />
          <div>
            <span className="text-sm font-medium text-white">
              {status === "loaded"
                ? "Model Ready"
                : status === "unloaded"
                  ? "Model Not Loaded"
                  : status === "error"
                    ? "Connection Error"
                    : "Checking..."}
            </span>
            {modelInfo?.loaded && (
              <p className="text-xs text-zinc-500">
                SAE width: {modelInfo.sae_width?.toLocaleString()} |{" "}
                {modelInfo.hook_point}
              </p>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {status === "unloaded" && (
            <button
              onClick={handleLoad}
              disabled={isLoading}
              className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 disabled:bg-zinc-700 text-white text-sm rounded transition-colors"
            >
              {isLoading ? "Loading..." : "Load Model"}
            </button>
          )}
          {status === "loaded" && (
            <button
              onClick={handleUnload}
              disabled={isLoading}
              className="px-3 py-1.5 bg-zinc-700 hover:bg-zinc-600 disabled:bg-zinc-800 text-white text-sm rounded transition-colors"
            >
              {isLoading ? "Unloading..." : "Unload"}
            </button>
          )}
          {status === "error" && (
            <button
              onClick={fetchStatus}
              className="px-3 py-1.5 bg-zinc-700 hover:bg-zinc-600 text-white text-sm rounded transition-colors"
            >
              Retry
            </button>
          )}
        </div>
      </div>

      {error && (
        <p className="mt-2 text-sm text-red-400">{error}</p>
      )}
    </div>
  );
}
