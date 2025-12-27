"use client";

import { useEffect, useState } from "react";
import { useFlowStore } from "@/state/flowStore";
import { getSystemStatus } from "@/lib/api";
import type { VRAMStatus } from "@/types/flow";

interface VRAMBarProps {
  status: VRAMStatus;
}

function VRAMBar({ status }: VRAMBarProps) {
  const usedPercent = ((status.total_gb - status.free_gb) / status.total_gb) * 100;

  const getPressureColor = (pressure: string) => {
    switch (pressure) {
      case "low":
        return "bg-green-500";
      case "moderate":
        return "bg-yellow-500";
      case "high":
        return "bg-orange-500";
      case "critical":
        return "bg-red-500";
      default:
        return "bg-zinc-500";
    }
  };

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-zinc-400">VRAM</span>
        <span className="text-zinc-300 font-mono">
          {(status.total_gb - status.free_gb).toFixed(1)} / {status.total_gb.toFixed(1)} GB
        </span>
      </div>
      <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
        <div
          className={`h-full ${getPressureColor(status.pressure)} transition-all duration-300`}
          style={{ width: `${usedPercent}%` }}
        />
      </div>
    </div>
  );
}

interface VRAMMonitorProps {
  compact?: boolean;
  showRefresh?: boolean;
}

export function VRAMMonitor({ compact = false, showRefresh = true }: VRAMMonitorProps) {
  const { systemStatus, setSystemStatus, setLoadingStatus, isLoadingStatus } =
    useFlowStore();
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = async () => {
    setLoadingStatus(true);
    setError(null);
    try {
      const status = await getSystemStatus();
      setSystemStatus(status);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoadingStatus(false);
    }
  };

  // Fetch status on mount and every 30 seconds
  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  if (error) {
    return (
      <div className="flex items-center gap-2 text-red-400 text-sm">
        <span>Error: {error}</span>
        <button
          onClick={fetchStatus}
          className="text-xs underline hover:text-red-300"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!systemStatus) {
    return (
      <div className="flex items-center gap-2 text-zinc-500 text-sm">
        <div className="w-3 h-3 border-2 border-zinc-500 border-t-transparent rounded-full animate-spin" />
        <span>Loading...</span>
      </div>
    );
  }

  if (compact) {
    return (
      <div className="flex items-center gap-3">
        {/* Model Status */}
        <div className="flex items-center gap-1.5">
          <div
            className={`w-2 h-2 rounded-full ${
              systemStatus.model_loaded ? "bg-green-500" : "bg-zinc-500"
            }`}
          />
          <span className="text-xs text-zinc-400">
            {systemStatus.model_loaded ? "Model Loaded" : "No Model"}
          </span>
        </div>

        {/* VRAM Mini Bar */}
        <div className="flex items-center gap-2">
          <div className="w-24 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
            <div
              className={`h-full ${
                systemStatus.vram.pressure === "low"
                  ? "bg-green-500"
                  : systemStatus.vram.pressure === "moderate"
                  ? "bg-yellow-500"
                  : systemStatus.vram.pressure === "high"
                  ? "bg-orange-500"
                  : "bg-red-500"
              }`}
              style={{
                width: `${
                  ((systemStatus.vram.total_gb - systemStatus.vram.free_gb) /
                    systemStatus.vram.total_gb) *
                  100
                }%`,
              }}
            />
          </div>
          <span className="text-xs text-zinc-400 font-mono">
            {systemStatus.vram.free_gb.toFixed(1)}GB free
          </span>
        </div>

        {/* Loaded SAEs Count */}
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-zinc-400">SAEs:</span>
          <span className="text-xs text-zinc-300 font-mono">
            {systemStatus.saes.loaded_count}
          </span>
        </div>

        {showRefresh && (
          <button
            onClick={fetchStatus}
            disabled={isLoadingStatus}
            className="p-1 text-zinc-500 hover:text-zinc-300 transition-colors disabled:opacity-50"
            title="Refresh status"
          >
            <svg
              className={`w-3.5 h-3.5 ${isLoadingStatus ? "animate-spin" : ""}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
              />
            </svg>
          </button>
        )}
      </div>
    );
  }

  // Full view
  return (
    <div className="bg-zinc-900 rounded-lg p-4 border border-zinc-800 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-white">System Status</h3>
        {showRefresh && (
          <button
            onClick={fetchStatus}
            disabled={isLoadingStatus}
            className="p-1.5 text-zinc-500 hover:text-zinc-300 transition-colors disabled:opacity-50"
            title="Refresh status"
          >
            <svg
              className={`w-4 h-4 ${isLoadingStatus ? "animate-spin" : ""}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
              />
            </svg>
          </button>
        )}
      </div>

      {/* Model Status */}
      <div className="flex items-center gap-2">
        <div
          className={`w-2.5 h-2.5 rounded-full ${
            systemStatus.model_loaded ? "bg-green-500" : "bg-zinc-500"
          }`}
        />
        <span className="text-sm text-zinc-300">
          {systemStatus.model_loaded
            ? systemStatus.model_name || "Model Loaded"
            : "No Model Loaded"}
        </span>
      </div>

      {/* VRAM Bar */}
      <VRAMBar status={systemStatus.vram} />

      {/* Loaded SAEs */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="text-zinc-400">Loaded SAEs</span>
          <span className="text-zinc-300">
            {systemStatus.saes.loaded_count} ({systemStatus.saes.total_size_gb.toFixed(2)} GB)
          </span>
        </div>
        {systemStatus.saes.loaded_layers.length > 0 ? (
          <div className="flex flex-wrap gap-1.5">
            {systemStatus.saes.loaded_layers.map((layer) => (
              <span
                key={layer}
                className="px-2 py-0.5 bg-zinc-800 text-zinc-300 text-xs rounded"
              >
                Layer {layer}
              </span>
            ))}
          </div>
        ) : (
          <span className="text-xs text-zinc-500">No SAEs loaded</span>
        )}
      </div>
    </div>
  );
}
