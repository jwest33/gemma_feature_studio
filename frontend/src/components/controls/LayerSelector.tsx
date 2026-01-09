"use client";

import { useEffect, useCallback } from "react";
import { useFlowStore, useHasHydrated } from "@/state/flowStore";
import { getLayerHexByPosition } from "@/types/flow";
import { getSAECacheStatus, downloadSAEs } from "@/lib/api";

interface LayerSelectorProps {
  disabled?: boolean;
}

export function LayerSelector({ disabled = false }: LayerSelectorProps) {
  const {
    selectedLayers,
    toggleLayer,
    systemStatus,
    getAvailableLayers,
    saeCacheStatus,
    isCheckingCache,
    isDownloadingSAEs,
    setSAECacheStatus,
    setIsCheckingCache,
    setIsDownloadingSAEs,
    modelConfig,
  } = useFlowStore();
  const hasHydrated = useHasHydrated();
  const loadedLayers = systemStatus?.saes?.loaded_layers ?? [];
  const availableLayers = getAvailableLayers();

  // Check cache status when component mounts or SAE config changes
  const checkCacheStatus = useCallback(async () => {
    if (availableLayers.length === 0) return;

    setIsCheckingCache(true);
    try {
      const response = await getSAECacheStatus(availableLayers);
      const statusMap: Record<number, boolean> = {};
      for (const layer of response.layers) {
        statusMap[layer.layer] = layer.cached;
      }
      setSAECacheStatus(statusMap);
    } catch (error) {
      console.error("Failed to check SAE cache status:", error);
    } finally {
      setIsCheckingCache(false);
    }
  }, [availableLayers, setSAECacheStatus, setIsCheckingCache]);

  // Check cache status on mount and when config changes
  useEffect(() => {
    if (hasHydrated) {
      checkCacheStatus();
    }
  }, [hasHydrated, modelConfig.saePresetId, modelConfig.saeL0, modelConfig.modelSize, checkCacheStatus]);

  // Handle downloading uncached SAEs
  const handleDownload = useCallback(async () => {
    const uncachedSelected = selectedLayers.filter(
      (layer) => saeCacheStatus[layer] === false
    );
    if (uncachedSelected.length === 0) return;

    setIsDownloadingSAEs(true);
    try {
      const response = await downloadSAEs(uncachedSelected);
      // Update cache status for downloaded layers
      const newStatus = { ...saeCacheStatus };
      for (const layer of response.downloaded) {
        newStatus[layer] = true;
      }
      setSAECacheStatus(newStatus);
    } catch (error) {
      console.error("Failed to download SAEs:", error);
    } finally {
      setIsDownloadingSAEs(false);
    }
  }, [selectedLayers, saeCacheStatus, setSAECacheStatus, setIsDownloadingSAEs]);

  // Count uncached selected layers
  const uncachedSelectedCount = selectedLayers.filter(
    (layer) => saeCacheStatus[layer] === false
  ).length;

  // Show loading skeleton until hydration is complete to prevent SSR mismatch
  if (!hasHydrated) {
    return (
      <div className="flex items-center gap-4">
        <span className="text-sm text-zinc-400 font-medium">Layers:</span>
        <div className="flex items-center gap-3">
          {[1, 2, 3, 4].map((i) => (
            <div
              key={i}
              className="w-24 h-8 bg-zinc-800/50 rounded-md animate-pulse"
            />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-4">
      <span className="text-sm text-zinc-400 font-medium">Layers:</span>
      <div className="flex items-center gap-3">
        {availableLayers.map((layer) => {
          const isSelected = selectedLayers.includes(layer);
          const isLoaded = loadedLayers.includes(layer);
          const isCached = saeCacheStatus[layer];
          const color = getLayerHexByPosition(layer, availableLayers);

          return (
            <label
              key={layer}
              className={`
                flex items-center gap-2 px-3 py-1.5 rounded-md cursor-pointer
                transition-all duration-150
                ${disabled ? "opacity-50 cursor-not-allowed" : ""}
                ${!isSelected ? "bg-zinc-800/50 border border-zinc-700 hover:border-zinc-500" : "border"}
              `}
              style={isSelected ? {
                backgroundColor: `${color}20`,
                borderColor: color,
              } : undefined}
            >
              <input
                type="checkbox"
                checked={isSelected}
                onChange={() => !disabled && toggleLayer(layer)}
                disabled={disabled}
                className="sr-only"
              />
              <span
                className="w-3 h-3 rounded-sm border-2 flex items-center justify-center transition-colors"
                style={isSelected ? {
                  backgroundColor: color,
                  borderColor: color,
                } : {
                  borderColor: "#71717a",
                }}
              >
                {isSelected && (
                  <svg
                    className="w-2 h-2 text-white"
                    fill="currentColor"
                    viewBox="0 0 12 12"
                  >
                    <path d="M10.28 2.28L3.989 8.575 1.695 6.28A1 1 0 00.28 7.695l3 3a1 1 0 001.414 0l7-7A1 1 0 0010.28 2.28z" />
                  </svg>
                )}
              </span>
              <span
                className={`text-sm font-medium ${
                  isSelected ? "text-white" : "text-zinc-400"
                }`}
              >
                Layer {layer}
              </span>
              {/* Status indicators: VRAM loaded > cached on disk > not cached */}
              {isLoaded ? (
                <span
                  className="w-1.5 h-1.5 rounded-full bg-green-500"
                  title="Loaded in VRAM"
                />
              ) : isCached === true ? (
                <span
                  className="w-1.5 h-1.5 rounded-full bg-blue-500"
                  title="Cached on disk"
                />
              ) : isCached === false ? (
                <span
                  className="w-1.5 h-1.5 rounded-full bg-amber-500"
                  title="Not cached - will download"
                />
              ) : isCheckingCache ? (
                <span
                  className="w-1.5 h-1.5 rounded-full bg-zinc-500 animate-pulse"
                  title="Checking cache..."
                />
              ) : null}
            </label>
          );
        })}
      </div>

      {/* Download button for uncached selected layers */}
      {uncachedSelectedCount > 0 && (
        <button
          onClick={handleDownload}
          disabled={isDownloadingSAEs || disabled}
          className={`
            flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium
            transition-all duration-150
            ${isDownloadingSAEs
              ? "bg-amber-600/50 text-amber-200 cursor-wait"
              : "bg-amber-600 hover:bg-amber-500 text-white"
            }
            ${disabled ? "opacity-50 cursor-not-allowed" : ""}
          `}
        >
          {isDownloadingSAEs ? (
            <>
              <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              <span>Downloading...</span>
            </>
          ) : (
            <>
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              <span>Download {uncachedSelectedCount} SAE{uncachedSelectedCount > 1 ? "s" : ""}</span>
            </>
          )}
        </button>
      )}
    </div>
  );
}
