"use client";

import { useFlowStore } from "@/state/flowStore";
import {
  AVAILABLE_LAYERS,
  getLayerColor,
  getLayerBorderClass,
} from "@/types/flow";

interface LayerSelectorProps {
  disabled?: boolean;
}

export function LayerSelector({ disabled = false }: LayerSelectorProps) {
  const { selectedLayers, toggleLayer, systemStatus } = useFlowStore();
  const loadedLayers = systemStatus?.saes?.loaded_layers ?? [];

  return (
    <div className="flex items-center gap-4">
      <span className="text-sm text-zinc-400 font-medium">Layers:</span>
      <div className="flex items-center gap-3">
        {AVAILABLE_LAYERS.map((layer) => {
          const isSelected = selectedLayers.includes(layer);
          const isLoaded = loadedLayers.includes(layer);
          const color = getLayerColor(layer);
          const borderClass = getLayerBorderClass(layer);

          return (
            <label
              key={layer}
              className={`
                flex items-center gap-2 px-3 py-1.5 rounded-md cursor-pointer
                transition-all duration-150
                ${disabled ? "opacity-50 cursor-not-allowed" : ""}
                ${
                  isSelected
                    ? `bg-${color}-500/20 ${borderClass} border`
                    : "bg-zinc-800/50 border border-zinc-700 hover:border-zinc-500"
                }
              `}
            >
              <input
                type="checkbox"
                checked={isSelected}
                onChange={() => !disabled && toggleLayer(layer)}
                disabled={disabled}
                className="sr-only"
              />
              <span
                className={`
                  w-3 h-3 rounded-sm border-2 flex items-center justify-center
                  transition-colors
                  ${
                    isSelected
                      ? `bg-${color}-500 border-${color}-500`
                      : "border-zinc-500"
                  }
                `}
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
              {isLoaded && (
                <span className="w-1.5 h-1.5 rounded-full bg-green-500" title="Loaded" />
              )}
            </label>
          );
        })}
      </div>
    </div>
  );
}
