"use client";

import { useState, useCallback, useEffect, useMemo } from "react";
import debounce from "lodash/debounce";

// Layer color classes for consistent styling
const LAYER_COLORS: Record<number, { text: string; bg: string; border: string }> = {
  9: { text: "text-purple-400", bg: "bg-purple-500/20", border: "border-purple-500/30" },
  17: { text: "text-blue-400", bg: "bg-blue-500/20", border: "border-blue-500/30" },
  22: { text: "text-cyan-400", bg: "bg-cyan-500/20", border: "border-cyan-500/30" },
  29: { text: "text-green-400", bg: "bg-green-500/20", border: "border-green-500/30" },
};

function getLayerColors(layer?: number) {
  if (layer === undefined) return { text: "text-zinc-400", bg: "bg-zinc-500/20", border: "border-zinc-500/30" };
  return LAYER_COLORS[layer] || { text: "text-zinc-400", bg: "bg-zinc-500/20", border: "border-zinc-500/30" };
}

interface SteeringSliderProps {
  featureId: number;
  layer?: number;
  featureName?: string;
  value: number;
  enabled: boolean;
  onChange: (value: number) => void;
  onToggle: (enabled: boolean) => void;
  onRemove: () => void;
  min?: number;
  max?: number;
  step?: number;
  debounceMs?: number;
}

export function SteeringSlider({
  featureId,
  layer,
  featureName,
  value,
  enabled,
  onChange,
  onToggle,
  onRemove,
  min = -2,
  max = 2,
  step = 0.01,
  debounceMs = 300,
}: SteeringSliderProps) {
  const layerColors = getLayerColors(layer);
  const [localValue, setLocalValue] = useState(value);

  // Sync local value when prop changes externally
  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  // Debounced callback for server sync
  const debouncedOnChange = useMemo(
    () => debounce((v: number) => onChange(v), debounceMs),
    [onChange, debounceMs]
  );

  // Cleanup debounce on unmount
  useEffect(() => {
    return () => debouncedOnChange.cancel();
  }, [debouncedOnChange]);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const newValue = parseFloat(e.target.value);
      setLocalValue(newValue);
      debouncedOnChange(newValue);
    },
    [debouncedOnChange]
  );

  const handleReset = useCallback(() => {
    setLocalValue(0);
    onChange(0);
  }, [onChange]);

  // Calculate color based on value
  const getValueColor = (val: number): string => {
    if (val > 0) {
      const intensity = Math.min(val / max, 1);
      return `rgba(34, 197, 94, ${0.3 + intensity * 0.7})`; // green
    } else if (val < 0) {
      const intensity = Math.min(Math.abs(val) / Math.abs(min), 1);
      return `rgba(239, 68, 68, ${0.3 + intensity * 0.7})`; // red
    }
    return "rgba(161, 161, 170, 0.3)"; // neutral gray
  };

  const progressPercent = ((localValue - min) / (max - min)) * 100;

  return (
    <div
      className={`bg-zinc-900 rounded-lg p-3 border transition-opacity ${
        enabled ? "border-zinc-700" : "border-zinc-800 opacity-50"
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <button
            onClick={() => onToggle(!enabled)}
            className={`w-4 h-4 rounded border flex items-center justify-center transition-colors ${
              enabled
                ? "bg-blue-600 border-blue-500"
                : "bg-zinc-800 border-zinc-600"
            }`}
          >
            {enabled && (
              <svg
                className="w-3 h-3 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={3}
                  d="M5 13l4 4L19 7"
                />
              </svg>
            )}
          </button>
          <div className="flex items-center gap-2">
            {layer !== undefined && (
              <span className={`text-xs px-1.5 py-0.5 rounded ${layerColors.bg} ${layerColors.text} ${layerColors.border} border`}>
                L{layer}
              </span>
            )}
            <span className="text-sm font-medium text-zinc-300">
              #{featureId}
            </span>
            {featureName && (
              <span className="text-xs text-zinc-500 truncate max-w-[150px]">
                {featureName}
              </span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="number"
            min={min}
            max={max}
            step={step}
            value={localValue}
            onChange={(e) => {
              const val = parseFloat(e.target.value);
              if (!isNaN(val)) {
                const clamped = Math.max(min, Math.min(max, val));
                setLocalValue(clamped);
                debouncedOnChange(clamped);
              }
            }}
            disabled={!enabled}
            className="text-sm font-mono px-2 py-0.5 rounded min-w-[70px] text-center bg-transparent border-none focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:opacity-50"
            style={{ backgroundColor: getValueColor(localValue) }}
          />
          <button
            onClick={handleReset}
            className="text-zinc-500 hover:text-zinc-300 transition-colors p-1"
            title="Reset to 0"
          >
            <svg
              className="w-4 h-4"
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
          <button
            onClick={onRemove}
            className="text-zinc-500 hover:text-red-400 transition-colors p-1"
            title="Remove feature"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>
      </div>

      <div className="relative">
        <div className="absolute inset-0 h-2 bg-zinc-800 rounded-full top-1/2 -translate-y-1/2">
          <div
            className="absolute h-full rounded-full transition-all"
            style={{
              left: localValue >= 0 ? "50%" : `${progressPercent}%`,
              right: localValue >= 0 ? `${100 - progressPercent}%` : "50%",
              backgroundColor: localValue >= 0 ? "#22c55e" : "#ef4444",
            }}
          />
          <div className="absolute left-1/2 top-0 w-0.5 h-full bg-zinc-600 -translate-x-1/2" />
        </div>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={localValue}
          onChange={handleChange}
          disabled={!enabled}
          className="relative w-full h-5 appearance-none bg-transparent cursor-pointer disabled:cursor-not-allowed
            [&::-webkit-slider-thumb]:appearance-none
            [&::-webkit-slider-thumb]:w-4
            [&::-webkit-slider-thumb]:h-4
            [&::-webkit-slider-thumb]:rounded-full
            [&::-webkit-slider-thumb]:bg-white
            [&::-webkit-slider-thumb]:shadow-md
            [&::-webkit-slider-thumb]:cursor-pointer
            [&::-webkit-slider-thumb]:disabled:bg-zinc-500
            [&::-moz-range-thumb]:w-4
            [&::-moz-range-thumb]:h-4
            [&::-moz-range-thumb]:rounded-full
            [&::-moz-range-thumb]:bg-white
            [&::-moz-range-thumb]:border-none
            [&::-moz-range-thumb]:shadow-md
            [&::-moz-range-thumb]:cursor-pointer"
        />
      </div>

      <div className="flex justify-between text-xs text-zinc-600 mt-1">
        <span>{min}</span>
        <span>0</span>
        <span>+{max}</span>
      </div>
    </div>
  );
}
