"use client";

import { useCallback, useMemo, useState } from "react";
import {
  useFlowStore,
  useSteeringFeatures,
  useSteeringPanelExpanded,
  useSteeringNormalization,
  useSteeringClampFactor,
  useSteeringUnitNormalize,
  type GlobalSteeringFeature,
} from "@/state/flowStore";
import { SteeringSlider } from "./SteeringSlider";
import type { NormalizationMode } from "@/types/analysis";

interface GlobalSteeringPanelProps {
  onGenerate: () => void;
  isGenerating?: boolean;
  disabled?: boolean;
}

export function GlobalSteeringPanel({
  onGenerate,
  isGenerating = false,
  disabled = false,
}: GlobalSteeringPanelProps) {
  const features = useSteeringFeatures();
  const expanded = useSteeringPanelExpanded();
  const normalization = useSteeringNormalization();
  const clampFactor = useSteeringClampFactor();
  const unitNormalize = useSteeringUnitNormalize();

  const {
    toggleSteeringPanel,
    updateSteeringFeature,
    removeSteeringFeature,
    clearSteeringFeatures,
    setSteeringNormalization,
    setSteeringClampFactor,
    setSteeringUnitNormalize,
  } = useFlowStore();

  const [showAdvanced, setShowAdvanced] = useState(false);

  // Count of enabled features with non-zero coefficients
  const activeCount = useMemo(
    () => features.filter((f) => f.enabled && f.coefficient !== 0).length,
    [features]
  );

  const enabledCount = useMemo(
    () => features.filter((f) => f.enabled).length,
    [features]
  );

  const handleCoefficientChange = useCallback(
    (id: number, layer: number, coefficient: number) => {
      updateSteeringFeature(id, layer, { coefficient });
    },
    [updateSteeringFeature]
  );

  const handleToggle = useCallback(
    (id: number, layer: number, enabled: boolean) => {
      updateSteeringFeature(id, layer, { enabled });
    },
    [updateSteeringFeature]
  );

  const handleRemove = useCallback(
    (id: number, layer: number) => {
      removeSteeringFeature(id, layer);
    },
    [removeSteeringFeature]
  );

  const canGenerate = enabledCount > 0 && activeCount > 0 && !isGenerating && !disabled;

  return (
    <div className="border-b border-zinc-800">
      {/* Collapsible Header */}
      <button
        onClick={toggleSteeringPanel}
        className="w-full px-6 py-2 flex items-center justify-between hover:bg-zinc-900/50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <svg
            className={`w-4 h-4 text-zinc-400 transition-transform ${
              expanded ? "rotate-90" : ""
            }`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 5l7 7-7 7"
            />
          </svg>
          <span className="text-sm font-medium text-white">Steering Panel</span>
          {features.length > 0 && (
            <span className="px-2 py-0.5 text-xs rounded-full bg-blue-500/20 text-blue-400 border border-blue-500/30">
              {features.length} feature{features.length !== 1 ? "s" : ""}
              {activeCount > 0 && ` (${activeCount} active)`}
            </span>
          )}
        </div>
        {features.length === 0 && (
          <span className="text-xs text-zinc-500">
            Click "Add to Steering" on features to start
          </span>
        )}
      </button>

      {/* Expanded Content */}
      {expanded && (
        <div className="px-6 pb-4 space-y-4">
          {/* Feature List */}
          {features.length === 0 ? (
            <div className="text-center py-6 text-zinc-500 bg-zinc-900/50 rounded-lg border border-zinc-800">
              <p className="text-sm">No features added for steering</p>
              <p className="text-xs mt-1">
                Select a feature in the visualization and click "Add to Steering"
              </p>
            </div>
          ) : (
            <div className="space-y-2 max-h-[200px] overflow-y-auto pr-1">
              {features.map((feature) => (
                <SteeringSlider
                  key={`${feature.id}-${feature.layer}`}
                  featureId={feature.id}
                  layer={feature.layer}
                  featureName={feature.name}
                  value={feature.coefficient}
                  enabled={feature.enabled}
                  onChange={(v) => handleCoefficientChange(feature.id, feature.layer, v)}
                  onToggle={(e) => handleToggle(feature.id, feature.layer, e)}
                  onRemove={() => handleRemove(feature.id, feature.layer)}
                />
              ))}
            </div>
          )}

          {/* Controls Row */}
          {features.length > 0 && (
            <div className="flex items-center justify-between gap-4 pt-2 border-t border-zinc-800">
              <div className="flex items-center gap-2">
                {/* Clear All */}
                <button
                  onClick={clearSteeringFeatures}
                  className="px-3 py-1.5 text-xs text-zinc-400 hover:text-red-400 border border-zinc-700 rounded hover:border-red-500/30 transition-colors"
                >
                  Clear All
                </button>

                {/* Advanced Toggle */}
                <button
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="px-3 py-1.5 text-xs text-zinc-400 hover:text-white border border-zinc-700 rounded hover:bg-zinc-800 transition-colors"
                >
                  {showAdvanced ? "Hide Options" : "Options"}
                </button>
              </div>

              {/* Generate Button */}
              <button
                onClick={onGenerate}
                disabled={!canGenerate}
                className={`px-6 py-2 rounded-lg font-medium transition-colors flex items-center gap-2 ${
                  isGenerating
                    ? "bg-blue-600/50 text-blue-200 cursor-wait"
                    : canGenerate
                    ? "bg-blue-600 text-white hover:bg-blue-500"
                    : "bg-zinc-700 text-zinc-400 cursor-not-allowed"
                }`}
              >
                {isGenerating ? (
                  <>
                    <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Generating...
                  </>
                ) : (
                  `Generate with Steering${activeCount > 0 ? ` (${activeCount})` : ""}`
                )}
              </button>
            </div>
          )}

          {/* Advanced Options */}
          {showAdvanced && features.length > 0 && (
            <div className="space-y-3 pt-3 border-t border-zinc-800">
              {/* Normalization Mode */}
              <div>
                <label className="text-xs text-zinc-400 block mb-2">Normalization Mode</label>
                <div className="flex gap-2">
                  {(["preserve_norm", "clamp", "none"] as NormalizationMode[]).map((mode) => (
                    <button
                      key={mode}
                      onClick={() => setSteeringNormalization(mode)}
                      className={`px-3 py-1.5 text-xs rounded transition-colors ${
                        normalization === mode
                          ? "bg-blue-600 text-white"
                          : "bg-zinc-700 text-zinc-300 hover:bg-zinc-600"
                      }`}
                    >
                      {mode === "preserve_norm" ? "Preserve Norm" : mode === "clamp" ? "Clamp" : "None"}
                    </button>
                  ))}
                </div>
                <p className="text-xs text-zinc-500 mt-1">
                  {normalization === "preserve_norm" && "Rescale to maintain original activation norm"}
                  {normalization === "clamp" && "Allow bounded norm changes within a factor"}
                  {normalization === "none" && "Raw steering without normalization"}
                </p>
              </div>

              {/* Clamp Factor (only shown when clamp mode is selected) */}
              {normalization === "clamp" && (
                <div>
                  <div className="flex items-center justify-between">
                    <label className="text-xs text-zinc-400">Clamp Factor</label>
                    <span className="text-xs font-mono text-zinc-300">{clampFactor.toFixed(1)}x</span>
                  </div>
                  <input
                    type="range"
                    min="1.0"
                    max="3.0"
                    step="0.1"
                    value={clampFactor}
                    onChange={(e) => setSteeringClampFactor(parseFloat(e.target.value))}
                    className="w-full h-2 bg-zinc-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                  />
                </div>
              )}

              {/* Unit Normalize Toggle */}
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-xs text-zinc-400">Unit Normalize Decoder</label>
                  <p className="text-xs text-zinc-500">Normalize decoder vector to unit norm</p>
                </div>
                <button
                  onClick={() => setSteeringUnitNormalize(!unitNormalize)}
                  className={`w-10 h-5 rounded-full transition-colors ${
                    unitNormalize ? "bg-blue-600" : "bg-zinc-600"
                  }`}
                >
                  <div
                    className={`w-4 h-4 bg-white rounded-full transition-transform ${
                      unitNormalize ? "translate-x-5" : "translate-x-0.5"
                    }`}
                  />
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
