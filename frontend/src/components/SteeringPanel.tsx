"use client";

import { useCallback, useState } from "react";
import { SteeringSlider } from "./SteeringSlider";
import { useUndoableState } from "@/hooks/useUndoableState";

export interface SteeringFeatureConfig {
  id: number;
  name?: string;
  coefficient: number;
  enabled: boolean;
}

interface SteeringPanelProps {
  features: SteeringFeatureConfig[];
  onChange: (features: SteeringFeatureConfig[]) => void;
  onAddFeature?: (featureId: number) => void;
}

export function SteeringPanel({
  features: externalFeatures,
  onChange,
  onAddFeature,
}: SteeringPanelProps) {
  const {
    state: features,
    setState: setFeatures,
    undo,
    redo,
    canUndo,
    canRedo,
  } = useUndoableState<SteeringFeatureConfig[]>(externalFeatures, {
    historyLimit: 30,
    enableKeyboardShortcuts: true,
  });

  const [newFeatureId, setNewFeatureId] = useState("");

  // Sync with parent on changes
  const updateFeatures = useCallback(
    (newFeatures: SteeringFeatureConfig[]) => {
      setFeatures(newFeatures);
      onChange(newFeatures);
    },
    [setFeatures, onChange]
  );

  const handleCoefficientChange = useCallback(
    (featureId: number, coefficient: number) => {
      updateFeatures(
        features.map((f) =>
          f.id === featureId ? { ...f, coefficient } : f
        )
      );
    },
    [features, updateFeatures]
  );

  const handleToggle = useCallback(
    (featureId: number, enabled: boolean) => {
      updateFeatures(
        features.map((f) => (f.id === featureId ? { ...f, enabled } : f))
      );
    },
    [features, updateFeatures]
  );

  const handleRemove = useCallback(
    (featureId: number) => {
      updateFeatures(features.filter((f) => f.id !== featureId));
    },
    [features, updateFeatures]
  );

  const handleResetAll = useCallback(() => {
    updateFeatures(
      features.map((f) => ({ ...f, coefficient: 0, enabled: true }))
    );
  }, [features, updateFeatures]);

  const handleDisableAll = useCallback(() => {
    updateFeatures(features.map((f) => ({ ...f, enabled: false })));
  }, [features, updateFeatures]);

  const handleEnableAll = useCallback(() => {
    updateFeatures(features.map((f) => ({ ...f, enabled: true })));
  }, [features, updateFeatures]);

  const handleAddFeature = useCallback(() => {
    const id = parseInt(newFeatureId, 10);
    if (isNaN(id) || id < 0) return;
    if (features.some((f) => f.id === id)) return;

    const newFeature: SteeringFeatureConfig = {
      id,
      coefficient: 0,
      enabled: true,
    };
    updateFeatures([...features, newFeature]);
    setNewFeatureId("");
    onAddFeature?.(id);
  }, [newFeatureId, features, updateFeatures, onAddFeature]);

  const activeCount = features.filter((f) => f.enabled).length;

  return (
    <div className="bg-zinc-900 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-zinc-800">
        <div>
          <h3 className="font-medium text-white">Steering Configuration</h3>
          <p className="text-xs text-zinc-500 mt-0.5">
            {features.length} feature{features.length !== 1 ? "s" : ""} ({activeCount} active)
          </p>
        </div>

        <div className="flex items-center gap-2">
          {/* Undo/Redo */}
          <div className="flex items-center border border-zinc-700 rounded overflow-hidden">
            <button
              onClick={undo}
              disabled={!canUndo}
              className="p-1.5 text-zinc-400 hover:text-white hover:bg-zinc-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
              title="Undo (Ctrl+Z)"
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
                  d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6"
                />
              </svg>
            </button>
            <button
              onClick={redo}
              disabled={!canRedo}
              className="p-1.5 text-zinc-400 hover:text-white hover:bg-zinc-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors border-l border-zinc-700"
              title="Redo (Ctrl+Shift+Z)"
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
                  d="M21 10h-10a8 8 0 00-8 8v2M21 10l-6 6m6-6l-6-6"
                />
              </svg>
            </button>
          </div>

          {/* Actions dropdown */}
          <div className="relative group">
            <button className="px-3 py-1.5 text-sm text-zinc-400 hover:text-white border border-zinc-700 rounded hover:bg-zinc-800 transition-colors">
              Actions
            </button>
            <div className="absolute right-0 top-full mt-1 w-40 bg-zinc-800 border border-zinc-700 rounded shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10">
              <button
                onClick={handleResetAll}
                className="w-full px-3 py-2 text-left text-sm text-zinc-300 hover:bg-zinc-700 transition-colors"
              >
                Reset all to 0
              </button>
              <button
                onClick={handleEnableAll}
                className="w-full px-3 py-2 text-left text-sm text-zinc-300 hover:bg-zinc-700 transition-colors"
              >
                Enable all
              </button>
              <button
                onClick={handleDisableAll}
                className="w-full px-3 py-2 text-left text-sm text-zinc-300 hover:bg-zinc-700 transition-colors"
              >
                Disable all
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Add Feature */}
      <div className="p-3 border-b border-zinc-800">
        <div className="flex gap-2">
          <input
            type="number"
            placeholder="Feature ID"
            value={newFeatureId}
            onChange={(e) => setNewFeatureId(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleAddFeature()}
            className="flex-1 px-3 py-1.5 bg-zinc-800 border border-zinc-700 rounded text-sm text-white placeholder-zinc-500 focus:outline-none focus:border-zinc-600"
          />
          <button
            onClick={handleAddFeature}
            disabled={!newFeatureId}
            className="px-4 py-1.5 bg-blue-600 text-white text-sm rounded hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Add
          </button>
        </div>
      </div>

      {/* Feature List */}
      <div className="p-3 space-y-2 max-h-[400px] overflow-y-auto">
        {features.length === 0 ? (
          <div className="text-center py-8 text-zinc-500">
            <p>No features selected for steering</p>
            <p className="text-xs mt-1">
              Click features in the analysis panel or add by ID above
            </p>
          </div>
        ) : (
          features.map((feature) => (
            <SteeringSlider
              key={feature.id}
              featureId={feature.id}
              featureName={feature.name}
              value={feature.coefficient}
              enabled={feature.enabled}
              onChange={(v) => handleCoefficientChange(feature.id, v)}
              onToggle={(e) => handleToggle(feature.id, e)}
              onRemove={() => handleRemove(feature.id)}
            />
          ))
        )}
      </div>

      {/* Summary */}
      {features.length > 0 && (
        <div className="p-3 border-t border-zinc-800 bg-zinc-900/50">
          <div className="text-xs text-zinc-500">
            <span className="font-medium text-zinc-400">Active steering: </span>
            {features
              .filter((f) => f.enabled && f.coefficient !== 0)
              .map((f) => `#${f.id}: ${f.coefficient > 0 ? "+" : ""}${f.coefficient.toFixed(1)}`)
              .join(", ") || "None"}
          </div>
        </div>
      )}
    </div>
  );
}
