"use client";

import { useCallback, useMemo, useState } from "react";
import { SteeringPanel, type SteeringFeatureConfig } from "./SteeringPanel";
import { ComparisonPanel } from "./ComparisonPanel";
import { usePersistedExperiment } from "@/hooks/usePersistedExperiment";
import type { SteeringFeature } from "@/types/analysis";

export function SteeringWorkspace() {
  const {
    experimentName,
    features,
    hasExperiment,
    rename,
    setAllFeatures,
    saveRun,
    exportToJson,
  } = usePersistedExperiment();

  const [isEditing, setIsEditing] = useState(false);
  const [editName, setEditName] = useState("");

  const handleFeaturesChange = useCallback(
    (newFeatures: SteeringFeatureConfig[]) => {
      setAllFeatures(newFeatures);
    },
    [setAllFeatures]
  );

  // Convert to the format expected by ComparisonPanel
  const steeringFeatures: SteeringFeature[] = useMemo(
    () =>
      features
        .filter((f) => f.enabled)
        .map((f) => ({
          feature_id: f.id,
          coefficient: f.coefficient,
        })),
    [features]
  );

  const handleStartEdit = useCallback(() => {
    setEditName(experimentName ?? "");
    setIsEditing(true);
  }, [experimentName]);

  const handleSaveEdit = useCallback(() => {
    if (editName.trim()) {
      rename(editName.trim());
    }
    setIsEditing(false);
  }, [editName, rename]);

  const handleExport = useCallback(() => {
    const json = exportToJson();
    if (!json) return;

    const filename = `${experimentName?.replace(/[^a-z0-9]/gi, "_") ?? "experiment"}.json`;
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }, [exportToJson, experimentName]);

  const handleSaveRun = useCallback(
    (
      prompt: string,
      baselineOutput: string,
      steeredOutput: string,
      temperature: number,
      maxTokens: number
    ) => {
      saveRun(prompt, baselineOutput, steeredOutput, temperature, maxTokens);
    },
    [saveRun]
  );

  if (!hasExperiment) {
    return (
      <div className="text-center py-12 text-zinc-500">
        <p>Loading experiment...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Experiment Header */}
      <div className="flex items-center justify-between bg-zinc-900 rounded-lg px-4 py-3 border border-zinc-800">
        <div className="flex items-center gap-3">
          {isEditing ? (
            <input
              type="text"
              value={editName}
              onChange={(e) => setEditName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleSaveEdit();
                if (e.key === "Escape") setIsEditing(false);
              }}
              onBlur={handleSaveEdit}
              autoFocus
              className="px-2 py-1 bg-zinc-800 border border-zinc-600 rounded text-white focus:outline-none focus:border-blue-500"
            />
          ) : (
            <h2
              className="text-lg font-medium text-white cursor-pointer hover:text-zinc-300"
              onClick={handleStartEdit}
              title="Click to rename"
            >
              {experimentName}
            </h2>
          )}
          <button
            onClick={handleStartEdit}
            className="p-1 text-zinc-500 hover:text-white transition-colors"
            title="Rename experiment"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"
              />
            </svg>
          </button>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-sm text-zinc-500">
            {features.length} feature{features.length !== 1 ? "s" : ""}
          </span>
          <button
            onClick={handleExport}
            className="px-3 py-1.5 text-sm text-zinc-400 border border-zinc-700 rounded hover:bg-zinc-800 transition-colors"
          >
            Export
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Steering Panel - 1/3 width */}
        <div className="lg:col-span-1">
          <h3 className="text-md font-medium text-white mb-2">Feature Steering</h3>
          <p className="text-sm text-zinc-500 mb-3">
            Configure features to amplify or suppress. Use Ctrl+Z/Ctrl+Shift+Z to undo/redo.
          </p>
          <SteeringPanel features={features} onChange={handleFeaturesChange} />
        </div>

        {/* Comparison Panel - 2/3 width */}
        <div className="lg:col-span-2">
          <h3 className="text-md font-medium text-white mb-2">Generation Comparison</h3>
          <p className="text-sm text-zinc-500 mb-3">
            Compare baseline and steered outputs. Runs are auto-saved to experiment.
          </p>
          <ComparisonPanel
            steeringFeatures={steeringFeatures}
            onSaveRun={handleSaveRun}
          />
        </div>
      </div>
    </div>
  );
}

// Re-export for external use
export type { SteeringFeatureConfig };
