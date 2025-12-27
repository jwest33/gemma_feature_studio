"use client";

import { useState, useCallback } from "react";
import { NetworkGraph } from "./NetworkGraph";
import type { TokenActivations } from "@/types/analysis";
import { usePersistedExperiment } from "@/hooks/usePersistedExperiment";

interface NetworkPanelProps {
  tokens: TokenActivations[];
}

export function NetworkPanel({ tokens }: NetworkPanelProps) {
  const { addFeature, features } = usePersistedExperiment();
  const [recentlyAdded, setRecentlyAdded] = useState<number | null>(null);

  const selectedFeatures = features.map((f) => f.id);

  const handleFeatureSelect = useCallback(
    (featureId: number) => {
      // Check if already added
      if (selectedFeatures.includes(featureId)) {
        return;
      }

      addFeature(featureId, 0.5);
      setRecentlyAdded(featureId);

      // Clear notification after 2 seconds
      setTimeout(() => {
        setRecentlyAdded(null);
      }, 2000);
    },
    [addFeature, selectedFeatures]
  );

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-medium text-white">Feature Network</h2>
          <p className="text-sm text-zinc-500">
            Visualize feature activations across tokens. Click nodes to add to steering.
          </p>
        </div>

        {recentlyAdded !== null && (
          <div className="px-3 py-1.5 bg-green-900/50 border border-green-700 rounded text-green-300 text-sm flex items-center gap-2 animate-pulse">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            Feature #{recentlyAdded} added to steering
          </div>
        )}
      </div>

      <NetworkGraph
        tokens={tokens}
        onFeatureSelect={handleFeatureSelect}
        selectedFeatures={selectedFeatures}
      />
    </div>
  );
}

// Re-export for convenience
export { NetworkGraph } from "./NetworkGraph";
export { FeatureNode } from "./FeatureNode";
