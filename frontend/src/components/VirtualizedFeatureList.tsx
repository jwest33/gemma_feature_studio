"use client";

import { useState, useMemo, useCallback } from "react";
import { FixedSizeList as List } from "react-window";
import type { FeatureActivation } from "@/types/analysis";
import { getActivationColor, getContrastColor } from "@/lib/colorScale";

interface VirtualizedFeatureListProps {
  features: FeatureActivation[];
  maxActivation: number;
  onFeatureClick?: (feature: FeatureActivation) => void;
  height?: number;
}

export function VirtualizedFeatureList({
  features,
  maxActivation,
  onFeatureClick,
  height = 400,
}: VirtualizedFeatureListProps) {
  const [filter, setFilter] = useState("");
  const [minActivation, setMinActivation] = useState(0);

  const filteredFeatures = useMemo(() => {
    return features.filter((f) => {
      if (f.activation < minActivation) return false;
      if (filter && !String(f.id).includes(filter)) return false;
      return true;
    });
  }, [features, filter, minActivation]);

  const Row = useCallback(
    ({ index, style }: { index: number; style: React.CSSProperties }) => {
      const feature = filteredFeatures[index];
      const bgColor = getActivationColor(feature.activation, maxActivation);
      const barWidth = (feature.activation / maxActivation) * 100;

      return (
        <div
          style={style}
          className="flex items-center gap-3 px-3 py-1 border-b border-zinc-800 hover:bg-zinc-800/50 cursor-pointer"
          onClick={() => onFeatureClick?.(feature)}
        >
          <span className="w-16 font-mono text-sm text-zinc-400">
            #{feature.id}
          </span>
          <div className="flex-1 h-4 bg-zinc-800 rounded overflow-hidden">
            <div
              className="h-full rounded"
              style={{
                width: `${barWidth}%`,
                backgroundColor: bgColor,
              }}
            />
          </div>
          <span className="w-20 text-right font-mono text-sm text-zinc-300">
            {feature.activation.toFixed(4)}
          </span>
        </div>
      );
    },
    [filteredFeatures, maxActivation, onFeatureClick]
  );

  return (
    <div className="bg-zinc-900 rounded-lg overflow-hidden">
      <div className="p-3 border-b border-zinc-800 space-y-2">
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="Filter by ID..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="flex-1 px-2 py-1 bg-zinc-800 border border-zinc-700 rounded text-sm text-white placeholder-zinc-500 focus:outline-none focus:border-blue-500"
          />
          <div className="flex items-center gap-2">
            <label className="text-xs text-zinc-500">Min:</label>
            <input
              type="number"
              min={0}
              step={0.1}
              value={minActivation}
              onChange={(e) => setMinActivation(parseFloat(e.target.value) || 0)}
              className="w-20 px-2 py-1 bg-zinc-800 border border-zinc-700 rounded text-sm text-white focus:outline-none focus:border-blue-500"
            />
          </div>
        </div>
        <div className="text-xs text-zinc-500">
          Showing {filteredFeatures.length} of {features.length} features
        </div>
      </div>

      {filteredFeatures.length > 0 ? (
        <List
          height={height}
          itemCount={filteredFeatures.length}
          itemSize={36}
          width="100%"
        >
          {Row}
        </List>
      ) : (
        <div className="flex items-center justify-center h-32 text-zinc-500">
          No features match the filter
        </div>
      )}
    </div>
  );
}
