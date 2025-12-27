"use client";

import { useMemo } from "react";
import { Group } from "@visx/group";
import { HeatmapRect } from "@visx/heatmap";
import { scaleLinear, scaleBand } from "@visx/scale";
import { interpolateViridis } from "d3-scale-chromatic";
import type { TokenActivations, FeatureActivation } from "@/types/analysis";

interface FeatureHeatmapProps {
  tokens: TokenActivations[];
  width?: number;
  height?: number;
  maxFeatures?: number;
}

interface HeatmapData {
  bin: number;
  bins: { bin: number; count: number }[];
}

export function FeatureHeatmap({
  tokens,
  width = 800,
  height = 400,
  maxFeatures = 20,
}: FeatureHeatmapProps) {
  const { data, allFeatureIds, maxActivation } = useMemo(() => {
    // Collect all unique feature IDs from top features
    const featureSet = new Set<number>();
    let max = 0;

    for (const token of tokens) {
      for (const feat of token.top_features.slice(0, maxFeatures)) {
        featureSet.add(feat.id);
        if (feat.activation > max) max = feat.activation;
      }
    }

    const featureIds = Array.from(featureSet).sort((a, b) => a - b);

    // Build heatmap data: rows are features, columns are tokens
    const heatmapData: HeatmapData[] = featureIds.map((featId) => ({
      bin: featId,
      bins: tokens.map((token, tokenIdx) => {
        const feat = token.top_features.find((f) => f.id === featId);
        return {
          bin: tokenIdx,
          count: feat?.activation || 0,
        };
      }),
    }));

    return {
      data: heatmapData,
      allFeatureIds: featureIds,
      maxActivation: max || 1,
    };
  }, [tokens, maxFeatures]);

  const margin = { top: 40, left: 80, right: 20, bottom: 60 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  const xScale = scaleBand<number>({
    domain: tokens.map((_, i) => i),
    range: [0, innerWidth],
    padding: 0.05,
  });

  const yScale = scaleBand<number>({
    domain: allFeatureIds,
    range: [0, innerHeight],
    padding: 0.05,
  });

  const colorScale = scaleLinear<string>({
    domain: [0, maxActivation],
    range: ["#1a1a2e", "#fbbf24"],
  });

  if (tokens.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 bg-zinc-900 rounded-lg text-zinc-500">
        No data to display
      </div>
    );
  }

  return (
    <div className="bg-zinc-900 rounded-lg p-4">
      <h3 className="text-sm font-medium text-zinc-400 mb-2">
        Feature Activation Heatmap
      </h3>
      <svg width={width} height={height}>
        <Group left={margin.left} top={margin.top}>
          <HeatmapRect
            data={data}
            xScale={(d) => xScale(d) ?? 0}
            yScale={(d) => yScale(d) ?? 0}
            colorScale={colorScale}
            binWidth={xScale.bandwidth()}
            binHeight={yScale.bandwidth()}
          >
            {(heatmap) =>
              heatmap.map((heatmapBins) =>
                heatmapBins.map((bin) => (
                  <rect
                    key={`heatmap-rect-${bin.row}-${bin.column}`}
                    x={bin.x}
                    y={bin.y}
                    width={bin.width}
                    height={bin.height}
                    fill={bin.color}
                    className="cursor-pointer hover:stroke-white hover:stroke-1"
                  >
                    <title>
                      {`Feature ${allFeatureIds[bin.row]}\nToken: "${tokens[bin.column]?.token}"\nActivation: ${bin.count?.toFixed(4)}`}
                    </title>
                  </rect>
                ))
              )
            }
          </HeatmapRect>

          {/* Y-axis labels (feature IDs) */}
          {allFeatureIds.slice(0, 15).map((featId) => (
            <text
              key={`y-label-${featId}`}
              x={-8}
              y={(yScale(featId) ?? 0) + yScale.bandwidth() / 2}
              textAnchor="end"
              alignmentBaseline="middle"
              className="fill-zinc-500 text-xs"
            >
              {featId}
            </text>
          ))}

          {/* X-axis labels (tokens) */}
          {tokens.map((token, i) => (
            <text
              key={`x-label-${i}`}
              x={(xScale(i) ?? 0) + xScale.bandwidth() / 2}
              y={innerHeight + 16}
              textAnchor="middle"
              className="fill-zinc-500 text-xs"
              transform={`rotate(-45, ${(xScale(i) ?? 0) + xScale.bandwidth() / 2}, ${innerHeight + 16})`}
            >
              {token.token.slice(0, 8)}
            </text>
          ))}
        </Group>

        {/* Axis labels */}
        <text
          x={margin.left - 50}
          y={margin.top + innerHeight / 2}
          textAnchor="middle"
          className="fill-zinc-400 text-xs"
          transform={`rotate(-90, ${margin.left - 50}, ${margin.top + innerHeight / 2})`}
        >
          Feature ID
        </text>
        <text
          x={margin.left + innerWidth / 2}
          y={height - 10}
          textAnchor="middle"
          className="fill-zinc-400 text-xs"
        >
          Tokens
        </text>
      </svg>
    </div>
  );
}
