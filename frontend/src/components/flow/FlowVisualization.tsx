"use client";

import { useMemo, useRef, useState } from "react";
import { useFlowStore } from "@/state/flowStore";
import type { LayerActivations } from "@/types/analysis";
import { createActivationScale, getContrastColor } from "@/lib/colorScale";
import {
  deduplicateFeatures,
  type LayerFlowData,
  type DeduplicatedFeature,
} from "@/lib/flowDataUtils";

// ============================================================================
// Types
// ============================================================================

interface FlowVisualizationProps {
  generatedOutput?: string;
  isGenerating?: boolean;
}

interface FeatureNodePosition {
  x: number;
  y: number;
  feature: DeduplicatedFeature;
  layer: number;
}

interface TokenNodePosition {
  x: number;
  y: number;
  tokenIndex: number;
  token: string;
}

// ============================================================================
// Constants
// ============================================================================

const LAYOUT = {
  tokenColumnWidth: 100,
  layerColumnWidth: 140,
  layerGap: 80,
  tokenHeight: 28,
  tokenGap: 4,
  featureHeight: 28,
  featureGap: 6,
  headerHeight: 40,
  padding: 16,
  outputColumnWidth: 200,
};

const LAYER_HEX_COLORS: Record<number, string> = {
  9: "#a855f7",   // purple-500
  17: "#3b82f6",  // blue-500
  22: "#06b6d4",  // cyan-500
  29: "#22c55e",  // green-500
};

// ============================================================================
// SVG Flow Lines Component
// ============================================================================

interface FlowLinesProps {
  tokenPositions: TokenNodePosition[];
  featurePositions: Map<string, FeatureNodePosition>;
  layerFlowData: LayerFlowData[];
  selectedTokenIndex: number | null;
  selectedFeatureId: number | null;
  selectedLayer: number | null;
  hoveredToken: number | null;
  hoveredFeature: { id: number; layer: number } | null;
  maxActivation: number;
}

function FlowLines({
  tokenPositions,
  featurePositions,
  layerFlowData,
  selectedTokenIndex,
  selectedFeatureId,
  selectedLayer,
  hoveredToken,
  hoveredFeature,
  maxActivation,
}: FlowLinesProps) {
  const lines: JSX.Element[] = [];

  // Sort layers for sequential connections
  const sortedLayers = [...layerFlowData].sort((a, b) => a.layer - b.layer);

  // Draw token-to-feature connections for the first layer
  if (sortedLayers.length > 0) {
    const firstLayer = sortedLayers[0];

    firstLayer.features.forEach((feature) => {
      const featureKey = `${firstLayer.layer}-${feature.featureId}`;
      const featurePos = featurePositions.get(featureKey);
      if (!featurePos) return;

      feature.activations.forEach((activation) => {
        const tokenPos = tokenPositions[activation.tokenIndex];
        if (!tokenPos) return;

        const isHighlighted =
          (hoveredToken !== null && hoveredToken === activation.tokenIndex) ||
          (hoveredFeature !== null &&
           hoveredFeature.id === feature.featureId &&
           hoveredFeature.layer === firstLayer.layer) ||
          (selectedTokenIndex !== null && selectedTokenIndex === activation.tokenIndex) ||
          (selectedFeatureId !== null && selectedFeatureId === feature.featureId);

        const opacity = isHighlighted ? 0.8 : 0.15;
        const strokeWidth = isHighlighted ? 2 : 1;
        const color = LAYER_HEX_COLORS[firstLayer.layer] || "#888";

        // Bezier curve from token to feature
        const x1 = tokenPos.x + LAYOUT.tokenColumnWidth;
        const y1 = tokenPos.y + LAYOUT.tokenHeight / 2;
        const x2 = featurePos.x;
        const y2 = featurePos.y + LAYOUT.featureHeight / 2;
        const cx1 = x1 + (x2 - x1) * 0.4;
        const cx2 = x2 - (x2 - x1) * 0.4;

        lines.push(
          <path
            key={`token-${activation.tokenIndex}-to-${featureKey}`}
            d={`M ${x1} ${y1} C ${cx1} ${y1}, ${cx2} ${y2}, ${x2} ${y2}`}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            opacity={opacity}
            className="transition-opacity duration-150"
          />
        );
      });
    });
  }

  // Draw cross-layer feature connections
  for (let i = 0; i < sortedLayers.length - 1; i++) {
    const currentLayer = sortedLayers[i];
    const nextLayer = sortedLayers[i + 1];

    // For each feature in current layer that also exists in next layer
    currentLayer.features.forEach((currentFeature) => {
      const currentKey = `${currentLayer.layer}-${currentFeature.featureId}`;
      const currentPos = featurePositions.get(currentKey);
      if (!currentPos) return;

      // Check if same feature exists in next layer
      const nextFeature = nextLayer.features.find(
        (f) => f.featureId === currentFeature.featureId
      );
      if (!nextFeature) return;

      const nextKey = `${nextLayer.layer}-${nextFeature.featureId}`;
      const nextPos = featurePositions.get(nextKey);
      if (!nextPos) return;

      // Find shared tokens
      const currentTokens = new Set(currentFeature.activations.map((a) => a.tokenIndex));
      const hasSharedTokens = nextFeature.activations.some((a) =>
        currentTokens.has(a.tokenIndex)
      );

      if (!hasSharedTokens) return;

      const isHighlighted =
        (selectedFeatureId !== null && selectedFeatureId === currentFeature.featureId) ||
        (hoveredFeature !== null && hoveredFeature.id === currentFeature.featureId);

      const opacity = isHighlighted ? 0.9 : 0.2;
      const strokeWidth = isHighlighted ? 2.5 : 1.5;

      // Gradient from current layer color to next layer color
      const gradientId = `gradient-${currentLayer.layer}-${nextLayer.layer}-${currentFeature.featureId}`;

      const x1 = currentPos.x + LAYOUT.layerColumnWidth;
      const y1 = currentPos.y + LAYOUT.featureHeight / 2;
      const x2 = nextPos.x;
      const y2 = nextPos.y + LAYOUT.featureHeight / 2;
      const cx1 = x1 + (x2 - x1) * 0.4;
      const cx2 = x2 - (x2 - x1) * 0.4;

      lines.push(
        <defs key={`defs-${gradientId}`}>
          <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={LAYER_HEX_COLORS[currentLayer.layer] || "#888"} />
            <stop offset="100%" stopColor={LAYER_HEX_COLORS[nextLayer.layer] || "#888"} />
          </linearGradient>
        </defs>
      );

      lines.push(
        <path
          key={`layer-${currentLayer.layer}-to-${nextLayer.layer}-feature-${currentFeature.featureId}`}
          d={`M ${x1} ${y1} C ${cx1} ${y1}, ${cx2} ${y2}, ${x2} ${y2}`}
          fill="none"
          stroke={`url(#${gradientId})`}
          strokeWidth={strokeWidth}
          opacity={opacity}
          className="transition-opacity duration-150"
        />
      );
    });
  }

  // Draw connections from last layer to tokens that weren't in previous layers
  // (showing how activations flow to tokens that only appear in later layers)
  for (let i = 1; i < sortedLayers.length; i++) {
    const prevLayer = sortedLayers[i - 1];
    const currentLayer = sortedLayers[i];
    const prevFeatureIds = new Set(prevLayer.features.map((f) => f.featureId));

    currentLayer.features.forEach((feature) => {
      // Only for features NOT in previous layer
      if (prevFeatureIds.has(feature.featureId)) return;

      const featureKey = `${currentLayer.layer}-${feature.featureId}`;
      const featurePos = featurePositions.get(featureKey);
      if (!featurePos) return;

      feature.activations.forEach((activation) => {
        const tokenPos = tokenPositions[activation.tokenIndex];
        if (!tokenPos) return;

        const isHighlighted =
          (hoveredToken !== null && hoveredToken === activation.tokenIndex) ||
          (hoveredFeature !== null &&
            hoveredFeature.id === feature.featureId &&
            hoveredFeature.layer === currentLayer.layer) ||
          (selectedTokenIndex !== null && selectedTokenIndex === activation.tokenIndex) ||
          (selectedFeatureId !== null && selectedFeatureId === feature.featureId);

        const opacity = isHighlighted ? 0.7 : 0.1;
        const strokeWidth = isHighlighted ? 1.5 : 0.5;
        const color = LAYER_HEX_COLORS[currentLayer.layer] || "#888";

        // Curved line from token to feature
        const x1 = tokenPos.x + LAYOUT.tokenColumnWidth;
        const y1 = tokenPos.y + LAYOUT.tokenHeight / 2;
        const x2 = featurePos.x;
        const y2 = featurePos.y + LAYOUT.featureHeight / 2;
        const cx1 = x1 + (x2 - x1) * 0.3;
        const cx2 = x2 - (x2 - x1) * 0.3;

        lines.push(
          <path
            key={`token-${activation.tokenIndex}-to-layer${currentLayer.layer}-${feature.featureId}`}
            d={`M ${x1} ${y1} C ${cx1} ${y1}, ${cx2} ${y2}, ${x2} ${y2}`}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            opacity={opacity}
            strokeDasharray="4 2"
            className="transition-opacity duration-150"
          />
        );
      });
    });
  }

  return <g>{lines}</g>;
}

// ============================================================================
// Token Node Component
// ============================================================================

interface TokenNodeProps {
  token: string;
  index: number;
  x: number;
  y: number;
  isSelected: boolean;
  isHovered: boolean;
  onSelect: () => void;
  onHover: (hovered: boolean) => void;
}

function TokenNode({
  token,
  index,
  x,
  y,
  isSelected,
  isHovered,
  onSelect,
  onHover,
}: TokenNodeProps) {
  const displayToken = token.replace(/ /g, "\u00B7");

  return (
    <g
      transform={`translate(${x}, ${y})`}
      onClick={onSelect}
      onMouseEnter={() => onHover(true)}
      onMouseLeave={() => onHover(false)}
      className="cursor-pointer"
    >
      <rect
        x={0}
        y={0}
        width={LAYOUT.tokenColumnWidth}
        height={LAYOUT.tokenHeight}
        rx={4}
        fill={isSelected ? "rgba(59, 130, 246, 0.3)" : isHovered ? "rgba(255, 255, 255, 0.1)" : "rgba(39, 39, 42, 0.8)"}
        stroke={isSelected ? "#3b82f6" : "rgba(63, 63, 70, 0.8)"}
        strokeWidth={isSelected ? 2 : 1}
        className="transition-all duration-150"
      />
      <text
        x={8}
        y={LAYOUT.tokenHeight / 2}
        dominantBaseline="middle"
        className="text-[10px] fill-zinc-500 font-mono"
      >
        {index}
      </text>
      <text
        x={28}
        y={LAYOUT.tokenHeight / 2}
        dominantBaseline="middle"
        className="text-xs fill-zinc-200 font-mono"
      >
        {displayToken.length > 8 ? displayToken.slice(0, 8) + "..." : displayToken}
      </text>
    </g>
  );
}

// ============================================================================
// Feature Node Component
// ============================================================================

interface FeatureNodeProps {
  feature: DeduplicatedFeature;
  x: number;
  y: number;
  maxActivation: number;
  isSelected: boolean;
  isHovered: boolean;
  onSelect: () => void;
  onHover: (hovered: boolean) => void;
}

function FeatureNode({
  feature,
  x,
  y,
  maxActivation,
  isSelected,
  isHovered,
  onSelect,
  onHover,
}: FeatureNodeProps) {
  const colorScale = useMemo(() => createActivationScale(maxActivation), [maxActivation]);
  const bgColor = colorScale(feature.maxActivation);
  const textColor = getContrastColor(bgColor);
  const tokenCount = feature.activations.length;

  return (
    <g
      transform={`translate(${x}, ${y})`}
      onClick={onSelect}
      onMouseEnter={() => onHover(true)}
      onMouseLeave={() => onHover(false)}
      className="cursor-pointer"
    >
      <rect
        x={0}
        y={0}
        width={LAYOUT.layerColumnWidth}
        height={LAYOUT.featureHeight}
        rx={4}
        fill={bgColor}
        stroke={isSelected ? "#ffffff" : isHovered ? "rgba(255, 255, 255, 0.6)" : "transparent"}
        strokeWidth={isSelected ? 2 : isHovered ? 1.5 : 0}
        className="transition-all duration-150"
      />
      <text
        x={8}
        y={LAYOUT.featureHeight / 2}
        dominantBaseline="middle"
        fill={textColor}
        className="text-xs font-mono font-medium"
      >
        {feature.featureId}
      </text>
      <text
        x={LAYOUT.layerColumnWidth - 8}
        y={LAYOUT.featureHeight / 2}
        dominantBaseline="middle"
        textAnchor="end"
        fill={textColor}
        className="text-[10px] font-mono opacity-70"
      >
        {tokenCount}t
      </text>
      <title>
        Feature {feature.featureId} | Max: {feature.maxActivation.toFixed(4)} | Tokens: {tokenCount}
      </title>
    </g>
  );
}

// ============================================================================
// Layer Header Component
// ============================================================================

interface LayerHeaderProps {
  layer: number;
  x: number;
  enabled: boolean;
  featureCount: number;
}

function LayerHeader({ layer, x, enabled, featureCount }: LayerHeaderProps) {
  const color = LAYER_HEX_COLORS[layer] || "#888";

  return (
    <g transform={`translate(${x}, ${LAYOUT.padding})`}>
      <rect
        x={0}
        y={0}
        width={LAYOUT.layerColumnWidth}
        height={LAYOUT.headerHeight - 8}
        rx={6}
        fill={enabled ? `${color}20` : "rgba(39, 39, 42, 0.3)"}
        stroke={enabled ? color : "rgba(63, 63, 70, 0.5)"}
        strokeWidth={enabled ? 2 : 1}
      />
      <text
        x={LAYOUT.layerColumnWidth / 2}
        y={(LAYOUT.headerHeight - 8) / 2}
        textAnchor="middle"
        dominantBaseline="middle"
        fill={enabled ? color : "#71717a"}
        className="text-sm font-medium"
      >
        Layer {layer}
        {enabled && (
          <tspan className="text-[10px] opacity-70"> ({featureCount})</tspan>
        )}
      </text>
    </g>
  );
}

// ============================================================================
// Output Panel Component
// ============================================================================

interface OutputPanelProps {
  output?: string;
  isGenerating?: boolean;
  isSelected?: boolean;
  onSelect?: () => void;
  x: number;
  height: number;
}

function OutputPanel({ output, isGenerating, isSelected, onSelect, x, height }: OutputPanelProps) {
  return (
    <foreignObject x={x} y={LAYOUT.padding} width={LAYOUT.outputColumnWidth} height={height}>
      <div
        className={`h-full border-2 rounded-lg bg-green-500/5 overflow-hidden flex flex-col cursor-pointer transition-all ${
          isSelected ? "border-green-400 ring-2 ring-green-400/50" : "border-green-500/50 hover:border-green-400/70"
        }`}
        onClick={(e) => {
          e.stopPropagation();
          onSelect?.();
        }}
      >
        <div className="px-3 py-2 border-b border-green-500/50 bg-green-500/10 shrink-0">
          <span className="text-sm font-medium text-green-300">
            Generated Output
            {isGenerating && (
              <span className="ml-2 inline-block w-2 h-2 bg-green-400 rounded-full animate-pulse" />
            )}
            {isSelected && (
              <span className="ml-2 text-xs text-green-400">(click to copy below)</span>
            )}
          </span>
        </div>
        <div className="flex-1 p-3 overflow-y-auto">
          {output ? (
            <p className="text-sm text-zinc-300 whitespace-pre-wrap">{output}</p>
          ) : (
            <p className="text-sm text-zinc-500 italic">
              {isGenerating ? "Generating..." : "Output will appear here after generation"}
            </p>
          )}
        </div>
      </div>
    </foreignObject>
  );
}

// ============================================================================
// Pan/Zoom State and Helpers
// ============================================================================

interface Transform {
  x: number;
  y: number;
  scale: number;
}

const MIN_ZOOM = 0.25;
const MAX_ZOOM = 3;
const ZOOM_SENSITIVITY = 0.001;

// ============================================================================
// Main Flow Visualization Component
// ============================================================================

export function FlowVisualization({ generatedOutput, isGenerating }: FlowVisualizationProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoveredToken, setHoveredToken] = useState<number | null>(null);
  const [hoveredFeature, setHoveredFeature] = useState<{ id: number; layer: number } | null>(null);

  // Pan and zoom state
  const [transform, setTransform] = useState<Transform>({ x: 0, y: 0, scale: 1 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });

  const { selectedLayers, selection, selectToken, selectFeature, selectOutput } = useFlowStore();
  const activePrompt = useFlowStore((state) => state.getActivePrompt());

  // Get layer data from active prompt
  const layerDataMap = activePrompt?.layerData ?? new Map<number, LayerActivations>();
  const tokens = activePrompt?.tokens ?? [];

  // Compute deduplicated features per layer
  const layerFlowData = useMemo(() => {
    const data: LayerFlowData[] = [];

    selectedLayers.forEach((layer) => {
      const layerActivations = layerDataMap.get(layer);
      if (layerActivations?.token_activations) {
        data.push(deduplicateFeatures(layerActivations.token_activations, layer, 10));
      }
    });

    return data.sort((a, b) => a.layer - b.layer);
  }, [layerDataMap, selectedLayers]);

  // Find max activation across all layers for consistent coloring
  const globalMaxActivation = useMemo(() => {
    return Math.max(1, ...layerFlowData.map((l) => l.maxActivation));
  }, [layerFlowData]);

  // Compute token positions
  const tokenPositions: TokenNodePosition[] = useMemo(() => {
    return tokens.map((token, idx) => ({
      x: LAYOUT.padding,
      y: LAYOUT.headerHeight + LAYOUT.padding + idx * (LAYOUT.tokenHeight + LAYOUT.tokenGap),
      tokenIndex: idx,
      token,
    }));
  }, [tokens]);

  // Compute feature positions per layer
  const featurePositions = useMemo(() => {
    const positions = new Map<string, FeatureNodePosition>();
    const sortedLayers = [...selectedLayers].sort((a, b) => a - b);

    sortedLayers.forEach((layer, layerIndex) => {
      const layerData = layerFlowData.find((l) => l.layer === layer);
      if (!layerData) return;

      const x =
        LAYOUT.padding +
        LAYOUT.tokenColumnWidth +
        LAYOUT.layerGap +
        layerIndex * (LAYOUT.layerColumnWidth + LAYOUT.layerGap);

      layerData.features.forEach((feature, featureIndex) => {
        const y =
          LAYOUT.headerHeight +
          LAYOUT.padding +
          featureIndex * (LAYOUT.featureHeight + LAYOUT.featureGap);

        positions.set(`${layer}-${feature.featureId}`, {
          x,
          y,
          feature,
          layer,
        });
      });
    });

    return positions;
  }, [layerFlowData, selectedLayers]);

  // Calculate SVG dimensions
  const svgDimensions = useMemo(() => {
    const sortedLayers = [...selectedLayers].sort((a, b) => a - b);
    const maxFeatures = Math.max(1, ...layerFlowData.map((l) => l.features.length));

    const width =
      LAYOUT.padding * 2 +
      LAYOUT.tokenColumnWidth +
      LAYOUT.layerGap +
      sortedLayers.length * (LAYOUT.layerColumnWidth + LAYOUT.layerGap) +
      LAYOUT.outputColumnWidth;

    const height =
      LAYOUT.headerHeight +
      LAYOUT.padding * 2 +
      Math.max(
        tokens.length * (LAYOUT.tokenHeight + LAYOUT.tokenGap),
        maxFeatures * (LAYOUT.featureHeight + LAYOUT.featureGap)
      );

    return { width: Math.max(800, width), height: Math.max(400, height) };
  }, [tokens.length, layerFlowData, selectedLayers]);

  // Output column position
  const outputX = useMemo(() => {
    const sortedLayers = [...selectedLayers].sort((a, b) => a - b);
    return (
      LAYOUT.padding +
      LAYOUT.tokenColumnWidth +
      LAYOUT.layerGap +
      sortedLayers.length * (LAYOUT.layerColumnWidth + LAYOUT.layerGap)
    );
  }, [selectedLayers]);

  // Pan handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    // Only start panning with left mouse button and when not clicking on interactive elements
    if (e.button !== 0) return;
    setIsPanning(true);
    setPanStart({ x: e.clientX - transform.x, y: e.clientY - transform.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isPanning) return;
    setTransform((prev) => ({
      ...prev,
      x: e.clientX - panStart.x,
      y: e.clientY - panStart.y,
    }));
  };

  const handleMouseUp = () => {
    setIsPanning(false);
  };

  const handleMouseLeave = () => {
    setIsPanning(false);
  };

  // Zoom handler
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();

    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;

    // Get mouse position relative to SVG
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Calculate new scale
    const delta = -e.deltaY * ZOOM_SENSITIVITY;
    const newScale = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, transform.scale * (1 + delta)));

    // Calculate new position to zoom towards mouse cursor
    const scaleRatio = newScale / transform.scale;
    const newX = mouseX - (mouseX - transform.x) * scaleRatio;
    const newY = mouseY - (mouseY - transform.y) * scaleRatio;

    setTransform({ x: newX, y: newY, scale: newScale });
  };

  // Reset view handler
  const resetView = () => {
    setTransform({ x: 0, y: 0, scale: 1 });
  };

  if (!activePrompt || tokens.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-zinc-500">
        <div className="text-center">
          <p className="text-lg mb-2">No analysis data</p>
          <p className="text-sm">Add a prompt and click Analyze to see the flow visualization</p>
        </div>
      </div>
    );
  }

  const sortedLayers = [...selectedLayers].sort((a, b) => a - b);

  return (
    <div ref={containerRef} className="h-full overflow-hidden relative">
      {/* Zoom controls */}
      <div className="absolute top-4 right-4 z-10 flex gap-2">
        <button
          onClick={() => setTransform((prev) => ({ ...prev, scale: Math.min(MAX_ZOOM, prev.scale * 1.2) }))}
          className="w-8 h-8 bg-zinc-800 hover:bg-zinc-700 rounded flex items-center justify-center text-zinc-300 border border-zinc-700"
          title="Zoom in"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
        </button>
        <button
          onClick={() => setTransform((prev) => ({ ...prev, scale: Math.max(MIN_ZOOM, prev.scale / 1.2) }))}
          className="w-8 h-8 bg-zinc-800 hover:bg-zinc-700 rounded flex items-center justify-center text-zinc-300 border border-zinc-700"
          title="Zoom out"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
          </svg>
        </button>
        <button
          onClick={resetView}
          className="px-2 h-8 bg-zinc-800 hover:bg-zinc-700 rounded flex items-center justify-center text-zinc-300 text-xs border border-zinc-700"
          title="Reset view"
        >
          Reset
        </button>
        <span className="h-8 px-2 bg-zinc-900 rounded flex items-center text-zinc-500 text-xs border border-zinc-700">
          {Math.round(transform.scale * 100)}%
        </span>
      </div>

      {/* SVG Canvas */}
      <svg
        ref={svgRef}
        width="100%"
        height="100%"
        className={`bg-zinc-950 ${isPanning ? "cursor-grabbing" : "cursor-grab"}`}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onWheel={handleWheel}
      >
        {/* Transformed content group */}
        <g transform={`translate(${transform.x}, ${transform.y}) scale(${transform.scale})`}>
          {/* Flow lines (rendered first, behind nodes) */}
          <FlowLines
            tokenPositions={tokenPositions}
            featurePositions={featurePositions}
            layerFlowData={layerFlowData}
            selectedTokenIndex={selection.tokenIndex}
            selectedFeatureId={selection.featureId}
            selectedLayer={selection.layer}
            hoveredToken={hoveredToken}
            hoveredFeature={hoveredFeature}
            maxActivation={globalMaxActivation}
          />

          {/* Input tokens header */}
          <g transform={`translate(${LAYOUT.padding}, ${LAYOUT.padding})`}>
            <rect
              x={0}
              y={0}
              width={LAYOUT.tokenColumnWidth}
              height={LAYOUT.headerHeight - 8}
              rx={6}
              fill="rgba(39, 39, 42, 0.5)"
              stroke="rgba(63, 63, 70, 0.8)"
            />
            <text
              x={LAYOUT.tokenColumnWidth / 2}
              y={(LAYOUT.headerHeight - 8) / 2}
              textAnchor="middle"
              dominantBaseline="middle"
              fill="#a1a1aa"
              className="text-sm font-medium"
            >
              Input
            </text>
          </g>

          {/* Token nodes */}
          {tokenPositions.map((pos) => (
            <TokenNode
              key={pos.tokenIndex}
              token={pos.token}
              index={pos.tokenIndex}
              x={pos.x}
              y={pos.y}
              isSelected={selection.tokenIndex === pos.tokenIndex}
              isHovered={hoveredToken === pos.tokenIndex}
              onSelect={() => selectToken(pos.tokenIndex)}
              onHover={(hovered) => setHoveredToken(hovered ? pos.tokenIndex : null)}
            />
          ))}

          {/* Layer headers */}
          {sortedLayers.map((layer, layerIndex) => {
            const x =
              LAYOUT.padding +
              LAYOUT.tokenColumnWidth +
              LAYOUT.layerGap +
              layerIndex * (LAYOUT.layerColumnWidth + LAYOUT.layerGap);
            const layerData = layerFlowData.find((l) => l.layer === layer);

            return (
              <LayerHeader
                key={layer}
                layer={layer}
                x={x}
                enabled={true}
                featureCount={layerData?.features.length ?? 0}
              />
            );
          })}

          {/* Feature nodes */}
          {Array.from(featurePositions.entries()).map(([key, pos]) => (
            <FeatureNode
              key={key}
              feature={pos.feature}
              x={pos.x}
              y={pos.y}
              maxActivation={globalMaxActivation}
              isSelected={
                selection.featureId === pos.feature.featureId &&
                selection.layer === pos.layer
              }
              isHovered={
                hoveredFeature?.id === pos.feature.featureId &&
                hoveredFeature?.layer === pos.layer
              }
              onSelect={() => selectFeature(pos.feature.featureId, pos.layer)}
              onHover={(hovered) =>
                setHoveredFeature(
                  hovered ? { id: pos.feature.featureId, layer: pos.layer } : null
                )
              }
            />
          ))}

          {/* Output panel */}
          <OutputPanel
            output={generatedOutput}
            isGenerating={isGenerating}
            isSelected={selection.showOutput}
            onSelect={selectOutput}
            x={outputX}
            height={svgDimensions.height - LAYOUT.padding * 2}
          />
        </g>
      </svg>

      {/* Pan hint */}
      <div className="absolute bottom-4 left-4 text-xs text-zinc-600">
        Drag to pan | Scroll to zoom
      </div>
    </div>
  );
}
