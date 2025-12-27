"use client";

import { useMemo, useCallback, useState } from "react";
import ReactFlow, {
  Node,
  Edge,
  Controls,
  MiniMap,
  Background,
  BackgroundVariant,
  useNodesState,
  useEdgesState,
  NodeTypes,
  MarkerType,
} from "reactflow";
import "reactflow/dist/style.css";
import { FeatureNode, FeatureNodeData } from "./FeatureNode";
import type { TokenActivations } from "@/types/analysis";

interface NetworkGraphProps {
  tokens: TokenActivations[];
  topK?: number;
  edgeThreshold?: number;
  onFeatureSelect?: (featureId: number) => void;
  selectedFeatures?: number[];
}

const nodeTypes: NodeTypes = {
  feature: FeatureNode,
};

// Calculate edge weights based on feature co-activation
function calculateEdges(
  tokens: TokenActivations[],
  topK: number,
  threshold: number
): Edge[] {
  const edges: Edge[] = [];
  const edgeWeights = new Map<string, number>();

  // Look for co-occurring features across adjacent tokens
  for (let i = 0; i < tokens.length - 1; i++) {
    const currentFeatures = tokens[i].top_features.slice(0, topK);
    const nextFeatures = tokens[i + 1].top_features.slice(0, topK);

    for (const curr of currentFeatures) {
      for (const next of nextFeatures) {
        // Create edge if both features have significant activation
        const weight = Math.min(curr.activation, next.activation);
        if (weight >= threshold) {
          const edgeId = `e-${i}-${curr.id}-${i + 1}-${next.id}`;
          edgeWeights.set(edgeId, weight);
        }
      }
    }
  }

  // Convert to edges with normalized weights
  const maxWeight = Math.max(...Array.from(edgeWeights.values()), 1);

  edgeWeights.forEach((weight, id) => {
    const parts = id.split("-");
    const sourceToken = parseInt(parts[1]);
    const sourceFeature = parseInt(parts[2]);
    const targetToken = parseInt(parts[3]);
    const targetFeature = parseInt(parts[4]);

    const normalizedWeight = weight / maxWeight;

    edges.push({
      id,
      source: `node-${sourceToken}-${sourceFeature}`,
      target: `node-${targetToken}-${targetFeature}`,
      animated: normalizedWeight > 0.7,
      style: {
        stroke: `rgba(99, 102, 241, ${0.2 + normalizedWeight * 0.6})`,
        strokeWidth: 1 + normalizedWeight * 2,
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        width: 15,
        height: 15,
        color: `rgba(99, 102, 241, ${0.3 + normalizedWeight * 0.5})`,
      },
    });
  });

  return edges;
}

export function NetworkGraph({
  tokens,
  topK = 5,
  edgeThreshold = 0.5,
  onFeatureSelect,
  selectedFeatures = [],
}: NetworkGraphProps) {
  const [showEdges, setShowEdges] = useState(true);
  const [localTopK, setLocalTopK] = useState(topK);
  const [localThreshold, setLocalThreshold] = useState(edgeThreshold);

  // Calculate max activation across all features
  const maxActivation = useMemo(() => {
    let max = 0;
    for (const token of tokens) {
      for (const feature of token.top_features) {
        max = Math.max(max, feature.activation);
      }
    }
    return max;
  }, [tokens]);

  // Create nodes from token activations
  const initialNodes = useMemo((): Node<FeatureNodeData>[] => {
    const nodes: Node<FeatureNodeData>[] = [];
    const nodeSpacingX = 150;
    const nodeSpacingY = 80;

    tokens.forEach((tokenData, tokenIndex) => {
      const topFeatures = tokenData.top_features.slice(0, localTopK);

      topFeatures.forEach((feature, featureIndex) => {
        nodes.push({
          id: `node-${tokenIndex}-${feature.id}`,
          type: "feature",
          position: {
            x: tokenIndex * nodeSpacingX,
            y: featureIndex * nodeSpacingY,
          },
          data: {
            featureId: feature.id,
            activation: feature.activation,
            maxActivation,
            tokenIndex,
            token: tokenData.token,
            isSelected: selectedFeatures.includes(feature.id),
            onSelect: onFeatureSelect,
          },
        });
      });
    });

    return nodes;
  }, [tokens, localTopK, maxActivation, selectedFeatures, onFeatureSelect]);

  // Create edges based on co-activation
  const initialEdges = useMemo(
    () => (showEdges ? calculateEdges(tokens, localTopK, localThreshold) : []),
    [tokens, localTopK, localThreshold, showEdges]
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Update nodes when props change
  useMemo(() => {
    setNodes(initialNodes);
    setEdges(initialEdges);
  }, [initialNodes, initialEdges, setNodes, setEdges]);

  const handleNodeClick = useCallback(
    (_: React.MouseEvent, node: Node<FeatureNodeData>) => {
      onFeatureSelect?.(node.data.featureId);
    },
    [onFeatureSelect]
  );

  if (tokens.length === 0) {
    return (
      <div className="h-[500px] bg-zinc-900 rounded-lg flex items-center justify-center text-zinc-500">
        Analyze a prompt to visualize the feature network
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Controls */}
      <div className="flex items-center gap-4 text-sm">
        <div className="flex items-center gap-2">
          <label className="text-zinc-400">Top K:</label>
          <input
            type="number"
            min={1}
            max={20}
            value={localTopK}
            onChange={(e) => setLocalTopK(parseInt(e.target.value) || 5)}
            className="w-16 px-2 py-1 bg-zinc-800 border border-zinc-700 rounded text-white"
          />
        </div>
        <div className="flex items-center gap-2">
          <label className="text-zinc-400">Edge Threshold:</label>
          <input
            type="number"
            min={0}
            max={5}
            step={0.1}
            value={localThreshold}
            onChange={(e) => setLocalThreshold(parseFloat(e.target.value) || 0.5)}
            className="w-20 px-2 py-1 bg-zinc-800 border border-zinc-700 rounded text-white"
          />
        </div>
        <label className="flex items-center gap-2 text-zinc-400 cursor-pointer">
          <input
            type="checkbox"
            checked={showEdges}
            onChange={(e) => setShowEdges(e.target.checked)}
            className="rounded bg-zinc-800 border-zinc-700"
          />
          Show Edges
        </label>
      </div>

      {/* Graph */}
      <div className="h-[500px] bg-zinc-900 rounded-lg border border-zinc-800 overflow-hidden">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={handleNodeClick}
          nodeTypes={nodeTypes}
          fitView
          minZoom={0.2}
          maxZoom={2}
          defaultViewport={{ x: 0, y: 0, zoom: 0.8 }}
          proOptions={{ hideAttribution: true }}
        >
          <Background
            variant={BackgroundVariant.Dots}
            gap={16}
            size={1}
            color="#3f3f46"
          />
          <Controls
            className="!bg-zinc-800 !border-zinc-700 !rounded-lg !shadow-lg [&>button]:!bg-zinc-800 [&>button]:!border-zinc-700 [&>button]:!text-zinc-300 [&>button:hover]:!bg-zinc-700"
          />
          <MiniMap
            nodeStrokeWidth={3}
            className="!bg-zinc-800 !border-zinc-700 !rounded-lg"
            maskColor="rgba(0, 0, 0, 0.7)"
          />
        </ReactFlow>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-6 text-xs text-zinc-500">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded" style={{ background: "linear-gradient(to right, #440154, #21918c, #fde725)" }} />
          <span>Activation strength (low to high)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-8 h-0.5 bg-indigo-500 animate-pulse" />
          <span>Strong co-activation</span>
        </div>
        <span>Click node to add to steering</span>
      </div>
    </div>
  );
}
