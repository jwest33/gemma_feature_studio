"use client";

import { memo, useCallback } from "react";
import { Handle, Position, NodeProps } from "reactflow";
import { interpolateViridis } from "d3-scale-chromatic";

export interface FeatureNodeData {
  featureId: number;
  activation: number;
  maxActivation: number;
  tokenIndex: number;
  token: string;
  isSelected?: boolean;
  onSelect?: (featureId: number) => void;
}

function FeatureNodeComponent({ data, selected }: NodeProps<FeatureNodeData>) {
  const {
    featureId,
    activation,
    maxActivation,
    token,
    isSelected,
    onSelect,
  } = data;

  // Normalize activation for color
  const normalizedActivation = maxActivation > 0 ? activation / maxActivation : 0;
  const bgColor = interpolateViridis(normalizedActivation);

  // Determine text color based on background brightness
  const textColor = normalizedActivation > 0.5 ? "#000" : "#fff";

  const handleClick = useCallback(() => {
    onSelect?.(featureId);
  }, [featureId, onSelect]);

  return (
    <div
      onClick={handleClick}
      className={`
        px-3 py-2 rounded-lg shadow-md cursor-pointer
        transition-all duration-150
        ${selected || isSelected ? "ring-2 ring-blue-400 ring-offset-2 ring-offset-zinc-950" : ""}
        hover:scale-105 hover:shadow-lg
      `}
      style={{
        backgroundColor: bgColor,
        minWidth: "80px",
      }}
    >
      {/* Input handle (left) */}
      <Handle
        type="target"
        position={Position.Left}
        className="!w-2 !h-2 !bg-zinc-400 !border-none"
      />

      {/* Content */}
      <div className="text-center" style={{ color: textColor }}>
        <div className="text-xs font-medium opacity-75 truncate max-w-[60px]">
          {token.replace(/ /g, "\u00B7")}
        </div>
        <div className="text-sm font-bold">#{featureId}</div>
        <div className="text-xs opacity-75">{activation.toFixed(2)}</div>
      </div>

      {/* Output handle (right) */}
      <Handle
        type="source"
        position={Position.Right}
        className="!w-2 !h-2 !bg-zinc-400 !border-none"
      />
    </div>
  );
}

export const FeatureNode = memo(FeatureNodeComponent);
