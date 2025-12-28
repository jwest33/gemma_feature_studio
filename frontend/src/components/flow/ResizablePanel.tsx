"use client";

import { useCallback, useRef, useEffect, useState } from "react";
import { useFlowStore, type PanelPosition } from "@/state/flowStore";

interface ResizablePanelProps {
  children: React.ReactNode;
}

export function ResizablePanel({ children }: ResizablePanelProps) {
  const { panelConfig, setPanelSize, togglePanelPosition } = useFlowStore();
  const panelRef = useRef<HTMLDivElement>(null);
  const [isResizing, setIsResizing] = useState(false);
  const [startPos, setStartPos] = useState(0);
  const [startSize, setStartSize] = useState(0);

  const isBottom = panelConfig.position === "bottom";

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      setIsResizing(true);
      setStartPos(isBottom ? e.clientY : e.clientX);
      setStartSize(panelConfig.size);
    },
    [isBottom, panelConfig.size]
  );

  useEffect(() => {
    if (!isResizing) return;

    const handleMouseMove = (e: MouseEvent) => {
      const currentPos = isBottom ? e.clientY : e.clientX;
      const delta = startPos - currentPos;
      const newSize = startSize + delta;
      setPanelSize(newSize);
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizing, isBottom, startPos, startSize, setPanelSize]);

  const positionStyles = isBottom
    ? { height: panelConfig.size }
    : { width: panelConfig.size };

  const containerClasses = isBottom
    ? "border-t border-zinc-800 shrink-0 bg-zinc-900/50"
    : "border-l border-zinc-800 shrink-0 bg-zinc-900/50";

  const resizeHandleClasses = isBottom
    ? "absolute top-0 left-0 right-0 h-2 cursor-ns-resize group"
    : "absolute top-0 left-0 bottom-0 w-2 cursor-ew-resize group";

  const resizeBarClasses = isBottom
    ? "absolute top-0 left-0 right-0 h-0.5 bg-zinc-700 group-hover:bg-blue-500 transition-colors"
    : "absolute top-0 left-0 bottom-0 w-0.5 bg-zinc-700 group-hover:bg-blue-500 transition-colors";

  return (
    <div
      ref={panelRef}
      className={`relative ${containerClasses}`}
      style={positionStyles}
    >
      {/* Resize Handle */}
      <div
        className={resizeHandleClasses}
        onMouseDown={handleMouseDown}
      >
        <div className={`${resizeBarClasses} ${isResizing ? "bg-blue-500" : ""}`} />
      </div>

      {/* Position Toggle Button */}
      <button
        onClick={togglePanelPosition}
        className="absolute top-2 right-2 z-10 p-1.5 rounded bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 text-zinc-400 hover:text-zinc-200 transition-colors"
        title={isBottom ? "Move panel to right side" : "Move panel to bottom"}
      >
        {isBottom ? (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
          </svg>
        ) : (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
          </svg>
        )}
      </button>

      {/* Panel Content */}
      <div className={isBottom ? "h-full" : "h-full"}>
        {children}
      </div>

      {/* Resize overlay during drag */}
      {isResizing && (
        <div className="fixed inset-0 z-50 cursor-ns-resize" style={{ cursor: isBottom ? "ns-resize" : "ew-resize" }} />
      )}
    </div>
  );
}
