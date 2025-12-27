"use client";

import { useMemo } from "react";
import type { TokenActivations } from "@/types/analysis";
import { getActivationColor, getContrastColor } from "@/lib/colorScale";

interface TokenDisplayProps {
  tokens: TokenActivations[];
  selectedIndex: number | null;
  onTokenClick: (token: TokenActivations, index: number) => void;
  maxActivation?: number;
}

export function TokenDisplay({
  tokens,
  selectedIndex,
  onTokenClick,
  maxActivation: providedMax,
}: TokenDisplayProps) {
  const maxActivation = useMemo(() => {
    if (providedMax !== undefined) return providedMax;
    let max = 0;
    for (const token of tokens) {
      for (const feat of token.top_features) {
        if (feat.activation > max) max = feat.activation;
      }
    }
    return max || 1;
  }, [tokens, providedMax]);

  return (
    <div className="flex flex-wrap gap-1 p-4 bg-zinc-900 rounded-lg">
      {tokens.map((token, idx) => {
        const topActivation = token.top_features[0]?.activation || 0;
        const bgColor = getActivationColor(topActivation, maxActivation);
        const textColor = getContrastColor(bgColor);
        const isSelected = selectedIndex === idx;

        return (
          <button
            key={`${token.position}-${idx}`}
            onClick={() => onTokenClick(token, idx)}
            className={`
              px-2 py-1 rounded font-mono text-sm transition-all
              hover:ring-2 hover:ring-white/50
              ${isSelected ? "ring-2 ring-blue-500 scale-105" : ""}
            `}
            style={{
              backgroundColor: bgColor,
              color: textColor,
            }}
            title={`Token: "${token.token}"\nTop feature: ${token.top_features[0]?.id || "none"}\nActivation: ${topActivation.toFixed(4)}`}
          >
            {token.token.replace(/ /g, "\u00B7")}
          </button>
        );
      })}
    </div>
  );
}
