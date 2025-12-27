"use client";

import type { TokenActivations } from "@/types/analysis";
import { VirtualizedFeatureList } from "./VirtualizedFeatureList";

interface FeatureDetailProps {
  token: TokenActivations | null;
  onClose: () => void;
}

export function FeatureDetail({ token, onClose }: FeatureDetailProps) {
  if (!token) {
    return (
      <div className="bg-zinc-900 rounded-lg p-6 flex items-center justify-center text-zinc-500">
        Click a token to see its feature activations
      </div>
    );
  }

  const maxActivation =
    token.top_features.length > 0
      ? Math.max(...token.top_features.map((f) => f.activation))
      : 1;

  return (
    <div className="bg-zinc-900 rounded-lg overflow-hidden">
      <div className="flex items-center justify-between p-3 border-b border-zinc-800">
        <div>
          <h3 className="font-medium text-white">
            Token:{" "}
            <span className="font-mono bg-zinc-800 px-2 py-0.5 rounded">
              {token.token.replace(/ /g, "\u00B7")}
            </span>
          </h3>
          <p className="text-xs text-zinc-500 mt-1">
            Position: {token.position} | {token.top_features.length} active
            features
          </p>
        </div>
        <button
          onClick={onClose}
          className="text-zinc-400 hover:text-white transition-colors"
        >
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      </div>

      <VirtualizedFeatureList
        features={token.top_features}
        maxActivation={maxActivation}
        height={300}
      />
    </div>
  );
}
