"use client";

import { useMemo, useState } from "react";
import { useFlowStore } from "@/state/flowStore";
import type { FeatureActivation, TokenActivations } from "@/types/analysis";

// Full class names for Tailwind JIT - dynamic class construction doesn't work
const LAYER_TEXT_CLASSES: Record<number, string> = {
  9: "text-purple-400",
  17: "text-blue-400",
  22: "text-cyan-400",
  29: "text-green-400",
};

function getLayerTextClass(layer: number): string {
  return LAYER_TEXT_CLASSES[layer] || "text-zinc-400";
}

interface FeatureInspectorProps {
  onAddToSteering?: (featureId: number) => void;
  generatedOutput?: string;
}

export function FeatureInspector({ onAddToSteering, generatedOutput }: FeatureInspectorProps) {
  const { selection, selectedLayers, clearSelection } = useFlowStore();
  const activePrompt = useFlowStore((state) => state.getActivePrompt());
  const [copied, setCopied] = useState(false);

  // Get selected token data
  const selectedTokenData = useMemo(() => {
    if (selection.tokenIndex === null || !activePrompt) return null;

    const tokenData: {
      token: string;
      position: number;
      layerFeatures: Map<number, FeatureActivation[]>;
    } = {
      token: activePrompt.tokens[selection.tokenIndex] ?? "",
      position: selection.tokenIndex,
      layerFeatures: new Map(),
    };

    for (const layer of selectedLayers) {
      const layerData = activePrompt.layerData.get(layer);
      if (layerData && layerData.token_activations[selection.tokenIndex]) {
        tokenData.layerFeatures.set(
          layer,
          layerData.token_activations[selection.tokenIndex].top_features
        );
      }
    }

    return tokenData;
  }, [selection.tokenIndex, activePrompt, selectedLayers]);

  // Get selected feature data across layers
  const selectedFeatureData = useMemo(() => {
    if (selection.featureId === null || !activePrompt) return null;

    const featureData: {
      featureId: number;
      occurrences: Array<{
        layer: number;
        tokenIndex: number;
        token: string;
        activation: number;
      }>;
    } = {
      featureId: selection.featureId,
      occurrences: [],
    };

    for (const layer of selectedLayers) {
      const layerData = activePrompt.layerData.get(layer);
      if (!layerData) continue;

      layerData.token_activations.forEach((tokenAct, tokenIdx) => {
        const feature = tokenAct.top_features.find((f) => f.id === selection.featureId);
        if (feature) {
          featureData.occurrences.push({
            layer,
            tokenIndex: tokenIdx,
            token: activePrompt.tokens[tokenIdx] ?? "",
            activation: feature.activation,
          });
        }
      });
    }

    // Sort by activation
    featureData.occurrences.sort((a, b) => b.activation - a.activation);

    return featureData;
  }, [selection.featureId, activePrompt, selectedLayers]);

  if (!activePrompt) {
    return (
      <div className="h-full flex items-center justify-center text-zinc-500 text-sm">
        Analyze a prompt to see feature details
      </div>
    );
  }

  if (selection.tokenIndex === null && selection.featureId === null && !selection.showOutput) {
    return (
      <div className="h-full flex items-center justify-center text-zinc-500 text-sm">
        <div className="text-center">
          <p>Click on a token, feature, or output to inspect</p>
          <p className="text-xs mt-1 text-zinc-600">
            Tokens show feature activations • Features show where they appear • Output can be copied
          </p>
        </div>
      </div>
    );
  }

  // Handle copy to clipboard
  const handleCopy = async () => {
    if (generatedOutput) {
      await navigator.clipboard.writeText(generatedOutput);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="h-full flex flex-col p-4 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 shrink-0">
        <h3 className="text-sm font-medium text-white">
          {selection.showOutput
            ? "Generated Output"
            : selection.featureId !== null
            ? `Feature #${selection.featureId}`
            : selectedTokenData
            ? `Token: "${selectedTokenData.token.replace(/ /g, "·")}"`
            : "Inspector"}
        </h3>
        <div className="flex items-center gap-2">
          {selection.showOutput && generatedOutput && (
            <button
              onClick={handleCopy}
              className={`px-3 py-1 text-xs rounded transition-colors ${
                copied
                  ? "bg-green-600 text-white"
                  : "bg-green-600 text-white hover:bg-green-500"
              }`}
            >
              {copied ? "Copied!" : "Copy to Clipboard"}
            </button>
          )}
          {selection.featureId !== null && onAddToSteering && (
            <button
              onClick={() => onAddToSteering(selection.featureId!)}
              className="px-2 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-500"
            >
              Add to Steering
            </button>
          )}
          <button
            onClick={clearSelection}
            className="p-1 text-zinc-500 hover:text-zinc-300"
            title="Clear selection"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {/* Output Display */}
        {selection.showOutput && (
          <div className="space-y-4">
            {generatedOutput ? (
              <div className="bg-zinc-800 rounded-lg p-4 border border-green-500/30">
                <pre className="text-sm text-zinc-200 whitespace-pre-wrap font-mono select-all">
                  {generatedOutput}
                </pre>
              </div>
            ) : (
              <div className="text-zinc-500 text-sm italic">
                No output generated yet. Run analysis on a single prompt to generate output.
              </div>
            )}
          </div>
        )}

        {/* Token Inspector */}
        {selection.tokenIndex !== null && selectedTokenData && (
          <div className="space-y-4">
            <div className="text-xs text-zinc-400">
              Position: {selectedTokenData.position}
            </div>

            {/* Features by Layer */}
            <div className="space-y-3">
              {Array.from(selectedTokenData.layerFeatures.entries()).map(
                ([layer, features]) => (
                  <div key={layer}>
                    <div
                      className={`text-xs font-medium mb-2 ${getLayerTextClass(layer)}`}
                    >
                      Layer {layer}
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
                      {features.slice(0, 12).map((feat) => (
                        <div
                          key={feat.id}
                          className={`
                            p-2 rounded bg-zinc-800 cursor-pointer
                            hover:bg-zinc-700 transition-colors
                            ${selection.featureId === feat.id ? "ring-1 ring-blue-500" : ""}
                          `}
                          onClick={() => useFlowStore.getState().selectFeature(feat.id, layer)}
                        >
                          <div className="text-xs font-mono text-zinc-300">#{feat.id}</div>
                          <div className="text-xs text-zinc-500">
                            {feat.activation.toFixed(3)}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )
              )}
            </div>
          </div>
        )}

        {/* Feature Inspector */}
        {selection.featureId !== null && selectedFeatureData && (
          <div className="space-y-4">
            <div className="text-xs text-zinc-400">
              Found in {selectedFeatureData.occurrences.length} position(s)
            </div>

            {/* Occurrences Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-zinc-400 text-xs">
                    <th className="pb-2 pr-4">Layer</th>
                    <th className="pb-2 pr-4">Token</th>
                    <th className="pb-2 pr-4">Position</th>
                    <th className="pb-2">Activation</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedFeatureData.occurrences.map((occ, idx) => (
                    <tr
                      key={idx}
                      className="border-t border-zinc-800 hover:bg-zinc-800/50 cursor-pointer"
                      onClick={() => {
                        useFlowStore.getState().selectToken(occ.tokenIndex);
                      }}
                    >
                      <td className={`py-2 pr-4 ${getLayerTextClass(occ.layer)}`}>
                        {occ.layer}
                      </td>
                      <td className="py-2 pr-4 font-mono text-zinc-300">
                        {occ.token.replace(/ /g, "·")}
                      </td>
                      <td className="py-2 pr-4 text-zinc-500">{occ.tokenIndex}</td>
                      <td className="py-2 font-mono text-zinc-300">
                        {occ.activation.toFixed(4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
