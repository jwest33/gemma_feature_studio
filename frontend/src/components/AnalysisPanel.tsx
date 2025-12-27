"use client";

import { useState, FormEvent } from "react";
import { useAnalysis } from "@/hooks/useAnalysis";
import { useSelectedToken } from "@/hooks/useSelectedToken";
import { TokenDisplay } from "./TokenDisplay";
import { FeatureHeatmap } from "./FeatureHeatmap";
import { FeatureDetail } from "./FeatureDetail";
import { NetworkPanel } from "./network/NetworkPanel";

type VisualizationView = "heatmap" | "network";

export function AnalysisPanel() {
  const [prompt, setPrompt] = useState("");
  const [topK, setTopK] = useState(50);
  const [vizView, setVizView] = useState<VisualizationView>("heatmap");
  const { data, isLoading, error, analyze, reset } = useAnalysis();
  const { selectedToken, selectedIndex, selectToken, clearSelection } =
    useSelectedToken();

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    clearSelection();
    await analyze({ prompt: prompt.trim(), top_k: topK });
  };

  const tokens = data?.layers[0]?.token_activations || [];

  return (
    <div className="space-y-6">
      {/* Input Form */}
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label
            htmlFor="prompt"
            className="block text-sm font-medium text-zinc-300 mb-2"
          >
            Enter prompt to analyze
          </label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="The meaning of life is..."
            className="w-full h-24 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-white placeholder-zinc-500 focus:outline-none focus:border-blue-500 resize-none"
          />
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <label htmlFor="topK" className="text-sm text-zinc-400">
              Top K Features:
            </label>
            <input
              id="topK"
              type="number"
              min={1}
              max={500}
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value) || 50)}
              className="w-20 px-2 py-1 bg-zinc-800 border border-zinc-700 rounded text-white focus:outline-none focus:border-blue-500"
            />
          </div>

          <div className="flex-1" />

          <button
            type="button"
            onClick={reset}
            disabled={!data}
            className="px-4 py-2 text-sm text-zinc-400 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Clear
          </button>

          <button
            type="submit"
            disabled={isLoading || !prompt.trim()}
            className="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-zinc-700 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors"
          >
            {isLoading ? "Analyzing..." : "Analyze"}
          </button>
        </div>
      </form>

      {/* Error Display */}
      {error && (
        <div className="p-4 bg-red-900/50 border border-red-700 rounded-lg text-red-200">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Results */}
      {data && (
        <div className="space-y-6">
          {/* Model Info */}
          <div className="flex items-center gap-4 text-sm text-zinc-500">
            <span>Model: {data.model_name}</span>
            <span>|</span>
            <span>SAE: {data.sae_id}</span>
            <span>|</span>
            <span>{data.tokens.length} tokens</span>
          </div>

          {/* Token Display */}
          <div>
            <h2 className="text-lg font-medium text-white mb-3">
              Token Activations
            </h2>
            <p className="text-sm text-zinc-500 mb-3">
              Click a token to view its top feature activations. Color intensity
              represents activation strength.
            </p>
            <TokenDisplay
              tokens={tokens}
              selectedIndex={selectedIndex}
              onTokenClick={selectToken}
            />
          </div>

          {/* Visualization Toggle */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-zinc-400">View:</span>
            <div className="inline-flex rounded-lg border border-zinc-700 overflow-hidden">
              <button
                onClick={() => setVizView("heatmap")}
                className={`px-4 py-1.5 text-sm transition-colors ${
                  vizView === "heatmap"
                    ? "bg-zinc-700 text-white"
                    : "text-zinc-400 hover:text-white hover:bg-zinc-800"
                }`}
              >
                Heatmap
              </button>
              <button
                onClick={() => setVizView("network")}
                className={`px-4 py-1.5 text-sm transition-colors border-l border-zinc-700 ${
                  vizView === "network"
                    ? "bg-zinc-700 text-white"
                    : "text-zinc-400 hover:text-white hover:bg-zinc-800"
                }`}
              >
                Network
              </button>
            </div>
          </div>

          {/* Heatmap View */}
          {vizView === "heatmap" && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Heatmap */}
              <div>
                <h2 className="text-lg font-medium text-white mb-3">
                  Activation Heatmap
                </h2>
                <FeatureHeatmap tokens={tokens} width={450} height={350} />
              </div>

              {/* Feature Detail */}
              <div>
                <h2 className="text-lg font-medium text-white mb-3">
                  Feature Details
                </h2>
                <FeatureDetail token={selectedToken} onClose={clearSelection} />
              </div>
            </div>
          )}

          {/* Network View */}
          {vizView === "network" && <NetworkPanel tokens={tokens} />}
        </div>
      )}

      {/* Empty State */}
      {!data && !isLoading && !error && (
        <div className="text-center py-12 text-zinc-500">
          Enter a prompt above to analyze feature activations
        </div>
      )}
    </div>
  );
}
