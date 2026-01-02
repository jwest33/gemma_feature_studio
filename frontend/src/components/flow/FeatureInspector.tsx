"use client";

import { useMemo, useState, useCallback, useEffect } from "react";
import { useFlowStore } from "@/state/flowStore";
import type { FeatureActivation, NeuronpediaFeatureResponse, NeuronpediaActivation } from "@/types/analysis";
import { getNeuronpediaFeature } from "@/lib/api";

// =============================================================================
// Neuronpedia Export Utilities
// =============================================================================

interface ParsedMessage {
  role: "user" | "model" | "system";
  content: string;
  tokens: Array<{
    token: string;
    activation: number;
  }>;
  maxActivation: number;
}

interface ParsedActivation {
  messages: ParsedMessage[];
  fullText: string;
  maxValue: number;
  maxTokenIndex: number;
}

/**
 * Parse a Neuronpedia activation into user/model message segments.
 * Handles Gemma's <start_of_turn>user/<start_of_turn>model format.
 */
function parseActivationMessages(activation: NeuronpediaActivation): ParsedActivation {
  const { tokens, values, maxValue, maxTokenIndex } = activation;

  const messages: ParsedMessage[] = [];
  let currentRole: "user" | "model" | "system" = "system";
  let currentTokens: Array<{ token: string; activation: number }> = [];
  let fullText = "";

  // Special tokens that indicate role changes
  const specialTokens = new Set(['<bos>', '<eos>', '<pad>', '<start_of_turn>', '<end_of_turn>', '<sep>', '<cls>', '<mask>']);

  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    const value = values[i] || 0;

    // Check for role change tokens
    if (token === '<start_of_turn>') {
      // Save current segment if it has content
      if (currentTokens.length > 0) {
        const content = currentTokens.map(t => {
          let display = t.token;
          if (display.startsWith('▁') || display.startsWith('Ġ') || display.startsWith('·')) {
            display = ' ' + display.slice(1);
          }
          return display;
        }).join('').trim();

        if (content) {
          messages.push({
            role: currentRole,
            content,
            tokens: [...currentTokens],
            maxActivation: Math.max(...currentTokens.map(t => t.activation), 0),
          });
        }
        currentTokens = [];
      }
      continue;
    }

    // Check for role indicators
    if (token === 'user' && tokens[i - 1] === '<start_of_turn>') {
      currentRole = 'user';
      continue;
    }
    if (token === 'model' && tokens[i - 1] === '<start_of_turn>') {
      currentRole = 'model';
      continue;
    }

    // Skip other special tokens
    if (specialTokens.has(token)) {
      continue;
    }

    // Add token to current segment
    currentTokens.push({ token, activation: value });

    // Build full text
    let displayToken = token;
    if (displayToken.startsWith('▁') || displayToken.startsWith('Ġ') || displayToken.startsWith('·') || displayToken.startsWith(' ')) {
      displayToken = ' ' + displayToken.slice(1);
    }
    fullText += displayToken;
  }

  // Save final segment
  if (currentTokens.length > 0) {
    const content = currentTokens.map(t => {
      let display = t.token;
      if (display.startsWith('▁') || display.startsWith('Ġ') || display.startsWith('·')) {
        display = ' ' + display.slice(1);
      }
      return display;
    }).join('').trim();

    if (content) {
      messages.push({
        role: currentRole,
        content,
        tokens: [...currentTokens],
        maxActivation: Math.max(...currentTokens.map(t => t.activation), 0),
      });
    }
  }

  return {
    messages,
    fullText: fullText.trim(),
    maxValue,
    maxTokenIndex,
  };
}

interface ExportData {
  feature: {
    id: number;
    layer: string;
    modelId: string;
    description: string | null;
    neuronpediaUrl: string;
  };
  activations: Array<{
    index: number;
    maxActivation: number;
    fullText: string;
    messages: Array<{
      role: string;
      content: string;
      maxActivation: number;
      tokens: Array<{
        token: string;
        activation: number;
      }>;
    }>;
  }>;
  exportedAt: string;
}

function exportNeuronpediaData(data: NeuronpediaFeatureResponse): void {
  const exportData: ExportData = {
    feature: {
      id: data.index,
      layer: data.layer,
      modelId: data.modelId,
      description: data.description,
      neuronpediaUrl: data.neuronpedia_url,
    },
    activations: data.activations.map((act, idx) => {
      const parsed = parseActivationMessages(act);
      return {
        index: idx,
        maxActivation: parsed.maxValue,
        fullText: parsed.fullText,
        messages: parsed.messages.map(msg => ({
          role: msg.role,
          content: msg.content,
          maxActivation: msg.maxActivation,
          tokens: msg.tokens,
        })),
      };
    }),
    exportedAt: new Date().toISOString(),
  };

  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `neuronpedia-feature-${data.index}-layer-${data.layer}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function exportUserMessagesOnly(data: NeuronpediaFeatureResponse): void {
  const userMessages: string[] = [];

  for (const act of data.activations) {
    const parsed = parseActivationMessages(act);
    for (const msg of parsed.messages) {
      if (msg.role === 'user' && msg.content.trim()) {
        // Replace newlines with spaces so each message stays on one line
        const singleLineContent = msg.content.trim().replace(/[\r\n]+/g, ' ');
        userMessages.push(singleLineContent);
      }
    }
  }

  const content = userMessages.join('\n');
  const blob = new Blob([content], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `neuronpedia-feature-${data.index}-user-messages.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

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
  generatedOutput?: string;
  baselineOutput?: string;
}

export function FeatureInspector({
  generatedOutput,
  baselineOutput,
}: FeatureInspectorProps) {
  const { selection, selectedLayers, clearSelection, addSteeringFeature, hasSteeringFeature } = useFlowStore();
  const activePrompt = useFlowStore((state) => state.getActivePrompt());
  const [copied, setCopied] = useState(false);

  // Neuronpedia state
  const [neuronpediaData, setNeuronpediaData] = useState<NeuronpediaFeatureResponse | null>(null);
  const [neuronpediaLoading, setNeuronpediaLoading] = useState(false);
  const [neuronpediaError, setNeuronpediaError] = useState<string | null>(null);

  // Fetch Neuronpedia data when a feature is selected
  useEffect(() => {
    if (selection.featureId === null || selection.layer === null) {
      setNeuronpediaData(null);
      setNeuronpediaError(null);
      return;
    }

    const fetchNeuronpediaData = async () => {
      setNeuronpediaLoading(true);
      setNeuronpediaError(null);
      try {
        const data = await getNeuronpediaFeature({
          feature_id: selection.featureId!,
          layer: selection.layer!,
        });
        setNeuronpediaData(data);
      } catch (err) {
        setNeuronpediaError(err instanceof Error ? err.message : "Failed to fetch from Neuronpedia");
        setNeuronpediaData(null);
      } finally {
        setNeuronpediaLoading(false);
      }
    };

    fetchNeuronpediaData();
  }, [selection.featureId, selection.layer]);

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

  // Check if feature is already in steering panel
  const isFeatureInSteering = useMemo(() => {
    if (selection.featureId === null || selection.layer === null) return false;
    return hasSteeringFeature(selection.featureId, selection.layer);
  }, [selection.featureId, selection.layer, hasSteeringFeature]);

  // Handle adding feature to steering panel
  const handleAddToSteering = useCallback(() => {
    if (selection.featureId === null || selection.layer === null) return;

    // Use Neuronpedia description as the feature name if available
    const name = neuronpediaData?.description
      ? neuronpediaData.description.slice(0, 50) + (neuronpediaData.description.length > 50 ? "..." : "")
      : undefined;

    addSteeringFeature(selection.featureId, selection.layer, name);
  }, [selection.featureId, selection.layer, neuronpediaData, addSteeringFeature]);

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
            Tokens show feature activations • Features show where they appear • Add features to steering panel
          </p>
        </div>
      </div>
    );
  }

  // Handle copy to clipboard
  const handleCopy = async (text: string) => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
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
            {(generatedOutput || baselineOutput) ? (
              <div className="space-y-4">
                {/* Comparison View */}
                {baselineOutput && generatedOutput && (
                  <div className="grid grid-cols-2 gap-4">
                    {/* Baseline */}
                    <div className="bg-zinc-800 rounded-lg p-3 border border-zinc-700">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs text-zinc-400 font-medium">Baseline (no steering)</span>
                        <button
                          onClick={() => handleCopy(baselineOutput)}
                          className="text-xs text-zinc-500 hover:text-zinc-300"
                        >
                          Copy
                        </button>
                      </div>
                      <pre className="text-sm text-zinc-300 whitespace-pre-wrap font-mono select-all pb-4">
                        {baselineOutput}
                      </pre>
                    </div>
                    {/* Steered */}
                    <div className="bg-zinc-800 rounded-lg p-3 border border-green-500/30">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs text-green-400 font-medium">Steered Output</span>
                        <button
                          onClick={() => handleCopy(generatedOutput)}
                          className="text-xs text-zinc-500 hover:text-zinc-300"
                        >
                          {copied ? "Copied!" : "Copy"}
                        </button>
                      </div>
                      <pre className="text-sm text-zinc-200 whitespace-pre-wrap font-mono select-all pb-4">
                        {generatedOutput}
                      </pre>
                    </div>
                  </div>
                )}
                {/* Single output (no comparison) */}
                {!baselineOutput && generatedOutput && (
                  <div className="bg-zinc-800 rounded-lg p-4 border border-green-500/30">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-zinc-400">Generated Output</span>
                      <button
                        onClick={() => handleCopy(generatedOutput)}
                        className={`px-3 py-1 text-xs rounded transition-colors ${
                          copied
                            ? "bg-green-600 text-white"
                            : "bg-green-600 text-white hover:bg-green-500"
                        }`}
                      >
                        {copied ? "Copied!" : "Copy to Clipboard"}
                      </button>
                    </div>
                    <pre className="text-sm text-zinc-200 whitespace-pre-wrap font-mono select-all pb-4">
                      {generatedOutput}
                    </pre>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-zinc-500 text-sm italic">
                No output generated yet. Add features to the steering panel above and click Generate.
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
          <div className="flex-1 flex flex-col min-h-0 space-y-4">
            {/* Add to Steering Button */}
            <div className="shrink-0">
              <button
                onClick={handleAddToSteering}
                disabled={isFeatureInSteering}
                className={`w-full py-2.5 rounded-lg font-medium transition-colors flex items-center justify-center gap-2 ${
                  isFeatureInSteering
                    ? "bg-zinc-700 text-zinc-400 cursor-not-allowed"
                    : "bg-blue-600 text-white hover:bg-blue-500"
                }`}
              >
                {isFeatureInSteering ? (
                  <>
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    Added to Steering Panel
                  </>
                ) : (
                  <>
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                    </svg>
                    Add to Steering Panel
                  </>
                )}
              </button>
            </div>

            {/* Neuronpedia Feature Info */}
            <div className="bg-gradient-to-b from-zinc-800/80 to-zinc-800/40 rounded-lg p-4 border border-zinc-600 flex-1 min-h-0 flex flex-col overflow-hidden">
              <div className="flex items-center justify-between mb-4 shrink-0">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-emerald-500"></div>
                  <h4 className="text-sm font-semibold text-white">Neuronpedia</h4>
                </div>
                <div className="flex items-center gap-2">
                  {neuronpediaData?.hasData && neuronpediaData.activations.length > 0 && (
                    <>
                      <button
                        onClick={() => exportNeuronpediaData(neuronpediaData)}
                        className="text-xs text-emerald-400 hover:text-emerald-300 flex items-center gap-1.5 px-2 py-1 rounded bg-emerald-500/10 hover:bg-emerald-500/20 transition-colors"
                        title="Export all activations as JSON"
                      >
                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                        </svg>
                        JSON
                      </button>
                      <button
                        onClick={() => exportUserMessagesOnly(neuronpediaData)}
                        className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1.5 px-2 py-1 rounded bg-blue-500/10 hover:bg-blue-500/20 transition-colors"
                        title="Export user messages only as text file"
                      >
                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                        </svg>
                        User TXT
                      </button>
                    </>
                  )}
                  {neuronpediaData?.neuronpedia_url && (
                    <a
                      href={neuronpediaData.neuronpedia_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1.5 px-2 py-1 rounded bg-blue-500/10 hover:bg-blue-500/20 transition-colors"
                    >
                      View on Neuronpedia
                      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                      </svg>
                    </a>
                  )}
                </div>
              </div>

              {neuronpediaLoading && (
                <div className="flex items-center justify-center gap-3 text-zinc-300 text-sm py-8">
                  <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Loading from Neuronpedia...
                </div>
              )}

              {neuronpediaError && (
                <div className="text-red-400 text-sm bg-red-500/10 rounded-lg p-3 border border-red-500/20">
                  {neuronpediaError}
                </div>
              )}

              {neuronpediaData && !neuronpediaLoading && (
                <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
                  {/* Feature Description */}
                  <div className="shrink-0 mb-4">
                    {neuronpediaData.hasData && neuronpediaData.description ? (
                      <div className="bg-zinc-900/50 rounded-lg p-3 border border-zinc-700">
                        <div className="text-xs font-medium text-emerald-400 mb-2 uppercase tracking-wide">Description</div>
                        <p className="text-sm text-white leading-relaxed">{neuronpediaData.description}</p>
                      </div>
                    ) : neuronpediaData.hasData === false ? (
                      <div className="text-zinc-400 text-sm italic bg-zinc-900/30 rounded-lg p-3 text-center">
                        No data available for this feature on Neuronpedia.
                      </div>
                    ) : (
                      <div className="text-zinc-400 text-sm italic bg-zinc-900/30 rounded-lg p-3 text-center">
                        No description available.
                      </div>
                    )}
                  </div>

                  {/* Example Activations */}
                  {neuronpediaData.activations && neuronpediaData.activations.length > 0 && (
                    <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
                      <div className="text-xs font-medium text-cyan-400 mb-3 shrink-0 uppercase tracking-wide">
                        Example Activations ({neuronpediaData.activations.length})
                      </div>
                      <div className="flex-1 min-h-0 overflow-y-auto space-y-3 pr-1">
                        {neuronpediaData.activations.map((act, idx) => {
                          const parsed = parseActivationMessages(act);

                          return (
                            <div key={idx} className="bg-zinc-900/70 rounded-lg border border-zinc-700/50 overflow-hidden">
                              {/* Messages breakdown */}
                              <div className="divide-y divide-zinc-800">
                                {parsed.messages.map((msg, msgIdx) => {
                                  const roleColors = {
                                    user: { bg: 'bg-blue-500/10', border: 'border-l-blue-500', label: 'text-blue-400' },
                                    model: { bg: 'bg-emerald-500/10', border: 'border-l-emerald-500', label: 'text-emerald-400' },
                                    system: { bg: 'bg-zinc-500/10', border: 'border-l-zinc-500', label: 'text-zinc-400' },
                                  };
                                  const colors = roleColors[msg.role];

                                  return (
                                    <div key={msgIdx} className={`p-3 ${colors.bg} border-l-2 ${colors.border}`}>
                                      <div className="flex items-center justify-between mb-2">
                                        <span className={`text-xs font-medium uppercase tracking-wide ${colors.label}`}>
                                          {msg.role}
                                        </span>
                                        {msg.maxActivation > 0 && (
                                          <span className="text-xs text-zinc-500">
                                            max: <span className="text-yellow-300 font-mono">{msg.maxActivation.toFixed(3)}</span>
                                          </span>
                                        )}
                                      </div>
                                      <p className="text-sm leading-relaxed">
                                        {msg.tokens.map((t, tokenIdx) => {
                                          const maxVal = act.maxValue || 1;
                                          const intensity = maxVal > 0 ? Math.min(t.activation / maxVal, 1) : 0;
                                          const isMax = t.activation > 0 && t.activation === act.maxValue;

                                          // Clean up token display
                                          let displayToken = t.token;
                                          const needsSpaceBefore = displayToken.startsWith('▁') ||
                                                                   displayToken.startsWith('Ġ') ||
                                                                   displayToken.startsWith('·') ||
                                                                   displayToken.startsWith(' ');
                                          if (needsSpaceBefore) {
                                            displayToken = displayToken.slice(1);
                                          }

                                          // Check for newline tokens
                                          const isNewline = displayToken === '\\n' ||
                                                            displayToken === '\n' ||
                                                            displayToken === '<0x0A>' ||
                                                            displayToken === '<newline>' ||
                                                            displayToken === '⏎';

                                          if (isNewline) {
                                            return <br key={tokenIdx} />;
                                          }

                                          return (
                                            <span key={tokenIdx}>
                                              {needsSpaceBefore && tokenIdx > 0 && ' '}
                                              <span
                                                className={`rounded-sm ${
                                                  isMax
                                                    ? "bg-yellow-400/70 text-yellow-50 font-semibold px-0.5 py-0.5"
                                                    : intensity > 0.3
                                                    ? "bg-orange-500/60 text-orange-50 px-0.5"
                                                    : intensity > 0.05
                                                    ? "bg-zinc-500/50 text-zinc-100"
                                                    : "text-zinc-300"
                                                }`}
                                                title={`"${t.token}" - Activation: ${t.activation.toFixed(3)}`}
                                              >
                                                {displayToken}
                                              </span>
                                            </span>
                                          );
                                        })}
                                      </p>
                                    </div>
                                  );
                                })}
                              </div>
                              {/* Overall max activation */}
                              <div className="px-3 py-2 bg-zinc-800/50 text-xs text-zinc-500 flex items-center justify-between">
                                <span>Overall max activation:</span>
                                <span className="text-yellow-300 font-mono">{act.maxValue.toFixed(3)}</span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Feature Occurrences */}
            <div className="shrink-0">
              <div className="text-xs text-zinc-400 mb-2">
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
                    {selectedFeatureData.occurrences.slice(0, 10).map((occ, idx) => (
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
          </div>
        )}
      </div>
    </div>
  );
}
