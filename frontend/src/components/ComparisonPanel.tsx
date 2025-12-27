"use client";

import { useState, useCallback, useMemo } from "react";
import type { SteeringFeature } from "@/types/analysis";
import { useStreamingGeneration } from "@/hooks/useStreamingGeneration";

interface ComparisonPanelProps {
  steeringFeatures: SteeringFeature[];
  onSaveRun?: (
    prompt: string,
    baselineOutput: string,
    steeredOutput: string,
    temperature: number,
    maxTokens: number
  ) => void;
}

interface OutputCardProps {
  title: string;
  output: string;
  isStreaming: boolean;
  error: string | null;
  variant: "baseline" | "steered";
}

function OutputCard({ title, output, isStreaming, error, variant }: OutputCardProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    if (!output) return;
    await navigator.clipboard.writeText(output);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [output]);

  const borderColor = variant === "steered" ? "border-blue-600" : "border-zinc-700";
  const headerBg = variant === "steered" ? "bg-blue-900/20" : "bg-zinc-800/50";

  return (
    <div className={`flex-1 bg-zinc-900 rounded-lg border ${borderColor} overflow-hidden`}>
      <div className={`flex items-center justify-between px-4 py-2 ${headerBg} border-b border-zinc-800`}>
        <div className="flex items-center gap-2">
          <h4 className="text-sm font-medium text-zinc-300">{title}</h4>
          {isStreaming && (
            <span className="flex items-center gap-1 text-xs text-blue-400">
              <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse" />
              Generating...
            </span>
          )}
        </div>
        <button
          onClick={handleCopy}
          disabled={!output}
          className="p-1.5 text-zinc-500 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          title="Copy to clipboard"
        >
          {copied ? (
            <svg className="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          ) : (
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
          )}
        </button>
      </div>
      <div className="p-4 min-h-[150px] max-h-[300px] overflow-y-auto">
        {error ? (
          <div className="text-red-400 text-sm">{error}</div>
        ) : output ? (
          <div className="text-zinc-200 text-sm whitespace-pre-wrap font-mono leading-relaxed">
            {output}
            {isStreaming && <span className="inline-block w-2 h-4 bg-white/70 animate-pulse ml-0.5" />}
          </div>
        ) : (
          <div className="text-zinc-600 text-sm italic">
            {isStreaming ? "Waiting for response..." : "No output yet"}
          </div>
        )}
      </div>
    </div>
  );
}

function DiffView({ baseline, steered }: { baseline: string; steered: string }) {
  const diff = useMemo(() => {
    if (!baseline || !steered) return null;

    // Simple word-level diff
    const baseWords = baseline.split(/(\s+)/);
    const steeredWords = steered.split(/(\s+)/);

    const result: Array<{ text: string; type: "same" | "removed" | "added" }> = [];

    let i = 0;
    let j = 0;

    while (i < baseWords.length || j < steeredWords.length) {
      if (i >= baseWords.length) {
        result.push({ text: steeredWords[j], type: "added" });
        j++;
      } else if (j >= steeredWords.length) {
        result.push({ text: baseWords[i], type: "removed" });
        i++;
      } else if (baseWords[i] === steeredWords[j]) {
        result.push({ text: baseWords[i], type: "same" });
        i++;
        j++;
      } else {
        // Simple heuristic: add removed then added
        result.push({ text: baseWords[i], type: "removed" });
        result.push({ text: steeredWords[j], type: "added" });
        i++;
        j++;
      }
    }

    return result;
  }, [baseline, steered]);

  if (!diff) return null;

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-700 overflow-hidden mt-4">
      <div className="px-4 py-2 bg-zinc-800/50 border-b border-zinc-800">
        <h4 className="text-sm font-medium text-zinc-300">Diff View</h4>
      </div>
      <div className="p-4 max-h-[200px] overflow-y-auto">
        <div className="text-sm font-mono leading-relaxed">
          {diff.map((item, idx) => (
            <span
              key={idx}
              className={
                item.type === "removed"
                  ? "bg-red-900/40 text-red-300 line-through"
                  : item.type === "added"
                  ? "bg-green-900/40 text-green-300"
                  : "text-zinc-300"
              }
            >
              {item.text}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

export function ComparisonPanel({ steeringFeatures, onSaveRun }: ComparisonPanelProps) {
  const [prompt, setPrompt] = useState("");
  const [maxTokens, setMaxTokens] = useState(100);
  const [temperature, setTemperature] = useState(0.7);
  const [showDiff, setShowDiff] = useState(false);
  const [lastSavedPrompt, setLastSavedPrompt] = useState<string | null>(null);

  const { result, generate, cancel, reset } = useStreamingGeneration();

  const activeFeatures = useMemo(
    () => steeringFeatures.filter((f) => f.coefficient !== 0),
    [steeringFeatures]
  );

  // Auto-save run when generation completes
  const isGenerating = result.baseline.isStreaming || result.steered.isStreaming;
  const hasOutput = result.baseline.output && result.steered.output;

  // Save run when generation completes
  useMemo(() => {
    if (
      !isGenerating &&
      hasOutput &&
      onSaveRun &&
      prompt.trim() &&
      prompt !== lastSavedPrompt
    ) {
      onSaveRun(
        prompt,
        result.baseline.output,
        result.steered.output,
        temperature,
        maxTokens
      );
      setLastSavedPrompt(prompt);
    }
  }, [isGenerating, hasOutput, onSaveRun, prompt, lastSavedPrompt, result, temperature, maxTokens]);

  const handleGenerate = useCallback(async () => {
    if (!prompt.trim()) return;
    setLastSavedPrompt(null); // Reset so new run will be saved
    await generate(prompt, activeFeatures, {
      maxTokens,
      temperature,
      includeBaseline: true,
    });
  }, [prompt, activeFeatures, maxTokens, temperature, generate]);

  const handleCancel = useCallback(() => {
    cancel();
  }, [cancel]);

  return (
    <div className="space-y-4">
      {/* Prompt Input */}
      <div className="bg-zinc-900 rounded-lg p-4 border border-zinc-800">
        <div className="flex gap-4 mb-3">
          <div className="flex-1">
            <label className="block text-xs text-zinc-500 mb-1">Prompt</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter a prompt to compare baseline vs steered generation..."
              className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded text-sm text-white placeholder-zinc-500 focus:outline-none focus:border-zinc-600 resize-none"
              rows={3}
            />
          </div>
        </div>

        <div className="flex items-end gap-4">
          <div>
            <label className="block text-xs text-zinc-500 mb-1">Max Tokens</label>
            <input
              type="number"
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value) || 100)}
              min={1}
              max={500}
              className="w-24 px-3 py-1.5 bg-zinc-800 border border-zinc-700 rounded text-sm text-white focus:outline-none focus:border-zinc-600"
            />
          </div>
          <div>
            <label className="block text-xs text-zinc-500 mb-1">Temperature</label>
            <input
              type="number"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value) || 0.7)}
              min={0}
              max={2}
              step={0.1}
              className="w-24 px-3 py-1.5 bg-zinc-800 border border-zinc-700 rounded text-sm text-white focus:outline-none focus:border-zinc-600"
            />
          </div>
          <div className="flex-1" />
          <div className="flex gap-2">
            {isGenerating ? (
              <button
                onClick={handleCancel}
                className="px-4 py-1.5 bg-red-600 text-white text-sm rounded hover:bg-red-500 transition-colors"
              >
                Cancel
              </button>
            ) : (
              <button
                onClick={handleGenerate}
                disabled={!prompt.trim() || activeFeatures.length === 0}
                className="px-4 py-1.5 bg-blue-600 text-white text-sm rounded hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Generate
              </button>
            )}
            <button
              onClick={reset}
              className="px-4 py-1.5 bg-zinc-700 text-white text-sm rounded hover:bg-zinc-600 transition-colors"
            >
              Clear
            </button>
          </div>
        </div>

        {activeFeatures.length === 0 && (
          <div className="mt-3 text-xs text-amber-400">
            No active steering features. Add features with non-zero coefficients to compare outputs.
          </div>
        )}
      </div>

      {/* Output Comparison */}
      <div className="flex gap-4">
        <OutputCard
          title="Baseline (No Steering)"
          output={result.baseline.output}
          isStreaming={result.baseline.isStreaming}
          error={result.baseline.error}
          variant="baseline"
        />
        <OutputCard
          title="Steered Output"
          output={result.steered.output}
          isStreaming={result.steered.isStreaming}
          error={result.steered.error}
          variant="steered"
        />
      </div>

      {/* Diff Toggle */}
      {result.baseline.output && result.steered.output && !isGenerating && (
        <>
          <button
            onClick={() => setShowDiff(!showDiff)}
            className="text-sm text-zinc-400 hover:text-white transition-colors flex items-center gap-1"
          >
            <svg
              className={`w-4 h-4 transition-transform ${showDiff ? "rotate-90" : ""}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            {showDiff ? "Hide" : "Show"} Diff
          </button>
          {showDiff && (
            <DiffView
              baseline={result.baseline.output}
              steered={result.steered.output}
            />
          )}
        </>
      )}
    </div>
  );
}
