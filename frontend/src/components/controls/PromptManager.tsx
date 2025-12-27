"use client";

import { useState, useCallback, ChangeEvent, KeyboardEvent } from "react";
import { useFlowStore } from "@/state/flowStore";

interface PromptManagerProps {
  compact?: boolean;
}

export function PromptManager({ compact = false }: PromptManagerProps) {
  const {
    prompts,
    activePromptId,
    addPrompt,
    addPrompts,
    removePrompt,
    setActivePrompt,
    clearPrompts,
  } = useFlowStore();

  const [inputValue, setInputValue] = useState("");
  const [isExpanded, setIsExpanded] = useState(false);

  const handleAddPrompt = useCallback(() => {
    const text = inputValue.trim();
    if (text) {
      // Check if multiple lines (batch mode)
      const lines = text.split("\n").filter((line) => line.trim().length > 0);
      if (lines.length > 1) {
        addPrompts(lines);
      } else {
        addPrompt(text);
      }
      setInputValue("");
    }
  }, [inputValue, addPrompt, addPrompts]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleAddPrompt();
      }
    },
    [handleAddPrompt]
  );

  const handleFileUpload = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target?.result as string;
        const lines = text
          .split("\n")
          .map((line) => line.trim())
          .filter((line) => line.length > 0);
        if (lines.length > 0) {
          addPrompts(lines);
        }
      };
      reader.readAsText(file);

      // Reset input
      e.target.value = "";
    },
    [addPrompts]
  );

  if (compact) {
    return (
      <div className="flex items-center gap-2">
        <div className="flex items-center gap-1.5">
          <span className="text-sm text-zinc-400">Prompts:</span>
          <span className="text-sm text-zinc-300 font-mono">{prompts.length}</span>
        </div>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="p-1 text-zinc-500 hover:text-zinc-300 transition-colors"
          title={isExpanded ? "Collapse" : "Expand"}
        >
          <svg
            className={`w-4 h-4 transition-transform ${isExpanded ? "rotate-180" : ""}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 9l-7 7-7-7"
            />
          </svg>
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {/* Input Area */}
      <div className="flex gap-2">
        <div className="flex-1 relative">
          <textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Enter prompt (Shift+Enter for new line, Enter to add)..."
            className="w-full h-20 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg
                       text-white placeholder-zinc-500 resize-none
                       focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
          />
          <span className="absolute bottom-2 right-2 text-xs text-zinc-500">
            {inputValue.split("\n").filter((l) => l.trim()).length} lines
          </span>
        </div>
        <div className="flex flex-col gap-2">
          <button
            onClick={handleAddPrompt}
            disabled={!inputValue.trim()}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg
                       hover:bg-blue-500 transition-colors
                       disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Add
          </button>
          <label className="px-4 py-2 bg-zinc-700 text-zinc-300 rounded-lg
                           hover:bg-zinc-600 transition-colors cursor-pointer text-center">
            <input
              type="file"
              accept=".txt,.csv"
              onChange={handleFileUpload}
              className="sr-only"
            />
            Upload
          </label>
        </div>
      </div>

      {/* Prompt List */}
      {prompts.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm text-zinc-400">
              {prompts.length} prompt{prompts.length !== 1 ? "s" : ""}
            </span>
            <button
              onClick={clearPrompts}
              className="text-xs text-zinc-500 hover:text-red-400 transition-colors"
            >
              Clear All
            </button>
          </div>
          <div className="flex flex-col gap-1 max-h-48 overflow-y-auto">
            {prompts.map((prompt, idx) => (
              <div
                key={prompt.id}
                className={`
                  flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer
                  transition-colors
                  ${
                    prompt.id === activePromptId
                      ? "bg-blue-500/20 border border-blue-500/50"
                      : "bg-zinc-800/50 border border-zinc-700 hover:border-zinc-500"
                  }
                `}
                onClick={() => setActivePrompt(prompt.id)}
              >
                <span className="text-zinc-500 text-xs w-6">{idx + 1}.</span>
                <span className="flex-1 text-sm text-white truncate">
                  {prompt.text}
                </span>
                {prompt.analyzedAt && (
                  <span className="w-2 h-2 rounded-full bg-green-500" title="Analyzed" />
                )}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    removePrompt(prompt.id);
                  }}
                  className="p-1 text-zinc-500 hover:text-red-400 transition-colors"
                  title="Remove"
                >
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
