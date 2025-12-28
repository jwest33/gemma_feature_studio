"use client";

import { useState, useCallback } from "react";
import { useFlowStore, useModelConfig } from "@/state/flowStore";
import { SAE_PRESETS } from "@/types/flow";
import type { SAEPreset } from "@/types/flow";

interface ModelSelectorProps {
  disabled?: boolean;
  onConfigChange?: () => void;
}

export function ModelSelector({ disabled = false, onConfigChange }: ModelSelectorProps) {
  const modelConfig = useModelConfig();
  const { setModelPath, setSaePreset, getSaePreset } = useFlowStore();

  const [localModelPath, setLocalModelPath] = useState(modelConfig.modelPath);
  const [isEditing, setIsEditing] = useState(false);

  const currentPreset = getSaePreset();

  const handleModelPathBlur = useCallback(() => {
    if (localModelPath !== modelConfig.modelPath) {
      setModelPath(localModelPath);
      onConfigChange?.();
    }
    setIsEditing(false);
  }, [localModelPath, modelConfig.modelPath, setModelPath, onConfigChange]);

  const handleModelPathKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleModelPathBlur();
    }
  }, [handleModelPathBlur]);

  const handlePresetChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    setSaePreset(e.target.value);
    onConfigChange?.();
  }, [setSaePreset, onConfigChange]);

  return (
    <div className="flex items-center gap-6">
      {/* Model Path Input */}
      <div className="flex items-center gap-2">
        <label className="text-sm text-zinc-400 font-medium whitespace-nowrap">
          Model:
        </label>
        <input
          type="text"
          value={isEditing ? localModelPath : modelConfig.modelPath}
          onChange={(e) => {
            setLocalModelPath(e.target.value);
            setIsEditing(true);
          }}
          onBlur={handleModelPathBlur}
          onKeyDown={handleModelPathKeyDown}
          onFocus={() => setLocalModelPath(modelConfig.modelPath)}
          disabled={disabled}
          placeholder="HuggingFace model ID or local path"
          className={`
            w-64 px-3 py-1.5 rounded-md text-sm
            bg-zinc-800/50 border border-zinc-700
            text-white placeholder-zinc-500
            focus:outline-none focus:border-zinc-500
            ${disabled ? "opacity-50 cursor-not-allowed" : ""}
          `}
        />
      </div>

      {/* SAE Preset Dropdown */}
      <div className="flex items-center gap-2">
        <label className="text-sm text-zinc-400 font-medium whitespace-nowrap">
          SAE:
        </label>
        <select
          value={modelConfig.saePresetId}
          onChange={handlePresetChange}
          disabled={disabled}
          className={`
            px-3 py-1.5 rounded-md text-sm
            bg-zinc-800/50 border border-zinc-700
            text-white
            focus:outline-none focus:border-zinc-500
            ${disabled ? "opacity-50 cursor-not-allowed" : ""}
          `}
        >
          {SAE_PRESETS.map((preset) => (
            <option key={preset.id} value={preset.id}>
              {preset.label}
            </option>
          ))}
        </select>
      </div>

      {/* Current config hint */}
      {currentPreset && (
        <span className="text-xs text-zinc-500">
          {currentPreset.width} features
        </span>
      )}
    </div>
  );
}
