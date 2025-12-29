"use client";

import { useState, useCallback } from "react";
import { useFlowStore, useModelConfig } from "@/state/flowStore";
import { SAE_PRESETS, detectModelSize, getSaeRepoForModelSize } from "@/types/flow";
import type { SAEPreset } from "@/types/flow";
import { configureModel, getSystemStatus } from "@/lib/api";

interface ModelSelectorProps {
  disabled?: boolean;
  onConfigChange?: () => void;
}

export function ModelSelector({ disabled = false, onConfigChange }: ModelSelectorProps) {
  const modelConfig = useModelConfig();
  const { setModelPath, setSaePreset, getSaePreset, setSystemStatus } = useFlowStore();

  const [localModelPath, setLocalModelPath] = useState(modelConfig.modelPath);
  const [isEditing, setIsEditing] = useState(false);
  const [isApplying, setIsApplying] = useState(false);
  const [applyError, setApplyError] = useState<string | null>(null);

  const currentPreset = getSaePreset();

  // Check if there are unapplied changes
  const hasChanges = localModelPath !== modelConfig.modelPath;

  const handleModelPathBlur = useCallback(() => {
    setIsEditing(false);
  }, []);

  const handlePresetChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    setSaePreset(e.target.value);
    onConfigChange?.();
  }, [setSaePreset, onConfigChange]);

  const handleApplyConfig = useCallback(async () => {
    if (isApplying || disabled) return;

    const pathToApply = localModelPath.trim() || modelConfig.modelPath;

    setIsApplying(true);
    setApplyError(null);

    try {
      // Detect model size and get appropriate SAE repo
      const modelSize = detectModelSize(pathToApply);
      const saeRepo = getSaeRepoForModelSize(modelSize);
      const preset = currentPreset || SAE_PRESETS[0];

      // Configure the backend
      await configureModel({
        model_name: pathToApply,
        sae_repo: saeRepo,
        sae_width: preset.width,
        sae_l0: preset.l0,
        sae_type: preset.type,
      });

      // Update frontend state
      setModelPath(pathToApply);

      // Refresh system status to get new available layers
      const status = await getSystemStatus();
      setSystemStatus(status);

      onConfigChange?.();
    } catch (err) {
      setApplyError(err instanceof Error ? err.message : "Failed to apply configuration");
    } finally {
      setIsApplying(false);
    }
  }, [
    isApplying,
    disabled,
    localModelPath,
    modelConfig.modelPath,
    currentPreset,
    setModelPath,
    setSystemStatus,
    onConfigChange,
  ]);

  const handleModelPathKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleApplyConfig();
    } else if (e.key === "Escape") {
      setLocalModelPath(modelConfig.modelPath);
      setIsEditing(false);
    }
  }, [handleApplyConfig, modelConfig.modelPath]);

  return (
    <div className="flex items-center gap-6">
      {/* Model Path Input with Apply Button */}
      <div className="flex items-center gap-2">
        <label className="text-sm text-zinc-400 font-medium whitespace-nowrap">
          Model:
        </label>
        <div className="flex items-center gap-1">
          <input
            type="text"
            value={isEditing ? localModelPath : modelConfig.modelPath}
            onChange={(e) => {
              setLocalModelPath(e.target.value);
              setIsEditing(true);
              setApplyError(null);
            }}
            onBlur={handleModelPathBlur}
            onKeyDown={handleModelPathKeyDown}
            onFocus={() => {
              setLocalModelPath(modelConfig.modelPath);
              setIsEditing(true);
            }}
            disabled={disabled || isApplying}
            placeholder="HuggingFace model ID or local path"
            className={`
              w-64 px-3 py-1.5 rounded-l-md text-sm
              bg-zinc-800/50 border border-zinc-700 border-r-0
              text-white placeholder-zinc-500
              focus:outline-none focus:border-zinc-500 focus:z-10
              ${disabled || isApplying ? "opacity-50 cursor-not-allowed" : ""}
              ${hasChanges ? "border-amber-500/50" : ""}
            `}
          />
          <button
            onClick={handleApplyConfig}
            disabled={disabled || isApplying}
            title={hasChanges ? "Apply model configuration (Enter)" : "Apply model configuration"}
            className={`
              px-3 py-1.5 rounded-r-md text-sm font-medium
              border border-zinc-700 border-l-0
              transition-colors duration-150
              flex items-center gap-1.5
              ${disabled || isApplying
                ? "bg-zinc-800/30 text-zinc-500 cursor-not-allowed"
                : hasChanges
                  ? "bg-amber-600 hover:bg-amber-500 text-white"
                  : "bg-zinc-700 hover:bg-zinc-600 text-zinc-300"
              }
            `}
          >
            {isApplying ? (
              <>
                <svg className="w-3.5 h-3.5 animate-spin" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                <span>Applying...</span>
              </>
            ) : (
              <>
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span>Apply</span>
              </>
            )}
          </button>
        </div>
        {applyError && (
          <span className="text-xs text-red-400 max-w-48 truncate" title={applyError}>
            {applyError}
          </span>
        )}
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
