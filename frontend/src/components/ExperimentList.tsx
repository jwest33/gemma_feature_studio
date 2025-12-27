"use client";

import { useState, useCallback, useRef } from "react";
import { useExperimentStore } from "@/state/experimentStore";
import type { ExperimentSummary } from "@/types/experiment";

interface ExperimentCardProps {
  experiment: ExperimentSummary;
  isActive: boolean;
  onSelect: () => void;
  onDuplicate: () => void;
  onDelete: () => void;
  onExport: () => void;
}

function ExperimentCard({
  experiment,
  isActive,
  onSelect,
  onDuplicate,
  onDelete,
  onExport,
}: ExperimentCardProps) {
  const [showMenu, setShowMenu] = useState(false);

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <div
      className={`bg-zinc-900 rounded-lg border p-4 transition-colors cursor-pointer ${
        isActive
          ? "border-blue-500 ring-1 ring-blue-500/50"
          : "border-zinc-800 hover:border-zinc-700"
      }`}
      onClick={onSelect}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <h3 className="font-medium text-white truncate">{experiment.name}</h3>
          {experiment.description && (
            <p className="text-sm text-zinc-500 mt-1 line-clamp-2">
              {experiment.description}
            </p>
          )}
          <div className="flex items-center gap-4 mt-2 text-xs text-zinc-500">
            <span>{experiment.featureCount} features</span>
            <span>{experiment.runCount} runs</span>
          </div>
          <div className="text-xs text-zinc-600 mt-1">
            Updated {formatDate(experiment.updatedAt)}
          </div>
        </div>

        <div className="relative ml-2">
          <button
            onClick={(e) => {
              e.stopPropagation();
              setShowMenu(!showMenu);
            }}
            className="p-1.5 text-zinc-500 hover:text-white hover:bg-zinc-800 rounded transition-colors"
          >
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path d="M10 6a2 2 0 110-4 2 2 0 010 4zM10 12a2 2 0 110-4 2 2 0 010 4zM10 18a2 2 0 110-4 2 2 0 010 4z" />
            </svg>
          </button>

          {showMenu && (
            <>
              <div
                className="fixed inset-0 z-10"
                onClick={(e) => {
                  e.stopPropagation();
                  setShowMenu(false);
                }}
              />
              <div className="absolute right-0 top-full mt-1 w-36 bg-zinc-800 border border-zinc-700 rounded shadow-lg z-20">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDuplicate();
                    setShowMenu(false);
                  }}
                  className="w-full px-3 py-2 text-left text-sm text-zinc-300 hover:bg-zinc-700 transition-colors"
                >
                  Duplicate
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onExport();
                    setShowMenu(false);
                  }}
                  className="w-full px-3 py-2 text-left text-sm text-zinc-300 hover:bg-zinc-700 transition-colors"
                >
                  Export JSON
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete();
                    setShowMenu(false);
                  }}
                  className="w-full px-3 py-2 text-left text-sm text-red-400 hover:bg-zinc-700 transition-colors"
                >
                  Delete
                </button>
              </div>
            </>
          )}
        </div>
      </div>

      {isActive && (
        <div className="mt-3 pt-3 border-t border-zinc-800">
          <span className="text-xs text-blue-400 flex items-center gap-1">
            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                clipRule="evenodd"
              />
            </svg>
            Active experiment
          </span>
        </div>
      )}
    </div>
  );
}

export function ExperimentList() {
  const store = useExperimentStore();
  const experiments = store.getExperimentSummaries();
  const [newName, setNewName] = useState("");
  const [isCreating, setIsCreating] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleCreate = useCallback(() => {
    if (!newName.trim()) return;
    store.createExperiment(newName.trim());
    setNewName("");
    setIsCreating(false);
  }, [newName, store]);

  const handleExport = useCallback(
    (id: string) => {
      const json = store.exportExperiment(id);
      if (!json) return;

      const experiment = store.getExperimentById(id);
      const filename = `${experiment?.name.replace(/[^a-z0-9]/gi, "_") ?? "experiment"}.json`;

      const blob = new Blob([json], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    },
    [store]
  );

  const handleExportAll = useCallback(() => {
    const json = store.exportAllExperiments();
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "all_experiments.json";
    a.click();
    URL.revokeObjectURL(url);
  }, [store]);

  const handleImport = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = (event) => {
        const content = event.target?.result as string;
        try {
          // Try to parse as array first (multiple experiments)
          const data = JSON.parse(content);
          if (Array.isArray(data)) {
            const count = store.importExperiments(content);
            if (count > 0) {
              setImportError(null);
            } else {
              setImportError("No valid experiments found in file");
            }
          } else {
            // Single experiment
            const result = store.importExperiment(content);
            if (result) {
              setImportError(null);
            } else {
              setImportError("Invalid experiment format");
            }
          }
        } catch {
          setImportError("Failed to parse JSON file");
        }
      };
      reader.readAsText(file);

      // Reset input
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    },
    [store]
  );

  const handleDelete = useCallback(
    (id: string) => {
      if (confirm("Are you sure you want to delete this experiment?")) {
        store.deleteExperiment(id);
      }
    },
    [store]
  );

  // Sort by updated date, newest first
  const sortedExperiments = [...experiments].sort(
    (a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-medium text-white">Experiments</h2>
          <p className="text-sm text-zinc-500">
            Save and manage your steering configurations
          </p>
        </div>

        <div className="flex items-center gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            onChange={handleImport}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-3 py-1.5 text-sm text-zinc-400 border border-zinc-700 rounded hover:bg-zinc-800 transition-colors"
          >
            Import
          </button>
          {experiments.length > 0 && (
            <button
              onClick={handleExportAll}
              className="px-3 py-1.5 text-sm text-zinc-400 border border-zinc-700 rounded hover:bg-zinc-800 transition-colors"
            >
              Export All
            </button>
          )}
          <button
            onClick={() => setIsCreating(true)}
            className="px-4 py-1.5 text-sm bg-blue-600 text-white rounded hover:bg-blue-500 transition-colors"
          >
            New Experiment
          </button>
        </div>
      </div>

      {/* Import Error */}
      {importError && (
        <div className="p-3 bg-red-900/50 border border-red-700 rounded text-red-200 text-sm flex items-center justify-between">
          <span>{importError}</span>
          <button
            onClick={() => setImportError(null)}
            className="text-red-400 hover:text-red-200"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      )}

      {/* Create New */}
      {isCreating && (
        <div className="bg-zinc-900 rounded-lg border border-zinc-700 p-4">
          <h3 className="text-sm font-medium text-white mb-3">Create New Experiment</h3>
          <div className="flex gap-2">
            <input
              type="text"
              placeholder="Experiment name"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleCreate()}
              autoFocus
              className="flex-1 px-3 py-1.5 bg-zinc-800 border border-zinc-700 rounded text-sm text-white placeholder-zinc-500 focus:outline-none focus:border-zinc-600"
            />
            <button
              onClick={handleCreate}
              disabled={!newName.trim()}
              className="px-4 py-1.5 bg-blue-600 text-white text-sm rounded hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Create
            </button>
            <button
              onClick={() => {
                setIsCreating(false);
                setNewName("");
              }}
              className="px-4 py-1.5 text-zinc-400 text-sm rounded hover:bg-zinc-800 transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Experiment Grid */}
      {sortedExperiments.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {sortedExperiments.map((exp) => (
            <ExperimentCard
              key={exp.id}
              experiment={exp}
              isActive={exp.id === store.currentExperimentId}
              onSelect={() => store.setCurrentExperiment(exp.id)}
              onDuplicate={() => store.duplicateExperiment(exp.id)}
              onDelete={() => handleDelete(exp.id)}
              onExport={() => handleExport(exp.id)}
            />
          ))}
        </div>
      ) : (
        <div className="text-center py-12 bg-zinc-900 rounded-lg border border-zinc-800">
          <div className="text-zinc-500 mb-4">
            <svg
              className="w-12 h-12 mx-auto mb-3"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"
              />
            </svg>
            <p className="text-lg">No experiments yet</p>
            <p className="text-sm mt-1">
              Create your first experiment to start saving steering configurations
            </p>
          </div>
          <button
            onClick={() => setIsCreating(true)}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-500 transition-colors"
          >
            Create Experiment
          </button>
        </div>
      )}
    </div>
  );
}
