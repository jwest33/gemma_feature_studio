import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import type {
  SteeringExperiment,
  SteeringFeatureState,
  ExperimentRun,
  ExperimentSummary,
} from "@/types/experiment";
import {
  createExperiment,
  createExperimentRun,
  experimentToSummary,
} from "@/types/experiment";

interface ExperimentState {
  experiments: SteeringExperiment[];
  currentExperimentId: string | null;

  // Selectors
  getCurrentExperiment: () => SteeringExperiment | null;
  getExperimentById: (id: string) => SteeringExperiment | null;
  getExperimentSummaries: () => ExperimentSummary[];

  // Experiment CRUD
  createExperiment: (name: string) => SteeringExperiment;
  updateExperiment: (id: string, updates: Partial<SteeringExperiment>) => void;
  deleteExperiment: (id: string) => void;
  duplicateExperiment: (id: string) => SteeringExperiment | null;
  setCurrentExperiment: (id: string | null) => void;

  // Steering features
  addFeature: (feature: SteeringFeatureState) => void;
  updateFeature: (featureId: number, updates: Partial<SteeringFeatureState>) => void;
  removeFeature: (featureId: number) => void;
  setFeatures: (features: SteeringFeatureState[]) => void;
  resetFeatures: () => void;

  // Runs
  addRun: (
    prompt: string,
    baselineOutput: string,
    steeredOutput: string,
    temperature?: number,
    maxTokens?: number
  ) => void;
  deleteRun: (runId: string) => void;
  clearRuns: () => void;

  // Import/Export
  exportExperiment: (id: string) => string | null;
  importExperiment: (json: string) => SteeringExperiment | null;
  exportAllExperiments: () => string;
  importExperiments: (json: string) => number;
}

export const useExperimentStore = create<ExperimentState>()(
  persist(
    (set, get) => ({
      experiments: [],
      currentExperimentId: null,

      // Selectors
      getCurrentExperiment: () => {
        const { experiments, currentExperimentId } = get();
        if (!currentExperimentId) return null;
        return experiments.find((e) => e.id === currentExperimentId) ?? null;
      },

      getExperimentById: (id: string) => {
        return get().experiments.find((e) => e.id === id) ?? null;
      },

      getExperimentSummaries: () => {
        return get().experiments.map(experimentToSummary);
      },

      // Experiment CRUD
      createExperiment: (name: string) => {
        const experiment = createExperiment(name);
        set((state) => ({
          experiments: [...state.experiments, experiment],
          currentExperimentId: experiment.id,
        }));
        return experiment;
      },

      updateExperiment: (id: string, updates: Partial<SteeringExperiment>) => {
        set((state) => ({
          experiments: state.experiments.map((e) =>
            e.id === id
              ? { ...e, ...updates, updatedAt: new Date().toISOString() }
              : e
          ),
        }));
      },

      deleteExperiment: (id: string) => {
        set((state) => ({
          experiments: state.experiments.filter((e) => e.id !== id),
          currentExperimentId:
            state.currentExperimentId === id ? null : state.currentExperimentId,
        }));
      },

      duplicateExperiment: (id: string) => {
        const original = get().getExperimentById(id);
        if (!original) return null;

        const now = new Date().toISOString();
        const duplicate: SteeringExperiment = {
          ...original,
          id: crypto.randomUUID(),
          name: `${original.name} (Copy)`,
          createdAt: now,
          updatedAt: now,
          runs: [], // Don't copy runs
        };

        set((state) => ({
          experiments: [...state.experiments, duplicate],
        }));

        return duplicate;
      },

      setCurrentExperiment: (id: string | null) => {
        set({ currentExperimentId: id });
      },

      // Steering features
      addFeature: (feature: SteeringFeatureState) => {
        const current = get().getCurrentExperiment();
        if (!current) return;

        // Don't add duplicates
        if (current.steering.features.some((f) => f.id === feature.id)) return;

        get().updateExperiment(current.id, {
          steering: {
            ...current.steering,
            features: [...current.steering.features, feature],
          },
        });
      },

      updateFeature: (featureId: number, updates: Partial<SteeringFeatureState>) => {
        const current = get().getCurrentExperiment();
        if (!current) return;

        get().updateExperiment(current.id, {
          steering: {
            ...current.steering,
            features: current.steering.features.map((f) =>
              f.id === featureId ? { ...f, ...updates } : f
            ),
          },
        });
      },

      removeFeature: (featureId: number) => {
        const current = get().getCurrentExperiment();
        if (!current) return;

        get().updateExperiment(current.id, {
          steering: {
            ...current.steering,
            features: current.steering.features.filter((f) => f.id !== featureId),
          },
        });
      },

      setFeatures: (features: SteeringFeatureState[]) => {
        const current = get().getCurrentExperiment();
        if (!current) return;

        get().updateExperiment(current.id, {
          steering: {
            ...current.steering,
            features,
          },
        });
      },

      resetFeatures: () => {
        const current = get().getCurrentExperiment();
        if (!current) return;

        get().updateExperiment(current.id, {
          steering: {
            ...current.steering,
            features: current.steering.features.map((f) => ({
              ...f,
              coefficient: 0,
              enabled: true,
            })),
          },
        });
      },

      // Runs
      addRun: (
        prompt: string,
        baselineOutput: string,
        steeredOutput: string,
        temperature = 0.7,
        maxTokens = 100
      ) => {
        const current = get().getCurrentExperiment();
        if (!current) return;

        const run = createExperimentRun(
          prompt,
          baselineOutput,
          steeredOutput,
          temperature,
          maxTokens
        );

        get().updateExperiment(current.id, {
          runs: [...current.runs, run],
        });
      },

      deleteRun: (runId: string) => {
        const current = get().getCurrentExperiment();
        if (!current) return;

        get().updateExperiment(current.id, {
          runs: current.runs.filter((r) => r.id !== runId),
        });
      },

      clearRuns: () => {
        const current = get().getCurrentExperiment();
        if (!current) return;

        get().updateExperiment(current.id, {
          runs: [],
        });
      },

      // Import/Export
      exportExperiment: (id: string) => {
        const experiment = get().getExperimentById(id);
        if (!experiment) return null;
        return JSON.stringify(experiment, null, 2);
      },

      importExperiment: (json: string) => {
        try {
          const data = JSON.parse(json) as SteeringExperiment;

          // Validate required fields
          if (!data.id || !data.name || !data.steering) {
            throw new Error("Invalid experiment format");
          }

          // Generate new ID to avoid conflicts
          const experiment: SteeringExperiment = {
            ...data,
            id: crypto.randomUUID(),
            name: data.name.includes("(Imported)")
              ? data.name
              : `${data.name} (Imported)`,
            updatedAt: new Date().toISOString(),
          };

          set((state) => ({
            experiments: [...state.experiments, experiment],
          }));

          return experiment;
        } catch {
          return null;
        }
      },

      exportAllExperiments: () => {
        return JSON.stringify(get().experiments, null, 2);
      },

      importExperiments: (json: string) => {
        try {
          const data = JSON.parse(json) as SteeringExperiment[];
          if (!Array.isArray(data)) return 0;

          const now = new Date().toISOString();
          const imported = data
            .filter((e) => e.id && e.name && e.steering)
            .map((e) => ({
              ...e,
              id: crypto.randomUUID(),
              updatedAt: now,
            }));

          set((state) => ({
            experiments: [...state.experiments, ...imported],
          }));

          return imported.length;
        } catch {
          return 0;
        }
      },
    }),
    {
      name: "lm-feature-studio-experiments",
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        experiments: state.experiments,
        currentExperimentId: state.currentExperimentId,
      }),
    }
  )
);
