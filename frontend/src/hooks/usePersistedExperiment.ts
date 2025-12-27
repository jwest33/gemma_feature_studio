"use client";

import { useCallback, useEffect } from "react";
import { useExperimentStore } from "@/state/experimentStore";
import type { SteeringFeatureState } from "@/types/experiment";

interface UsePersistedExperimentReturn {
  // Current experiment
  experimentId: string | null;
  experimentName: string | null;
  features: SteeringFeatureState[];
  hasExperiment: boolean;

  // Experiment actions
  createNew: (name: string) => void;
  rename: (name: string) => void;
  close: () => void;

  // Feature actions
  addFeature: (id: number, coefficient?: number) => void;
  updateFeature: (id: number, coefficient: number) => void;
  toggleFeature: (id: number, enabled: boolean) => void;
  removeFeature: (id: number) => void;
  setAllFeatures: (features: SteeringFeatureState[]) => void;
  resetAllFeatures: () => void;

  // Run actions
  saveRun: (
    prompt: string,
    baselineOutput: string,
    steeredOutput: string,
    temperature?: number,
    maxTokens?: number
  ) => void;

  // Export
  exportToJson: () => string | null;
}

export function usePersistedExperiment(): UsePersistedExperimentReturn {
  const store = useExperimentStore();
  const current = store.getCurrentExperiment();

  // Auto-create experiment if none exists
  useEffect(() => {
    if (store.experiments.length === 0) {
      store.createExperiment("Untitled Experiment");
    } else if (!store.currentExperimentId && store.experiments.length > 0) {
      store.setCurrentExperiment(store.experiments[0].id);
    }
  }, [store]);

  const createNew = useCallback(
    (name: string) => {
      store.createExperiment(name);
    },
    [store]
  );

  const rename = useCallback(
    (name: string) => {
      if (current) {
        store.updateExperiment(current.id, { name });
      }
    },
    [current, store]
  );

  const close = useCallback(() => {
    store.setCurrentExperiment(null);
  }, [store]);

  const addFeature = useCallback(
    (id: number, coefficient: number = 0.5) => {
      store.addFeature({ id, coefficient, enabled: true });
    },
    [store]
  );

  const updateFeature = useCallback(
    (id: number, coefficient: number) => {
      store.updateFeature(id, { coefficient });
    },
    [store]
  );

  const toggleFeature = useCallback(
    (id: number, enabled: boolean) => {
      store.updateFeature(id, { enabled });
    },
    [store]
  );

  const removeFeature = useCallback(
    (id: number) => {
      store.removeFeature(id);
    },
    [store]
  );

  const setAllFeatures = useCallback(
    (features: SteeringFeatureState[]) => {
      store.setFeatures(features);
    },
    [store]
  );

  const resetAllFeatures = useCallback(() => {
    store.resetFeatures();
  }, [store]);

  const saveRun = useCallback(
    (
      prompt: string,
      baselineOutput: string,
      steeredOutput: string,
      temperature?: number,
      maxTokens?: number
    ) => {
      store.addRun(prompt, baselineOutput, steeredOutput, temperature, maxTokens);
    },
    [store]
  );

  const exportToJson = useCallback(() => {
    if (current) {
      return store.exportExperiment(current.id);
    }
    return null;
  }, [current, store]);

  return {
    experimentId: current?.id ?? null,
    experimentName: current?.name ?? null,
    features: current?.steering.features ?? [],
    hasExperiment: !!current,

    createNew,
    rename,
    close,

    addFeature,
    updateFeature,
    toggleFeature,
    removeFeature,
    setAllFeatures,
    resetAllFeatures,

    saveRun,
    exportToJson,
  };
}
