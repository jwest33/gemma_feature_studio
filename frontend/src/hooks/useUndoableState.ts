"use client";

import { useState, useCallback, useEffect, useRef } from "react";

interface UndoableStateOptions {
  historyLimit?: number;
  enableKeyboardShortcuts?: boolean;
}

interface UndoableStateReturn<T> {
  state: T;
  setState: (value: T | ((prev: T) => T)) => void;
  undo: () => void;
  redo: () => void;
  canUndo: boolean;
  canRedo: boolean;
  reset: (value: T) => void;
  historyLength: number;
}

export function useUndoableState<T>(
  initialState: T,
  options: UndoableStateOptions = {}
): UndoableStateReturn<T> {
  const { historyLimit = 50, enableKeyboardShortcuts = true } = options;

  const [history, setHistory] = useState<T[]>([initialState]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const skipNextPush = useRef(false);

  const state = history[currentIndex];

  const setState = useCallback(
    (value: T | ((prev: T) => T)) => {
      if (skipNextPush.current) {
        skipNextPush.current = false;
        return;
      }

      setHistory((prev) => {
        const currentState = prev[currentIndex];
        const newState =
          typeof value === "function"
            ? (value as (prev: T) => T)(currentState)
            : value;

        // Don't add to history if state hasn't changed
        if (JSON.stringify(newState) === JSON.stringify(currentState)) {
          return prev;
        }

        // Remove any future history when making a new change
        const newHistory = prev.slice(0, currentIndex + 1);
        newHistory.push(newState);

        // Trim history to limit
        if (newHistory.length > historyLimit) {
          return newHistory.slice(-historyLimit);
        }

        return newHistory;
      });

      setCurrentIndex((prev) => {
        const newIndex = Math.min(prev + 1, historyLimit - 1);
        return newIndex;
      });
    },
    [currentIndex, historyLimit]
  );

  const undo = useCallback(() => {
    setCurrentIndex((prev) => Math.max(0, prev - 1));
  }, []);

  const redo = useCallback(() => {
    setCurrentIndex((prev) => Math.min(history.length - 1, prev + 1));
  }, [history.length]);

  const reset = useCallback((value: T) => {
    setHistory([value]);
    setCurrentIndex(0);
  }, []);

  const canUndo = currentIndex > 0;
  const canRedo = currentIndex < history.length - 1;

  // Keyboard shortcuts
  useEffect(() => {
    if (!enableKeyboardShortcuts) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "z") {
        e.preventDefault();
        if (e.shiftKey) {
          redo();
        } else {
          undo();
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [enableKeyboardShortcuts, undo, redo]);

  return {
    state,
    setState,
    undo,
    redo,
    canUndo,
    canRedo,
    reset,
    historyLength: history.length,
  };
}
