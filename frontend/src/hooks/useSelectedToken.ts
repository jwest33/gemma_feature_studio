"use client";

import { useState, useCallback } from "react";
import type { TokenActivations } from "@/types/analysis";

interface UseSelectedTokenResult {
  selectedToken: TokenActivations | null;
  selectedIndex: number | null;
  selectToken: (token: TokenActivations, index: number) => void;
  clearSelection: () => void;
}

export function useSelectedToken(): UseSelectedTokenResult {
  const [selectedToken, setSelectedToken] = useState<TokenActivations | null>(
    null
  );
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

  const selectToken = useCallback(
    (token: TokenActivations, index: number) => {
      setSelectedToken(token);
      setSelectedIndex(index);
    },
    []
  );

  const clearSelection = useCallback(() => {
    setSelectedToken(null);
    setSelectedIndex(null);
  }, []);

  return { selectedToken, selectedIndex, selectToken, clearSelection };
}
