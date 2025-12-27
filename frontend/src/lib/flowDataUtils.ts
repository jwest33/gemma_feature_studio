import type { TokenActivations, FeatureActivation } from "@/types/analysis";

/**
 * Represents a unique feature in a layer with all tokens that activate it
 */
export interface DeduplicatedFeature {
  featureId: number;
  layer: number;
  activations: Array<{
    tokenIndex: number;
    token: string;
    activation: number;
  }>;
  maxActivation: number;
  avgActivation: number;
}

/**
 * Represents the flow data for visualization with deduplicated features
 */
export interface LayerFlowData {
  layer: number;
  features: DeduplicatedFeature[];
  maxActivation: number;
}

/**
 * Represents a connection between a token and a feature
 */
export interface TokenFeatureConnection {
  tokenIndex: number;
  featureId: number;
  layer: number;
  activation: number;
}

/**
 * Represents a connection between features across layers
 */
export interface CrossLayerConnection {
  featureId: number;
  fromLayer: number;
  toLayer: number;
  fromActivation: number;
  toActivation: number;
  sharedTokens: number[]; // Token indices that activate both
}

/**
 * Transform token activations into deduplicated feature data
 */
export function deduplicateFeatures(
  tokenActivations: TokenActivations[],
  layer: number,
  topK: number = 10
): LayerFlowData {
  const featureMap = new Map<number, DeduplicatedFeature>();

  // Collect all activations per feature
  tokenActivations.forEach((tokenData, tokenIndex) => {
    tokenData.top_features.slice(0, topK).forEach((feat) => {
      if (!featureMap.has(feat.id)) {
        featureMap.set(feat.id, {
          featureId: feat.id,
          layer,
          activations: [],
          maxActivation: 0,
          avgActivation: 0,
        });
      }

      const feature = featureMap.get(feat.id)!;
      feature.activations.push({
        tokenIndex,
        token: tokenData.token,
        activation: feat.activation,
      });
      feature.maxActivation = Math.max(feature.maxActivation, feat.activation);
    });
  });

  // Calculate averages and convert to array
  const features = Array.from(featureMap.values());
  features.forEach((feat) => {
    feat.avgActivation =
      feat.activations.reduce((sum, a) => sum + a.activation, 0) /
      feat.activations.length;
  });

  // Sort by max activation (descending)
  features.sort((a, b) => b.maxActivation - a.maxActivation);

  // Calculate overall max for the layer
  const maxActivation = features.length > 0
    ? Math.max(...features.map((f) => f.maxActivation))
    : 1;

  return {
    layer,
    features,
    maxActivation,
  };
}

/**
 * Get all token-to-feature connections for flow lines
 */
export function getTokenFeatureConnections(
  tokenActivations: TokenActivations[],
  layer: number,
  topK: number = 10
): TokenFeatureConnection[] {
  const connections: TokenFeatureConnection[] = [];

  tokenActivations.forEach((tokenData, tokenIndex) => {
    tokenData.top_features.slice(0, topK).forEach((feat) => {
      connections.push({
        tokenIndex,
        featureId: feat.id,
        layer,
        activation: feat.activation,
      });
    });
  });

  return connections;
}

/**
 * Find features that appear in consecutive layers (for cross-layer flow lines)
 */
export function getCrossLayerConnections(
  layerFlowData: LayerFlowData[]
): CrossLayerConnection[] {
  const connections: CrossLayerConnection[] = [];

  // Sort by layer
  const sortedLayers = [...layerFlowData].sort((a, b) => a.layer - b.layer);

  for (let i = 0; i < sortedLayers.length - 1; i++) {
    const currentLayer = sortedLayers[i];
    const nextLayer = sortedLayers[i + 1];

    const currentFeatureIds = new Set(currentLayer.features.map((f) => f.featureId));
    const nextFeatureMap = new Map(nextLayer.features.map((f) => [f.featureId, f]));

    // Find shared features
    currentLayer.features.forEach((currentFeature) => {
      const nextFeature = nextFeatureMap.get(currentFeature.featureId);
      if (nextFeature) {
        // Find shared tokens
        const currentTokens = new Set(currentFeature.activations.map((a) => a.tokenIndex));
        const sharedTokens = nextFeature.activations
          .filter((a) => currentTokens.has(a.tokenIndex))
          .map((a) => a.tokenIndex);

        connections.push({
          featureId: currentFeature.featureId,
          fromLayer: currentLayer.layer,
          toLayer: nextLayer.layer,
          fromActivation: currentFeature.maxActivation,
          toActivation: nextFeature.maxActivation,
          sharedTokens,
        });
      }
    });
  }

  return connections;
}

/**
 * Get the position index of a feature within its layer (for layout purposes)
 */
export function getFeaturePositionInLayer(
  featureId: number,
  layerData: LayerFlowData
): number {
  return layerData.features.findIndex((f) => f.featureId === featureId);
}

/**
 * Compute layout positions for deduplicated features
 * Returns a map of featureId -> { x, y } positions
 */
export interface FeaturePosition {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface LayoutConfig {
  tokenColumnWidth: number;
  layerColumnWidth: number;
  featureHeight: number;
  featureGap: number;
  layerGap: number;
  headerHeight: number;
  padding: number;
}

export const DEFAULT_LAYOUT_CONFIG: LayoutConfig = {
  tokenColumnWidth: 120,
  layerColumnWidth: 180,
  featureHeight: 32,
  featureGap: 4,
  layerGap: 60,
  headerHeight: 40,
  padding: 16,
};

export function computeFeatureLayout(
  layerFlowData: LayerFlowData[],
  layerOrder: number[],
  config: LayoutConfig = DEFAULT_LAYOUT_CONFIG
): Map<string, FeaturePosition> {
  const positions = new Map<string, FeaturePosition>();

  layerOrder.forEach((layer, layerIndex) => {
    const layerData = layerFlowData.find((l) => l.layer === layer);
    if (!layerData) return;

    const x =
      config.padding +
      config.tokenColumnWidth +
      config.layerGap +
      layerIndex * (config.layerColumnWidth + config.layerGap);

    layerData.features.forEach((feature, featureIndex) => {
      const y =
        config.headerHeight +
        config.padding +
        featureIndex * (config.featureHeight + config.featureGap);

      const key = `${layer}-${feature.featureId}`;
      positions.set(key, {
        x,
        y,
        width: config.layerColumnWidth,
        height: config.featureHeight,
      });
    });
  });

  return positions;
}

/**
 * Compute token positions for layout
 */
export function computeTokenLayout(
  tokens: string[],
  config: LayoutConfig = DEFAULT_LAYOUT_CONFIG
): Map<number, FeaturePosition> {
  const positions = new Map<number, FeaturePosition>();

  tokens.forEach((_, tokenIndex) => {
    const y =
      config.headerHeight +
      config.padding +
      tokenIndex * (config.featureHeight + config.featureGap);

    positions.set(tokenIndex, {
      x: config.padding,
      y,
      width: config.tokenColumnWidth,
      height: config.featureHeight,
    });
  });

  return positions;
}
