import { scaleSequential, scaleDiverging } from "d3-scale";
import { interpolateViridis, interpolateRdBu } from "d3-scale-chromatic";

export function createActivationScale(maxActivation: number) {
  return scaleSequential(interpolateViridis).domain([0, maxActivation]);
}

export function createSteeringScale() {
  return scaleDiverging(interpolateRdBu).domain([-2, 0, 2]);
}

export function getActivationColor(
  activation: number,
  maxActivation: number
): string {
  const scale = createActivationScale(maxActivation);
  return scale(activation);
}

export function getContrastColor(backgroundColor: string): string {
  // Simple contrast calculation - extract RGB and compute luminance
  const hex = backgroundColor.replace("#", "");
  if (hex.length !== 6) return "#ffffff";

  const r = parseInt(hex.slice(0, 2), 16);
  const g = parseInt(hex.slice(2, 4), 16);
  const b = parseInt(hex.slice(4, 6), 16);

  // Relative luminance formula
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;

  return luminance > 0.5 ? "#000000" : "#ffffff";
}

export function activationToOpacity(
  activation: number,
  maxActivation: number
): number {
  if (maxActivation === 0) return 0;
  return Math.min(activation / maxActivation, 1);
}
