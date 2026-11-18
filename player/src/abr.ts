// Adaptive bitrate: the same rules as server/abr.go (shared test vectors in
// server/testdata/abr_vectors.json). Bitrates in kbps, buffer in seconds.

export interface AbrState {
  buffer: number;
  throughput: number[]; // measured kbps of recent segments, oldest first
  last: number; // rendition of the previous segment, -1 at start
}

export function fit(kbps: number[], budget: number): number {
  let best = 0;
  kbps.forEach((k, i) => {
    if (k <= budget) best = i;
  });
  return best;
}

export function harmonic(xs: number[], n: number): number {
  if (xs.length === 0) return 0;
  const tail = xs.slice(-n);
  return tail.length / tail.reduce((s, x) => s + 1 / Math.max(x, 1), 0);
}

export function throughputRule(st: AbrState, kbps: number[]): number {
  if (st.throughput.length === 0) return 0;
  return fit(kbps, 0.8 * harmonic(st.throughput, 5));
}

export function bufferRule(st: AbrState, kbps: number[]): number {
  const reservoir = 5, cushion = 20;
  if (st.buffer <= reservoir) return 0;
  if (st.buffer >= reservoir + cushion) return kbps.length - 1;
  const budget = kbps[0] + ((kbps[kbps.length - 1] - kbps[0]) * (st.buffer - reservoir)) / cushion;
  return fit(kbps, budget);
}

/** the player's rule: throughput with a buffer-dependent safety factor, one step up at a time */
export function hybridRule(st: AbrState, kbps: number[]): number {
  if (st.throughput.length === 0 || st.buffer < 2) return 0;
  const safety = st.buffer > 10 ? 0.9 : 0.7;
  let choice = fit(kbps, safety * harmonic(st.throughput, 4));
  if (st.last >= 0 && choice > st.last + 1) choice = st.last + 1;
  return choice;
}

export const rules = { throughput: throughputRule, buffer: bufferRule, hybrid: hybridRule };
