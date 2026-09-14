// Hidden-state market: 2 states (up/down) switching every 8..15 s. The cue is
// a 32x32 visual pattern (left block = up, right block = down) that matches
// the two lamps on the 3D monitor.

export const RETINA = 32
export const RETINA_COLS = RETINA * RETINA

export type MarketState = 'up' | 'down'
export type Choice = 0 | 1 | 2
export type Outcome = 'correct' | 'wrong' | 'hold'

export const CHOICE_NAMES = ['BUY', 'SELL', 'HOLD'] as const

export function cuePattern(state: MarketState): Float32Array {
  const p = new Float32Array(RETINA_COLS)
  if (state === 'up') {
    for (let y = 2; y < 10; y++) for (let x = 2; x < 12; x++) p[y * RETINA + x] = 1
  } else {
    for (let y = 2; y < 10; y++) for (let x = 20; x < 30; x++) p[y * RETINA + x] = 1
  }
  return p
}

// fixed exploration rate: annealing would confound the learning curve, so the
// plasticity on/off gap must come from weight changes alone
export const EPSILON = 0.2