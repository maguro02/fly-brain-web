// Confirm the per-episode architecture: HARD RESET -> present cue -> deliberate
// T seconds -> read out argmax. Test readout times 10/20/30/40s for both states.
// Expect up->BUY, down->SELL once the (slow) down state has converged.
import { readFileSync } from 'node:fs'

const buf = readFileSync('data/connectome.bin')
const meta = JSON.parse(readFileSync('data/metadata.json', 'utf8'))
const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
const n = dv.getUint32(8, true); const m = dv.getUint32(12, true)
const nVisual = dv.getUint32(16, true); const nMotor = dv.getUint32(20, true)
const nDan = dv.getUint32(28, true); const nGaba = dv.getUint32(32, true); const nSoma = dv.getUint32(36, true)
let off = 40
const rowPtr = new Int32Array(buf.buffer, buf.byteOffset + off, n + 1); off += (n + 1) * 4
const colIdx = new Int32Array(buf.buffer, buf.byteOffset + off, m); off += m * 4
const W = new Float32Array(buf.buffer, buf.byteOffset + off, m); off += m * 4
const visualMap = new Int32Array(buf.buffer, buf.byteOffset + off, nVisual); off += nVisual * 4
const motorMap = new Int32Array(buf.buffer, buf.byteOffset + off, nMotor); off += nMotor * 4
off += nDan * 4 + nGaba * 4 + nSoma * 4 + 3 * nSoma * 4

function makeRng(seed) { let s = seed >>> 0; return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296 } }
const dt = meta.dt_ms; const a = dt / meta.tau_ms; const ALPHA = 0.08
const V = new Float32Array(n); const I = new Float32Array(n); const vis = new Float32Array(n)
const neuronEma = new Float32Array(n); const spiked = new Int32Array(n); let spikedCount = 0
function reset() { V.fill(meta.v_rest); I.fill(0); vis.fill(0); neuronEma.fill(0); spikedCount = 0 }

const RET = 32
const visPat = new Float32Array(nVisual)
function setLamp(state) {
  visPat.fill(0)
  if (state === 'up') { for (let y = 2; y < 10; y++) for (let x = 2; x < 12; x++) visPat[y * RET + x] = 1 }
  else { for (let y = 2; y < 10; y++) for (let x = 20; x < 30; x++) visPat[y * RET + x] = 1 }
  vis.fill(0)
  for (let c = 0; c < nVisual; c++) vis[visualMap[c]] = visPat[c]
}
function step(rnd) {
  for (let s = 0; s < spikedCount; s++) { const j = spiked[s]; for (let k = rowPtr[j]; k < rowPtr[j + 1]; k++) I[colIdx[k]] += W[k] }
  spikedCount = 0
  for (let i = 0; i < n; i++) {
    V[i] += (meta.v_rest - V[i]) * a + (I[i] + vis[i] * meta.vis_gain) * dt
    I[i] = 0
    const sp = V[i] > meta.v_th
    if (sp) { V[i] = meta.v_reset; spiked[spikedCount++] = i }
    neuronEma[i] += ((sp ? 1 : 0) - neuronEma[i]) * ALPHA
  }
  const bg = meta.background_rate_hz * (dt / 1000) * n
  const extra = Math.floor(bg) + (rnd() < bg % 1 ? 1 : 0)
  for (let e = 0; e < extra; e++) { const i = (rnd() * n) | 0; spiked[spikedCount++] = i; neuronEma[i] += (1 - neuronEma[i]) * ALPHA }
}
const groups = meta.motor_groups
function groupRates() {
  const out = new Float32Array(groups.length)
  for (let g = 0; g < groups.length; g++) { const mem = groups[g]; let s = 0; for (let k = 0; k < mem.length; k++) s += neuronEma[mem[k]]; out[g] = (s / mem.length) * 60 }
  return out
}
const names = ['BUY', 'SELL', 'HOLD']

const SEED = 0x12345678
const SAMPLES = 5 // average 5 independent episodes per (state, readout time)
for (const state of ['up', 'down']) {
  for (const secs of [2, 5, 10, 20, 30, 60]) {
    const steps = secs * 100
    let acc = [0, 0, 0]; let correct = 0
    for (let ep = 0; ep < SAMPLES; ep++) {
      const rnd = makeRng(SEED + ep)
      reset()
      setLamp(state)
      for (let i = 0; i < steps; i++) step(rnd)
      const r = groupRates()
      acc[0] += r[0]; acc[1] += r[1]; acc[2] += r[2]
      const am = r.indexOf(Math.max(r[0], r[1], r[2]))
      if ((state === 'up' && am === 0) || (state === 'down' && am === 1)) correct++
    }
    acc = acc.map((v) => v / SAMPLES)
    const am = acc.indexOf(Math.max(acc[0], acc[1], acc[2]))
    console.log(`${state} @${secs}s: [${acc.map((v) => v.toFixed(2)).join('  ')}]  argmax=${names[am]}  correct=${correct}/${SAMPLES}`)
  }
}