// Does the cue information exist ANYWHERE in the network?
// Measure per-neuron contrast (rateUp - rateDown) from CLEAN resets for all
// 211,577 neurons, at two readout times (5s, 10s). If any neurons carry a
// robust contrast, the fix is "read out from those neurons". If none do, the
// cue signal is not usable in this recurrent network and we need a redesign.
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

const SEED = 0x12345678
const SAMPLES = 3
function measure(state, secs) {
  const steps = secs * 100
  const acc = new Float32Array(n)
  for (let ep = 0; ep < SAMPLES; ep++) {
    const rnd = makeRng(SEED + ep)
    reset()
    setLamp(state)
    for (let i = 0; i < steps; i++) step(rnd)
    for (let i = 0; i < n; i++) acc[i] += neuronEma[i]
  }
  for (let i = 0; i < n; i++) acc[i] = (acc[i] / SAMPLES) * 60
  return acc
}

for (const secs of [5, 10]) {
  const up = measure('up', secs)
  const down = measure('down', secs)
  const contrast = new Float32Array(n)
  for (let i = 0; i < n; i++) contrast[i] = up[i] - down[i]
  const sorted = Array.from({ length: n }, (_, i) => i).sort((a, b) => contrast[b] - contrast[a])
  function pct(p) { const i = Math.min(n - 1, Math.max(0, Math.round(p * n))); return contrast[sorted[i]] }
  console.log(`\n=== all-neuron contrast @${secs}s (rateUp - rateDown), n=${n} ===`)
  console.log(`  max=+${contrast[sorted[0]].toFixed(3)}  p99=+${pct(0.01).toFixed(3)}  p75=+${pct(0.25).toFixed(3)}  median=${pct(0.5).toFixed(3)}  p25=${pct(0.75).toFixed(3)}  p01=${pct(0.99).toFixed(3)}  min=${contrast[sorted[n - 1]].toFixed(3)}`)
  const nPos = contrast.filter((c) => c > 0.2).length
  const nNeg = contrast.filter((c) => c < -0.2).length
  console.log(`  |c|>0.2: up-selective=${nPos}  down-selective=${nNeg}`)
  console.log('  top 8 up-selective neurons:')
  for (let k = 0; k < 8; k++) { const i = sorted[k]; console.log(`    n=${i}  up=${up[i].toFixed(2)}  down=${down[i].toFixed(2)}  c=+${contrast[i].toFixed(3)}`) }
  console.log('  top 8 down-selective neurons:')
  for (let k = n - 8; k < n; k++) { const i = sorted[k]; console.log(`    n=${i}  up=${up[i].toFixed(2)}  down=${down[i].toFixed(2)}  c=${contrast[i].toFixed(3)}`) }
}