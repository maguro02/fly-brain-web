// Parameter sweep: find LIF settings where descending-neuron (motor) mean
// firing rises with visual brightness (phototaxis signal). Reuses the binary.
import { readFileSync } from 'node:fs'

const buf = readFileSync('data/connectome.bin')
const meta = JSON.parse(readFileSync('data/metadata.json', 'utf8'))
const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
const n = dv.getUint32(8, true)
const m = dv.getUint32(12, true)
const nVisual = dv.getUint32(16, true)
const nMotor = dv.getUint32(20, true)
let off = 28
const rowPtr = new Int32Array(buf.buffer, buf.byteOffset + off, n + 1); off += (n + 1) * 4
const colIdx = new Int32Array(buf.buffer, buf.byteOffset + off, m); off += m * 4
const W0 = new Float32Array(buf.buffer, buf.byteOffset + off, m); off += m * 4
const visualMap = new Int32Array(buf.buffer, buf.byteOffset + off, nVisual); off += nVisual * 4
const motorMap = new Int32Array(buf.buffer, buf.byteOffset + off, nMotor); off += nMotor * 4

const dt = meta.dt_ms
const a = dt / meta.tau_ms
const V = new Float32Array(n)
const I = new Float32Array(n)
const vis = new Float32Array(n)
const neuronEma = new Float32Array(n)
const spiked = new Int32Array(n)
const motorEma = new Float32Array(nMotor)
const ALPHA = 0.08
let spikedCount = 0

let W = W0
let visGain = meta.vis_gain
let bgRate = meta.background_rate_hz

function step() {
  for (let s = 0; s < spikedCount; s++) {
    const j = spiked[s]
    for (let k = rowPtr[j]; k < rowPtr[j + 1]; k++) I[colIdx[k]] += W[k]
  }
  spikedCount = 0
  for (let i = 0; i < n; i++) {
    V[i] += (meta.v_rest - V[i]) * a + (I[i] + vis[i] * visGain) * dt
    I[i] = 0
    const sp = V[i] > meta.v_th
    if (sp) {
      V[i] = meta.v_reset
      spiked[spikedCount++] = i
    }
    neuronEma[i] += ((sp ? 1 : 0) - neuronEma[i]) * ALPHA
  }
  const bg = bgRate * (dt / 1000) * n
  const extra = Math.floor(bg) + (Math.random() < bg % 1 ? 1 : 0)
  for (let e = 0; e < extra; e++) {
    const i = (Math.random() * n) | 0
    spiked[spikedCount++] = i
    neuronEma[i] += (1 - neuronEma[i]) * ALPHA
  }
  for (let k = 0; k < nMotor; k++) {
    const i = motorMap[k]
    motorEma[k] += (neuronEma[i] - motorEma[k]) * ALPHA
  }
}

function motorMean() {
  let s = 0
  for (let k = 0; k < nMotor; k++) s += motorEma[k]
  return (s / nMotor) * 60
}

function runCondition(visFn, warmup, measure) {
  vis.fill(0)
  for (let c = 0; c < nVisual; c++) vis[visualMap[c]] = visFn(c)
  for (let i = 0; i < warmup; i++) step()
  let acc = 0
  for (let i = 0; i < measure; i++) {
    step()
    acc += motorMean()
  }
  return acc / measure
}

function trial(bg, vg, wscale, warmup = 200, measure = 40) {
  bgRate = bg
  visGain = vg
  W = wscale === 1 ? W0 : scale(W0, wscale)
  V.fill(meta.v_rest)
  I.fill(0)
  neuronEma.fill(0)
  motorEma.fill(0)
  const dark = runCondition(() => 0, warmup, measure)
  const bright = runCondition(() => 1, warmup, measure)
  return { dark, bright, delta: bright - dark }
}

const cache = new Map()
function scale(src, f) {
  const key = f
  if (!cache.has(key)) {
    const out = new Float32Array(src.length)
    for (let i = 0; i < src.length; i++) out[i] = src[i] * f
    cache.set(key, out)
  }
  return cache.get(key)
}

const results = []
for (const bg of [0.2, 0.5, 1.0, 2.0]) {
  for (const vg of [8, 20, 50]) {
    for (const ws of [0.02, 0.05, 0.1]) {
      const r = trial(bg, vg, ws)
      results.push({ bg, vg, ws, ...r })
      console.log(
        `bg=${bg} vg=${vg} ws=${ws}  dark=${r.dark.toFixed(2)} bright=${r.bright.toFixed(2)} Δ=${r.delta.toFixed(2)}`,
      )
    }
  }
}
results.sort((x, y) => y.delta - x.delta)
console.log('\nTOP 5 by Δ motor:')
for (const r of results.slice(0, 5))
  console.log(`  bg=${r.bg} vg=${r.vg} ws=${r.ws}  Δ=${r.delta.toFixed(2)} (dark=${r.dark.toFixed(2)} bright=${r.bright.toFixed(2)})`)