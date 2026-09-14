// Per-DN left/right contrast diagnostic.
// Measures each of the 1314 motor (DN) neurons' steady-state rate under
// up-lamp and down-lamp, then reports the contrast distribution
// (rateUp - rateDown) to see whether the DN layer carries usable
// left/right information, and which neurons carry it.
import { readFileSync, writeFileSync } from 'node:fs'

const buf = readFileSync('data/connectome.bin')
const meta = JSON.parse(readFileSync('data/metadata.json', 'utf8'))

const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
const n = dv.getUint32(8, true)
const m = dv.getUint32(12, true)
const nVisual = dv.getUint32(16, true)
const nMotor = dv.getUint32(20, true)
const nDan = dv.getUint32(28, true)
const nGaba = dv.getUint32(32, true)
const nSoma = dv.getUint32(36, true)
let off = 40
const rowPtr = new Int32Array(buf.buffer, buf.byteOffset + off, n + 1); off += (n + 1) * 4
const colIdx = new Int32Array(buf.buffer, buf.byteOffset + off, m); off += m * 4
const W = new Float32Array(buf.buffer, buf.byteOffset + off, m); off += m * 4
const visualMap = new Int32Array(buf.buffer, buf.byteOffset + off, nVisual); off += nVisual * 4
const motorMap = new Int32Array(buf.buffer, buf.byteOffset + off, nMotor); off += nMotor * 4
off += nDan * 4 + nGaba * 4 + nSoma * 4 + 3 * nSoma * 4

let seed = 0xabc
function rnd() {
  seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0
  return seed / 4294967296
}

const dt = meta.dt_ms
const a = dt / meta.tau_ms
const V = new Float32Array(n).fill(meta.v_rest)
const I = new Float32Array(n)
const vis = new Float32Array(n)
const neuronEma = new Float32Array(n)
const spiked = new Int32Array(n)
let spikedCount = 0
const ALPHA = 0.08

const RET = 32
const visPat = new Float32Array(nVisual)
function setLamp(state) {
  visPat.fill(0)
  if (state === 'up') {
    for (let y = 2; y < 10; y++) for (let x = 2; x < 12; x++) visPat[y * RET + x] = 1
  } else if (state === 'down') {
    for (let y = 2; y < 10; y++) for (let x = 20; x < 30; x++) visPat[y * RET + x] = 1
  }
  vis.fill(0)
  for (let c = 0; c < nVisual; c++) vis[visualMap[c]] = visPat[c]
}

function step() {
  for (let s = 0; s < spikedCount; s++) {
    const j = spiked[s]
    for (let k = rowPtr[j]; k < rowPtr[j + 1]; k++) I[colIdx[k]] += W[k]
  }
  spikedCount = 0
  for (let i = 0; i < n; i++) {
    V[i] += (meta.v_rest - V[i]) * a + (I[i] + vis[i] * meta.vis_gain) * dt
    I[i] = 0
    const sp = V[i] > meta.v_th
    if (sp) {
      V[i] = meta.v_reset
      spiked[spikedCount++] = i
    }
    neuronEma[i] += ((sp ? 1 : 0) - neuronEma[i]) * ALPHA
  }
  const bg = meta.background_rate_hz * (dt / 1000) * n
  const extra = Math.floor(bg) + (rnd() < bg % 1 ? 1 : 0)
  for (let e = 0; e < extra; e++) {
    const i = (rnd() * n) | 0
    spiked[spikedCount++] = i
    neuronEma[i] += (1 - neuronEma[i]) * ALPHA
  }
}

function measureDNs(state, warmup, steps) {
  setLamp(state)
  for (let i = 0; i < warmup; i++) step()
  const acc = new Float32Array(nMotor)
  for (let i = 0; i < steps; i++) {
    step()
    for (let k = 0; k < nMotor; k++) acc[k] += neuronEma[motorMap[k]]
  }
  for (let k = 0; k < nMotor; k++) acc[k] = (acc[k] / steps) * 60
  return acc
}

const WARMUP = 300
const STEPS = 1200
const up = measureDNs('up', WARMUP, STEPS)
const down = measureDNs('down', 300, STEPS)

const contrast = new Float32Array(nMotor)
for (let k = 0; k < nMotor; k++) contrast[k] = up[k] - down[k]

const sorted = Array.from({ length: nMotor }, (_, k) => k).sort((a, b) => contrast[b] - contrast[a])
const cSorted = sorted.map((k) => contrast[k])

function pct(p) {
  const i = Math.min(nMotor - 1, Math.max(0, Math.round(p * nMotor)))
  return cSorted[i]
}

console.log(`per-DN contrast (rateUp - rateDown), n=${nMotor}`)
console.log(`  max     = ${cSorted[0].toFixed(3)} Hz (dn=${sorted[0]})`)
console.log(`  p99     = ${pct(0.01).toFixed(3)}`)
console.log(`  p95     = ${pct(0.05).toFixed(3)}`)
console.log(`  p75     = ${pct(0.25).toFixed(3)}`)
console.log(`  median  = ${pct(0.5).toFixed(3)}`)
console.log(`  p25     = ${pct(0.75).toFixed(3)}`)
console.log(`  p05     = ${pct(0.95).toFixed(3)}`)
console.log(`  min     = ${cSorted[nMotor - 1].toFixed(3)} Hz (dn=${sorted[nMotor - 1]})`)

const nPos = cSorted.filter((c) => c > 0.05).length
const nNeg = cSorted.filter((c) => c < -0.05).length
const nStrongPos = cSorted.filter((c) => c > 0.2).length
const nStrongNeg = cSorted.filter((c) => c < -0.2).length
console.log(`  |c|>0.05: pos=${nPos} neg=${nNeg}`)
console.log(`  |c|>0.20: pos=${nStrongPos} neg=${nStrongNeg}`)

// group means if we picked top/bottom 438 by contrast
const G = 438
let buyMean = 0, sellMean = 0, holdMean = 0
for (let k = 0; k < G; k++) buyMean += cSorted[k]
for (let k = nMotor - G; k < nMotor; k++) sellMean += cSorted[k]
for (let k = G; k < nMotor - G; k++) holdMean += cSorted[k]
buyMean /= G; sellMean /= G; holdMean /= G
console.log(`contrast-ranked groups (top/bottom ${G}):`)
console.log(`  BUY mean contrast  = ${buyMean.toFixed(3)} Hz`)
console.log(`  HOLD mean contrast = ${holdMean.toFixed(3)} Hz`)
console.log(`  SELL mean contrast = ${sellMean.toFixed(3)} Hz`)

// top 20 most up-selective and down-selective DNs
console.log('top 10 up-selective DNs:')
for (let k = 0; k < 10; k++) console.log(`  dn=${sorted[k]} contrast=${cSorted[k].toFixed(3)} up=${up[sorted[k]].toFixed(2)} down=${down[sorted[k]].toFixed(2)}`)
console.log('top 10 down-selective DNs:')
for (let k = nMotor - 10; k < nMotor; k++) console.log(`  dn=${sorted[k]} contrast=${cSorted[k].toFixed(3)} up=${up[sorted[k]].toFixed(2)} down=${down[sorted[k]].toFixed(2)}`)

writeFileSync('/tmp/opencode/dn_contrast.json', JSON.stringify({ up: Array.from(up), down: Array.from(down), contrast: Array.from(contrast) }))
console.log('wrote /tmp/opencode/dn_contrast.json')