// Build CONTRAST-RANKED downstream action groups (the working readout).
// BUY = top up-selective downstream neurons, SELL = top down-selective,
// HOLD = neutral. This gives a state-aligned readout (up->BUY, down->SELL).
// Measures per-neuron contrast from clean resets, then assigns groups.
import { readFileSync, writeFileSync } from 'node:fs'

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

console.log('measuring per-neuron contrast (10s from clean reset)...')
const up = measure('up', 10)
const down = measure('down', 10)
const contrast = new Float32Array(n)
for (let i = 0; i < n; i++) contrast[i] = up[i] - down[i]

// exclude visual + motor
const excluded = new Uint8Array(n)
for (let c = 0; c < nVisual; c++) excluded[visualMap[c]] = 1
for (let k = 0; k < nMotor; k++) excluded[motorMap[k]] = 1

const pool = []
for (let i = 0; i < n; i++) if (!excluded[i]) pool.push(i)
pool.sort((a, b) => contrast[b] - contrast[a]) // descending contrast

const G = 438
const buy = pool.slice(0, G)          // top up-selective
const sell = pool.slice(pool.length - G, pool.length) // top down-selective
// HOLD: neutral (contrast closest to 0) — pick from the middle, excluding buy/sell
const buySet = new Set(buy), sellSet = new Set(sell)
const neutral = pool.filter((i) => !buySet.has(i) && !sellSet.has(i) && Math.abs(contrast[i]) < 0.5)
neutral.sort((a, b) => Math.abs(contrast[a]) - Math.abs(contrast[b]))
const hold = neutral.slice(0, G)

// sanity: disjoint
const seen = new Set()
for (const idx of [...buy, ...sell, ...hold]) {
  if (seen.has(idx)) throw new Error(`duplicate ${idx}`)
  seen.add(idx)
}
console.log(`BUY mean contrast=+${(buy.reduce((s, i) => s + contrast[i], 0) / G).toFixed(3)}  (up=${(buy.reduce((s, i) => s + up[i], 0) / G).toFixed(1)} down=${(buy.reduce((s, i) => s + down[i], 0) / G).toFixed(1)})`)
console.log(`SELL mean contrast=${(sell.reduce((s, i) => s + contrast[i], 0) / G).toFixed(3)}  (up=${(sell.reduce((s, i) => s + up[i], 0) / G).toFixed(1)} down=${(sell.reduce((s, i) => s + down[i], 0) / G).toFixed(1)})`)
console.log(`HOLD mean contrast=${(hold.reduce((s, i) => s + contrast[i], 0) / G).toFixed(3)}  (up=${(hold.reduce((s, i) => s + up[i], 0) / G).toFixed(1)} down=${(hold.reduce((s, i) => s + down[i], 0) / G).toFixed(1)})`)

writeFileSync('data/motor_groups.json', JSON.stringify({ groups: [buy, sell, hold], note: 'contrast-ranked downstream action groups' }))
meta.motor_groups = [buy, sell, hold]
writeFileSync('data/metadata.json', JSON.stringify(meta))
console.log('wrote data/motor_groups.json + patched data/metadata.json')