// Are the strong-contrast neurons the visual INPUT neurons (trivial readout)
// or DOWNSTREAM neurons (real computation)? Exclude visualMap neurons and see
// if strong contrast survives downstream.
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

// set of visual input neuron indices
const isVisual = new Uint8Array(n)
for (let c = 0; c < nVisual; c++) isVisual[visualMap[c]] = 1
// set of motor (DN) neuron indices
const isMotor = new Uint8Array(n)
for (let k = 0; k < nMotor; k++) isMotor[motorMap[k]] = 1

const up = measure('up', 10)
const down = measure('down', 10)
const contrast = new Float32Array(n)
for (let i = 0; i < n; i++) contrast[i] = up[i] - down[i]

// bucket by (isVisual, isMotor)
function stats(pred, label) {
  let cnt = 0, nPos = 0, nNeg = 0, maxC = 0
  for (let i = 0; i < n; i++) {
    if (!pred(i)) continue
    cnt++
    if (contrast[i] > 0.2) nPos++
    if (contrast[i] < -0.2) nNeg++
    if (Math.abs(contrast[i]) > maxC) maxC = Math.abs(contrast[i])
  }
  console.log(`  ${label}: n=${cnt}  up-sel(>0.2)=${nPos}  down-sel(<-0.2)=${nNeg}  max|c|=${maxC.toFixed(2)}`)
}
console.log(`=== @10s contrast by neuron type ===`)
stats((i) => isVisual[i] === 1 && isMotor[i] === 0, 'visual-input only')
stats((i) => isVisual[i] === 0 && isMotor[i] === 1, 'motor(DN) only')
stats((i) => isVisual[i] === 0 && isMotor[i] === 0, 'downstream (neither)')
stats((i) => isVisual[i] === 1 && isMotor[i] === 1, 'visual AND motor')
stats(() => true, 'ALL')

// among downstream (neither visual nor motor), top contrast neurons
const downIdx = []
for (let i = 0; i < n; i++) if (isVisual[i] === 0 && isMotor[i] === 0) downIdx.push(i)
downIdx.sort((a, b) => contrast[b] - contrast[a])
console.log(`\ndownstream top 8 up-selective:`)
for (let k = 0; k < 8; k++) { const i = downIdx[k]; console.log(`  n=${i}  up=${up[i].toFixed(2)}  down=${down[i].toFixed(2)}  c=+${contrast[i].toFixed(3)}`) }
console.log(`downstream top 8 down-selective:`)
for (let k = downIdx.length - 8; k < downIdx.length; k++) { const i = downIdx[k]; console.log(`  n=${i}  up=${up[i].toFixed(2)}  down=${down[i].toFixed(2)}  c=${contrast[i].toFixed(3)}`) }
const dnPos = downIdx.filter((i) => contrast[i] > 0.2).length
const dnNeg = downIdx.filter((i) => contrast[i] < -0.2).length
console.log(`\ndownstream |c|>0.2: up-sel=${dnPos}  down-sel=${dnNeg}  (of ${downIdx.length} downstream neurons)`)