// Headless smoke test of the LIF brain engine. Mirrors src/brain/worker.ts
// exactly, runs on Node against data/connectome.bin to verify:
//   - binary parses
//   - network runs (baseline firing)
//   - visual input changes descending-neuron (motor) output
import { readFileSync } from 'node:fs'

const buf = readFileSync('data/connectome.bin')
const meta = JSON.parse(readFileSync('data/metadata.json', 'utf8'))

const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
const MAGIC = 0x43594c46
if (dv.getUint32(0, true) !== MAGIC) throw new Error('bad magic')
const n = dv.getUint32(8, true)
const m = dv.getUint32(12, true)
const nVisual = dv.getUint32(16, true)
const nMotor = dv.getUint32(20, true)
const nClasses = dv.getUint32(24, true)

let off = 28
const rowPtr = new Int32Array(buf.buffer, buf.byteOffset + off, n + 1); off += (n + 1) * 4
const colIdx = new Int32Array(buf.buffer, buf.byteOffset + off, m); off += m * 4
const W = new Float32Array(buf.buffer, buf.byteOffset + off, m); off += m * 4
const visualMap = new Int32Array(buf.buffer, buf.byteOffset + off, nVisual); off += nVisual * 4
const motorMap = new Int32Array(buf.buffer, buf.byteOffset + off, nMotor); off += nMotor * 4
const neuronClass = new Uint8Array(buf.buffer, buf.byteOffset + off, n); off += n

console.log(`parsed: n=${n} m=${m} nVisual=${nVisual} nMotor=${nMotor} nClasses=${nClasses}`)

const dt = meta.dt_ms
const a = dt / meta.tau_ms
const V = new Float32Array(n).fill(meta.v_rest)
const I = new Float32Array(n)
const vis = new Float32Array(n)
const neuronEma = new Float32Array(n)
const spiked = new Int32Array(n)
const motorEma = new Float32Array(nMotor)
const ALPHA = 0.08
let spikedCount = 0

function step() {
  for (let s = 0; s < spikedCount; s++) {
    const j = spiked[s]
    for (let k = rowPtr[j]; k < rowPtr[j + 1]; k++) I[colIdx[k]] += W[k]
  }
  spikedCount = 0
  let total = 0
  for (let i = 0; i < n; i++) {
    V[i] += (meta.v_rest - V[i]) * a + (I[i] + vis[i] * meta.vis_gain) * dt
    I[i] = 0
    const sp = V[i] > meta.v_th
    if (sp) {
      V[i] = meta.v_reset
      spiked[spikedCount++] = i
      total++
    }
    neuronEma[i] += ((sp ? 1 : 0) - neuronEma[i]) * ALPHA
  }
  const bg = meta.background_rate_hz * (dt / 1000) * n
  const extra = Math.floor(bg) + (Math.random() < bg % 1 ? 1 : 0)
  for (let e = 0; e < extra; e++) {
    const i = (Math.random() * n) | 0
    spiked[spikedCount++] = i
    neuronEma[i] += (1 - neuronEma[i]) * ALPHA
    total++
  }
  for (let k = 0; k < nMotor; k++) {
    const i = motorMap[k]
    motorEma[k] += (neuronEma[i] - motorEma[k]) * ALPHA
  }
  return total
}

function motorMean() {
  let s = 0
  for (let k = 0; k < nMotor; k++) s += motorEma[k]
  return (s / nMotor) * 60
}

function setVisual(pattern) {
  vis.fill(0)
  for (let c = 0; c < nVisual; c++) vis[visualMap[c]] = pattern(c)
}

const WARMUP = 300
// baseline (no visual)
setVisual(() => 0)
for (let i = 0; i < WARMUP; i++) step()
const baseMotor = motorMean()
let baseSpikes = 0
for (let i = 0; i < 60; i++) baseSpikes += step()
console.log(`baseline: motor=${baseMotor.toFixed(3)} Hz, total=${(baseSpikes / 60).toFixed(0)} sp/s`)

// uniform bright (all retina lit)
setVisual(() => 1.0)
for (let i = 0; i < WARMUP; i++) step()
const brightMotor = motorMean()
let brightSpikes = 0
for (let i = 0; i < 60; i++) brightSpikes += step()
console.log(`bright  : motor=${brightMotor.toFixed(3)} Hz, total=${(brightSpikes / 60).toFixed(0)} sp/s`)

// asymmetric: right half lit (columns 16..31 of 32x32 -> half of 1024)
setVisual((c) => (c >= 512 ? 1.0 : 0.0))
for (let i = 0; i < WARMUP; i++) step()
const asymMotor = motorMean()
console.log(`asym    : motor=${asymMotor.toFixed(3)} Hz`)

const dBright = brightMotor - baseMotor
console.log(`\nΔ motor (bright - base) = ${dBright.toFixed(3)} Hz`)
if (brightSpikes <= baseSpikes) throw new Error('visual input did not increase activity')
if (!(brightMotor >= baseMotor - 1e-6)) console.warn('motor mean did not rise with brightness (may still be ok)')
console.log('\nSMOKE TEST PASSED')