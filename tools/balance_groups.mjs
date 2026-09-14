// Balance the 3 descending-neuron motor groups under task visual conditions.
// Contiguous ID slices have a ~4 Hz structural rate gap, and the cue-lamp
// visual drive adds a common-mode bias (one group always wins). We measure
// per-DN rates with the left lamp (up) and right lamp (down) on, and LPT
// partition on the average of the two rate vectors: the common visual drive
// cancels, leaving only the state-differential signal for the readout.
// Writes data/motor_groups.json consumed by tools/build_connectome.py.
import { readFileSync, writeFileSync } from 'node:fs'

const buf = readFileSync('data/connectome.bin')
const meta = JSON.parse(readFileSync('data/metadata.json', 'utf8'))

const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
const n = dv.getUint32(8, true)
const m = dv.getUint32(12, true)
const nVisual = dv.getUint32(16, true)
const nMotor = dv.getUint32(20, true)
const nClasses = dv.getUint32(24, true)
const nDan = dv.getUint32(28, true)
const nGaba = dv.getUint32(32, true)
const nSoma = dv.getUint32(36, true)
let off = 40
const rowPtr = new Int32Array(buf.buffer, buf.byteOffset + off, n + 1); off += (n + 1) * 4
const colIdx = new Int32Array(buf.buffer, buf.byteOffset + off, m); off += m * 4
const W = new Float32Array(buf.buffer, buf.byteOffset + off, m); off += m * 4
const visualMap = new Int32Array(buf.buffer, buf.byteOffset + off, nVisual); off += nVisual * 4
const motorMap = new Int32Array(buf.buffer, buf.byteOffset + off, nMotor); off += nMotor * 4
const danMap = new Int32Array(buf.buffer, buf.byteOffset + off, nDan); off += nDan * 4
const gabaMap = new Int32Array(buf.buffer, buf.byteOffset + off, nGaba); off += nGaba * 4
const somaIdx = new Int32Array(buf.buffer, buf.byteOffset + off, nSoma); off += nSoma * 4
off += 3 * nSoma * 4
const neuronClass = new Uint8Array(buf.buffer, buf.byteOffset + off, n)

let seed = 0xdeadbeef
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
  } else {
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

function measure(state, steps) {
  setLamp(state)
  const acc = new Float32Array(nMotor)
  for (let i = 0; i < steps; i++) {
    step()
    for (let k = 0; k < nMotor; k++) acc[k] += neuronEma[motorMap[k]]
  }
  const rates = new Float32Array(nMotor)
  for (let k = 0; k < nMotor; k++) rates[k] = (acc[k] / steps) * 60
  return rates
}

const WARMUP = 300
const MEASURE = 1500
setLamp('up')
for (let i = 0; i < WARMUP; i++) step()
const rateUp = measure('up', MEASURE)
const rateDown = measure('down', MEASURE)

const balance = new Float32Array(nMotor)
for (let k = 0; k < nMotor; k++) balance[k] = (rateUp[k] + rateDown[k]) / 2

// LPT partition into 3 groups
const order = Array.from({ length: nMotor }, (_, k) => k).sort((x, y) => balance[y] - balance[x])
const groups = [[], [], []]
const sums = [0, 0, 0]
for (const k of order) {
  let g = 0
  if (sums[1] < sums[0]) g = 1
  if (sums[2] < sums[g]) g = 2
  groups[g].push(motorMap[k])
  sums[g] += balance[k]
}
const meansB = groups.map((g, gi) => sums[gi] / g.length)
const meansUp = groups.map((g) => g.reduce((s, i) => s + rateUp[motorMap.indexOf(i)], 0) / g.length)
const meansDown = groups.map((g) => g.reduce((s, i) => s + rateDown[motorMap.indexOf(i)], 0) / g.length)
console.log('group sizes:', groups.map((g) => g.length).join(', '))
console.log('balance means (Hz):', meansB.map((v) => v.toFixed(3)).join(', '))
console.log('up-state means (Hz):', meansUp.map((v) => v.toFixed(3)).join(', '))
console.log('down-state means (Hz):', meansDown.map((v) => v.toFixed(3)).join(', '))
console.log('residual spread up:', (Math.max(...meansUp) - Math.min(...meansUp)).toFixed(3), 'Hz')
console.log('residual spread down:', (Math.max(...meansDown) - Math.min(...meansDown)).toFixed(3), 'Hz')

const out = {
  motor_groups: groups.map((g) => g.map((i) => Number(i))),
  baseline_rates_hz: meansB.map((v) => Number(v.toFixed(4))),
}
writeFileSync('data/motor_groups.json', JSON.stringify(out, null, 2))
console.log('wrote data/motor_groups.json')