// Measure steady-state group rates in three static conditions: dark, up-lamp,
// down-lamp. Separates the intrinsic (dark) group bias from the visual
// contribution and the state signal (up - down).
import { readFileSync } from 'node:fs'

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

const groups = meta.motor_groups
function groupRates() {
  const out = new Float32Array(groups.length)
  for (let g = 0; g < groups.length; g++) {
    const mem = groups[g]
    let s = 0
    for (let k = 0; k < mem.length; k++) s += neuronEma[mem[k]]
    out[g] = (s / mem.length) * 60
  }
  return out
}

function measure(state, warmup, steps) {
  setLamp(state)
  for (let i = 0; i < warmup; i++) step()
  const acc = [0, 0, 0]
  for (let i = 0; i < steps; i++) {
    step()
    const r = groupRates()
    acc[0] += r[0]; acc[1] += r[1]; acc[2] += r[2]
  }
  return [acc[0] / steps, acc[1] / steps, acc[2] / steps]
}

const WARMUP = 300
const STEPS = 1200
const dark = measure(null, WARMUP, STEPS)
const up = measure('up', 300, STEPS)
const down = measure('down', 300, STEPS)

console.log('group rates (Hz) [BUY, SELL, HOLD]:')
console.log(`  dark : ${dark.map((v) => v.toFixed(3)).join('  ')}`)
console.log(`  up   : ${up.map((v) => v.toFixed(3)).join('  ')}`)
console.log(`  down : ${down.map((v) => v.toFixed(3)).join('  ')}`)
console.log('visual contribution (lamp - dark):')
console.log(`  up-d   : ${up.map((v, i) => (v - dark[i]).toFixed(3)).join('  ')}`)
console.log(`  down-d : ${down.map((v, i) => (v - dark[i]).toFixed(3)).join('  ')}`)
console.log('state signal (up - down):')
console.log(`  up-down: ${up.map((v, i) => (v - down[i]).toFixed(3)).join('  ')}`)
console.log('intrinsic bias spread (dark max-min):', (Math.max(...dark) - Math.min(...dark)).toFixed(3), 'Hz')
console.log('state signal spread (up-down max-min):', (Math.max(...up.map((v, i) => v - down[i])) - Math.min(...up.map((v, i) => v - down[i]))).toFixed(3), 'Hz')