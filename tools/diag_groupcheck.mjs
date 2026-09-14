// Determine whether the DN contrast is seed-dependent (chaotic network) or
// whether the two diag scripts differ by something other than seed.
// Uses diag_contrast's exact per-DN measurement, then reports group means for
// the CURRENT metadata groups, under multiple seeds.
import { readFileSync } from 'node:fs'

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

function makeRng(seed) {
  let s = seed >>> 0
  return function () {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0
    return s / 4294967296
  }
}

const dt = meta.dt_ms
const a = dt / meta.tau_ms
const ALPHA = 0.08
const RET = 32
const nVisualPat = nVisual

function measure(seed, state, warmup, steps) {
  const rnd = makeRng(seed)
  const V = new Float32Array(n).fill(meta.v_rest)
  const I = new Float32Array(n)
  const vis = new Float32Array(n)
  const neuronEma = new Float32Array(n)
  const spiked = new Int32Array(n)
  let spikedCount = 0
  const visPat = new Float32Array(nVisualPat)
  function setLamp(st) {
    visPat.fill(0)
    if (st === 'up') {
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
      if (sp) { V[i] = meta.v_reset; spiked[spikedCount++] = i }
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

const names = ['BUY', 'SELL', 'HOLD']
const groups = meta.motor_groups
// rates[] is DN-local; groups hold full neuron indices -> build neuron->local
const neuronToLocal = new Int32Array(n).fill(-1)
for (let k = 0; k < nMotor; k++) neuronToLocal[motorMap[k]] = k
function groupMeans(rates) {
  const out = []
  for (let g = 0; g < groups.length; g++) {
    const mem = groups[g]
    let s = 0
    for (let k = 0; k < mem.length; k++) s += rates[neuronToLocal[mem[k]]]
    out.push(s / mem.length)
  }
  return out
}

const WARMUP = 300
const STEPS = 1200
const seeds = [0xabc, 0x12345678, 0xdeadbeef, 0x999999]
for (const seed of seeds) {
  const up = measure(seed, 'up', WARMUP, STEPS)
  const down = measure(seed, 'down', WARMUP, STEPS)
  const gu = groupMeans(up)
  const gd = groupMeans(down)
  console.log(`seed=0x${seed.toString(16)}:  up  [${gu.map((v) => v.toFixed(3)).join('  ')}]   down [${gd.map((v) => v.toFixed(3)).join('  ')}]`)
  console.log(`             up: BUY-SELL=${(gu[0] - gu[1]).toFixed(3)}  down: SELL-BUY=${(gd[1] - gd[0]).toFixed(3)}`)
}