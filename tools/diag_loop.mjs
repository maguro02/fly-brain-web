// Mirror learn_test.mjs EXACTLY (same seed, same loop, same bursts, no
// plasticity) but log the per-decision group rates, aggregated by state.
// Purpose: resolve the contradiction between the steady-state group
// measurement (up->BUY, down->SELL) and the task-loop argmax (SELL in both).
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
const danMap = new Int32Array(buf.buffer, buf.byteOffset + off, nDan); off += nDan * 4
const gabaMap = new Int32Array(buf.buffer, buf.byteOffset + off, nGaba); off += nGaba * 4
const somaIdx = new Int32Array(buf.buffer, buf.byteOffset + off, nSoma); off += nSoma * 4
off += 3 * nSoma * 4

let seed = 0x12345678
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
const BURST_STEPS = 3
const NO_BURSTS = process.argv[2] === 'noBursts'
let rewardBurst = 0
let punishBurst = 0

function fireBurst(map, count) {
  for (let t = 0; t < count; t++) {
    const i = map[t]
    V[i] = meta.v_reset
    spiked[spikedCount++] = i
    neuronEma[i] += (1 - neuronEma[i]) * ALPHA
  }
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
  if (NO_BURSTS) { rewardBurst = 0; punishBurst = 0 }
  if (rewardBurst > 0) { rewardBurst--; fireBurst(danMap, nDan) }
  if (punishBurst > 0) { punishBurst--; fireBurst(gabaMap, nGaba) }
}

const RET = 32
const visPat = new Float32Array(nVisual)
function applyVisual() {
  vis.fill(0)
  for (let c = 0; c < nVisual; c++) vis[visualMap[c]] = visPat[c]
}
function setLamp(state) {
  visPat.fill(0)
  if (state === 'up') {
    for (let y = 2; y < 10; y++) for (let x = 2; x < 12; x++) visPat[y * RET + x] = 1
  } else {
    for (let y = 2; y < 10; y++) for (let x = 20; x < 30; x++) visPat[y * RET + x] = 1
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

let state = 'up'
let stateLeft = 600
function nextDuration() { return 480 + Math.floor(rnd() * 420) }

const EPOCHS = 400
const DECISION_EVERY = 60

setLamp(state)
applyVisual()
for (let i = 0; i < 300; i++) step()

const names = ['BUY', 'SELL', 'HOLD']
const acc = { up: [0, 0, 0], down: [0, 0, 0] }
const tot = { up: 0, down: 0 }
const cnt = { up: 0, down: 0 }
const argmax = { up: [0, 0, 0], down: [0, 0, 0] }

for (let epoch = 0; epoch < EPOCHS; epoch++) {
  for (let s = 0; s < DECISION_EVERY; s++) {
    stateLeft--
    if (stateLeft <= 0) {
      state = state === 'up' ? 'down' : 'up'
      stateLeft = nextDuration()
    }
    setLamp(state)
    applyVisual()
    step()
  }
  const rates = groupRates()
  const eps = Math.max(0.1, 0.8 - 0.7 * (epoch / 100))
  let choice
  let isArgmax = false
  if (rnd() < eps) choice = (rnd() * 3) | 0
  else {
    choice = rates.indexOf(Math.max(rates[0], rates[1], rates[2]))
    isArgmax = true
  }
  if (isArgmax) {
    cnt[state]++
    argmax[state][choice]++
  }
  let outcome
  if (choice === 2) outcome = 'hold'
  else if ((choice === 0 && state === 'up') || (choice === 1 && state === 'down')) outcome = 'correct'
  else outcome = 'wrong'
  if (outcome === 'correct') rewardBurst = BURST_STEPS
  else if (outcome === 'wrong') punishBurst = BURST_STEPS

  for (let g = 0; g < 3; g++) acc[state][g] += rates[g]
  tot[state]++
}

for (const st of ['up', 'down']) {
  const mean = acc[st].map((v) => v / tot[st])
  console.log(`state=${st} (n=${tot[st]}):`)
  for (let g = 0; g < 3; g++) console.log(`  ${names[g]} mean=${mean[g].toFixed(3)} Hz`)
  const [b, s, h] = argmax[st]
  console.log(`  argmax (n=${cnt[st]}): BUY=${(100 * b / cnt[st]).toFixed(0)}% SELL=${(100 * s / cnt[st]).toFixed(0)}% HOLD=${(100 * h / cnt[st]).toFixed(0)}%`)
}