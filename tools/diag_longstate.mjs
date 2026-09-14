// Confirm the fix: with LONGER state durations (30s) the network converges and
// the readout becomes state-aligned (up->BUY, down->SELL). Bins group rates by
// time-since-switch to show the convergence trajectory inside the task loop.
// No bursts, no plasticity, seed 0x12345678 (the learn_test seed).
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
const Wraw = new Float32Array(buf.buffer, buf.byteOffset + off, m); off += m * 4
const visualMap = new Int32Array(buf.buffer, buf.byteOffset + off, nVisual); off += nVisual * 4
const motorMap = new Int32Array(buf.buffer, buf.byteOffset + off, nMotor); off += nMotor * 4
off += nDan * 4 + nGaba * 4 + nSoma * 4 + 3 * nSoma * 4

let seed = 0x12345678
function rnd() { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296 }

const VIS_GAIN = parseFloat(process.argv[2] || String(meta.vis_gain))
const STATE_LEN_ARG = parseInt(process.argv[3] || '3000', 10)
const WSCALE = parseFloat(process.argv[5] || '1')
const W = WSCALE === 1 ? Wraw : (() => { const w = new Float32Array(m); for (let i = 0; i < m; i++) w[i] = Wraw[i] * WSCALE; return w })()
const dt = meta.dt_ms; const a = dt / meta.tau_ms; const ALPHA = 0.08
const V = new Float32Array(n).fill(meta.v_rest)
const I = new Float32Array(n)
const vis = new Float32Array(n)
const neuronEma = new Float32Array(n)
const spiked = new Int32Array(n)
let spikedCount = 0

function step() {
  for (let s = 0; s < spikedCount; s++) { const j = spiked[s]; for (let k = rowPtr[j]; k < rowPtr[j + 1]; k++) I[colIdx[k]] += W[k] }
  spikedCount = 0
  for (let i = 0; i < n; i++) {
    V[i] += (meta.v_rest - V[i]) * a + (I[i] + vis[i] * VIS_GAIN) * dt
    I[i] = 0
    const sp = V[i] > meta.v_th
    if (sp) { V[i] = meta.v_reset; spiked[spikedCount++] = i }
    neuronEma[i] += ((sp ? 1 : 0) - neuronEma[i]) * ALPHA
  }
  const bg = meta.background_rate_hz * (dt / 1000) * n
  const extra = Math.floor(bg) + (rnd() < bg % 1 ? 1 : 0)
  for (let e = 0; e < extra; e++) { const i = (rnd() * n) | 0; spiked[spikedCount++] = i; neuronEma[i] += (1 - neuronEma[i]) * ALPHA }
}

const RET = 32
const visPat = new Float32Array(nVisual)
function applyVisual() { vis.fill(0); for (let c = 0; c < nVisual; c++) vis[visualMap[c]] = visPat[c] }
function setLamp(state) {
  visPat.fill(0)
  if (state === 'up') { for (let y = 2; y < 10; y++) for (let x = 2; x < 12; x++) visPat[y * RET + x] = 1 }
  else { for (let y = 2; y < 10; y++) for (let x = 20; x < 30; x++) visPat[y * RET + x] = 1 }
}

const groups = meta.motor_groups
function groupRates() {
  const out = new Float32Array(groups.length)
  for (let g = 0; g < groups.length; g++) {
    const mem = groups[g]; let s = 0
    for (let k = 0; k < mem.length; k++) s += neuronEma[mem[k]]
    out[g] = (s / mem.length) * 60
  }
  return out
}
const names = ['BUY', 'SELL', 'HOLD']
console.log(`=== vis_gain=${VIS_GAIN}  state_len=${STATE_LEN_ARG} (${(STATE_LEN_ARG * dt / 1000).toFixed(0)}s)  wscale=${WSCALE} ===`)

const STATE_LEN = STATE_LEN_ARG
const DECISION_EVERY = 60
let state = 'up'
let stateLeft = STATE_LEN
let tss = 0
setLamp(state); applyVisual()

const binEdges = [0, 300, 600, 1200, 1800, 2400, 3000]
const binAcc = { up: binEdges.slice(0, -1).map(() => [0, 0, 0]), down: binEdges.slice(0, -1).map(() => [0, 0, 0]) }
const binCnt = { up: binEdges.slice(0, -1).map(() => 0), down: binEdges.slice(0, -1).map(() => 0) }
const argmax = { up: [0, 0, 0], down: [0, 0, 0] }
const amCnt = { up: 0, down: 0 }

const N_STATES = parseInt(process.argv[4] || '8', 10)
for (let st = 0; st < N_STATES; st++) {
  for (let s = 0; s < STATE_LEN; s += DECISION_EVERY) {
    for (let d = 0; d < DECISION_EVERY; d++) {
      stateLeft--
      if (stateLeft <= 0) { state = state === 'up' ? 'down' : 'up'; stateLeft = STATE_LEN; tss = 0 }
      setLamp(state); applyVisual(); step(); tss++
    }
    const rates = groupRates()
    let b = 0
    while (b < binEdges.length - 2 && tss >= binEdges[b + 1]) b++
    for (let g = 0; g < 3; g++) binAcc[state][b][g] += rates[g]
    binCnt[state][b]++
    const am = rates.indexOf(Math.max(rates[0], rates[1], rates[2]))
    argmax[state][am]++; amCnt[state]++
  }
}

for (const st of ['up', 'down']) {
  console.log(`state=${st}:`)
  for (let b = 0; b < binCnt[st].length; b++) {
    const c = binCnt[st][b]
    if (c === 0) continue
    const mean = binAcc[st][b].map((v) => v / c)
    const am = mean.indexOf(Math.max(mean[0], mean[1], mean[2]))
    console.log(`  tss [${binEdges[b]},${binEdges[b + 1]}): n=${c}  ${names.map((nm, g) => `${nm}=${mean[g].toFixed(2)}`).join('  ')}  -> argmax=${names[am]}`)
  }
  const [b, s, h] = argmax[st]
  console.log(`  overall argmax (n=${amCnt[st]}): BUY=${(100 * b / amCnt[st]).toFixed(0)}% SELL=${(100 * s / amCnt[st]).toFixed(0)}% HOLD=${(100 * h / amCnt[st]).toFixed(0)}%`)
}