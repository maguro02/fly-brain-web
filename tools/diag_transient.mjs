// Isolate why the state signal present in steady-state disappears in the
// switching task loop. Two parts:
//   A) pure steady-state (fixed lamp, no switching) in this exact code ->
//      confirm the state signal exists here.
//   B) switching task loop, group rates binned by time-since-switch ->
//      reveal whether a slow transient keeps SELL high after each switch.
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
const names = ['BUY', 'SELL', 'HOLD']

// ---- A) pure steady-state in this code ----
function steady(state, warmup, steps) {
  setLamp(state)
  applyVisual()
  for (let i = 0; i < warmup; i++) step()
  const acc = [0, 0, 0]
  for (let i = 0; i < steps; i++) {
    step()
    const r = groupRates()
    acc[0] += r[0]; acc[1] += r[1]; acc[2] += r[2]
  }
  return [acc[0] / steps, acc[1] / steps, acc[2] / steps]
}
console.log('A) pure steady-state (this code, no switching, no bursts):')
const su = steady('up', 300, 1200)
const sd = steady('down', 300, 1200)
console.log(`   up  : ${su.map((v) => v.toFixed(3)).join('  ')}`)
console.log(`   down: ${sd.map((v) => v.toFixed(3)).join('  ')}`)

// ---- B) switching task loop, binned by time-since-switch ----
// reset network
V.fill(meta.v_rest); I.fill(0); neuronEma.fill(0); spikedCount = 0
let state = 'up'
let stateLeft = 600
let timeSinceSwitch = 0
function nextDuration() { return 480 + Math.floor(rnd() * 420) }

const EPOCHS = 400
const DECISION_EVERY = 60
setLamp(state)
applyVisual()
for (let i = 0; i < 300; i++) step()
timeSinceSwitch = 300

// bins: [0,60),[60,120),[120,240),[240,480),[480,inf)
const binEdges = [0, 60, 120, 240, 480, Infinity]
const binAcc = { up: binEdges.slice(0, -1).map(() => [0, 0, 0]), down: binEdges.slice(0, -1).map(() => [0, 0, 0]) }
const binCnt = { up: binEdges.slice(0, -1).map(() => 0), down: binEdges.slice(0, -1).map(() => 0) }

for (let epoch = 0; epoch < EPOCHS; epoch++) {
  for (let s = 0; s < DECISION_EVERY; s++) {
    stateLeft--
    if (stateLeft <= 0) {
      state = state === 'up' ? 'down' : 'up'
      stateLeft = nextDuration()
      timeSinceSwitch = 0
    }
    setLamp(state)
    applyVisual()
    step()
    timeSinceSwitch++
  }
  const rates = groupRates()
  let b = 0
  while (b < binEdges.length - 2 && timeSinceSwitch >= binEdges[b + 1]) b++
  for (let g = 0; g < 3; g++) binAcc[state][b][g] += rates[g]
  binCnt[state][b]++
}

console.log('\nB) switching task loop, group rates binned by time-since-switch:')
for (const st of ['up', 'down']) {
  console.log(`  state=${st}:`)
  for (let b = 0; b < binCnt[st].length; b++) {
    const c = binCnt[st][b]
    if (c === 0) continue
    const mean = binAcc[st][b].map((v) => v / c)
    const lo = binEdges[b], hi = binEdges[b + 1]
    console.log(`    tss [${lo},${hi === Infinity ? 'inf' : hi}): n=${c}  ${names.map((nm, g) => `${nm}=${mean[g].toFixed(2)}`).join('  ')}`)
  }
}