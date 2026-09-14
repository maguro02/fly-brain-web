// Headless learning test: mirror of src/brain/worker.ts (v2) + a hidden-state
// market. The fly reads a cue lamp on its retina (left block = up, right block
// = down), picks BUY/SELL/HOLD by argmax of 3 descending-neuron group rates
// (eps-greedy), and gets a dopamine (DAN) burst for correct bets, a GABA burst
// for wrong ones. Hebbian plasticity is gated by those bursts. Pass criterion:
// late accuracy must exceed chance (33% overall / 50% among active bets).
//
// usage: node tools/learn_test.mjs [epochs=400] [noPlasticity]
import { readFileSync } from 'node:fs'

const buf = readFileSync('data/connectome.bin')
const meta = JSON.parse(readFileSync('data/metadata.json', 'utf8'))

// ---- parse binary v2 ----
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
console.log(`parsed: n=${n} m=${m} nVisual=${nVisual} nMotor=${nMotor} nDan=${nDan} nGaba=${nGaba} nSoma=${nSoma}`)

// ---- seeded RNG ----
let seed = 0x12345678
function rnd() {
  seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0
  return seed / 4294967296
}

// ---- LIF state (exact mirror of worker v2) ----
const dt = meta.dt_ms
const a = dt / meta.tau_ms
const V = new Float32Array(n).fill(meta.v_rest)
const I = new Float32Array(n)
const vis = new Float32Array(n)
const neuronEma = new Float32Array(n)
const spiked = new Int32Array(n)
let spikedCount = 0
const ALPHA = 0.08

const RING = 150
const WINDOW = 30
const ETA = 0.005
const W_MAX = 2.0
const W_MIN = -2.0
const BURST_STEPS = 3
const ring = new Int32Array(RING).fill(-1)
let ringHead = 0
const lastSpike = new Int32Array(n).fill(-1e9)
let curStep = 0
let rewardBurst = 0
let punishBurst = 0
let plasticityUpdates = 0

function recordPlasticity(i) {
  lastSpike[i] = curStep
  ring[ringHead] = i
  ringHead = (ringHead + 1) % RING
}
function inWindow(j) {
  return curStep - lastSpike[j] < WINDOW
}
function hebbian(sign) {
  const delta = sign * ETA
  for (let r = 0; r < RING; r++) {
    const i = ring[r]
    if (i < 0) continue
    for (let k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
      const j = colIdx[k]
      if (j === i) continue
      if (inWindow(j)) {
        let w = W[k] + delta
        if (w > W_MAX) w = W_MAX
        else if (w < W_MIN) w = W_MIN
        W[k] = w
        plasticityUpdates++
      }
    }
  }
}
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
  let total = 0
  for (let i = 0; i < n; i++) {
    V[i] += (meta.v_rest - V[i]) * a + (I[i] + vis[i] * meta.vis_gain) * dt
    I[i] = 0
    const sp = V[i] > meta.v_th
    if (sp) {
      V[i] = meta.v_reset
      spiked[spikedCount++] = i
      recordPlasticity(i)
      total++
    }
    neuronEma[i] += ((sp ? 1 : 0) - neuronEma[i]) * ALPHA
  }
  const bg = meta.background_rate_hz * (dt / 1000) * n
  const extra = Math.floor(bg) + (rnd() < bg % 1 ? 1 : 0)
  for (let e = 0; e < extra; e++) {
    const i = (rnd() * n) | 0
    spiked[spikedCount++] = i
    neuronEma[i] += (1 - neuronEma[i]) * ALPHA
    total++
  }
  if (rewardBurst > 0) {
    rewardBurst--
    fireBurst(danMap, nDan)
  }
  if (punishBurst > 0) {
    punishBurst--
    fireBurst(gabaMap, nGaba)
  }
  curStep++
  return total
}

// ---- retina: 32x32, cue lamp blocks ----
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

// ---- choice readout ----
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

// ---- market ----
let state = 'up'
let stateLeft = 600
function nextDuration() {
  return 480 + Math.floor(rnd() * 420) // 8..15 s in steps
}

// ---- main loop ----
const EPOCHS = Number(process.argv[2] || 400)
const ENABLE_PLASTICITY = process.argv[3] !== 'noPlasticity'
const DECISION_EVERY = 60 // 1 s

console.log(`run: epochs=${EPOCHS} plasticity=${ENABLE_PLASTICITY}`)
const t0 = Date.now()

setLamp(state)
applyVisual()
for (let i = 0; i < 300; i++) step() // warmup

const hist = []
let nCorrect = 0
let nWrong = 0
let nHold = 0
const argmaxCount = { up: 0, down: 0 }
const argmaxChoice = { up: [0, 0, 0], down: [0, 0, 0] }

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
    argmaxCount[state]++
    argmaxChoice[state][choice]++
  }

  let outcome
  if (choice === 2) outcome = 'hold'
  else if ((choice === 0 && state === 'up') || (choice === 1 && state === 'down')) outcome = 'correct'
  else outcome = 'wrong'

  if (outcome === 'correct') {
    if (ENABLE_PLASTICITY) hebbian(1)
    rewardBurst = BURST_STEPS
    nCorrect++
  } else if (outcome === 'wrong') {
    if (ENABLE_PLASTICITY) hebbian(-1)
    punishBurst = BURST_STEPS
    nWrong++
  } else nHold++

  hist.push({ epoch, state, choice, outcome, gap: Math.max(...rates) - Math.min(...rates) })

  if (epoch % 25 === 0) {
    const win = hist.slice(-50)
    const c = win.filter((h) => h.outcome === 'correct').length
    const act = win.filter((h) => h.outcome !== 'hold').length
    const gap = win.reduce((s, h) => s + h.gap, 0) / win.length
    console.log(
      `epoch ${String(epoch).padStart(3)}: overall=${(100 * c / win.length).toFixed(0)}% active=${act ? (100 * c / act).toFixed(0) : 0}% holds=${win.filter((h) => h.outcome === 'hold').length}/50 rateGap=${gap.toFixed(3)}Hz plast=${plasticityUpdates}`,
    )
  }
}

const win = hist.slice(-100)
const c = win.filter((h) => h.outcome === 'correct').length
const act = win.filter((h) => h.outcome !== 'hold').length
const overall = c / win.length
const active = act ? c / act : 0
const gap = win.reduce((s, h) => s + h.gap, 0) / win.length
console.log(
  `\nFINAL (last 100): overall=${(100 * overall).toFixed(1)}% (chance 33.3%) active=${(100 * active).toFixed(1)}% (chance 50%) rateGap=${gap.toFixed(3)}Hz plast=${plasticityUpdates}`,
)
for (const st of ['up', 'down']) {
  const tot = argmaxCount[st]
  const [b, s, h] = argmaxChoice[st]
  const acc = st === 'up' ? b : s
  console.log(
    `argmax@${st}: n=${tot} BUY=${tot ? (100 * b / tot).toFixed(0) : 0}% SELL=${tot ? (100 * s / tot).toFixed(0) : 0}% HOLD=${tot ? (100 * h / tot).toFixed(0) : 0}% -> correct=${tot ? (100 * acc / tot).toFixed(0) : 0}%`,
  )
}
console.log(`elapsed: ${((Date.now() - t0) / 1000).toFixed(1)}s`)
if (overall > 0.4 && active > 0.55) console.log('LEARNING TEST PASSED')
else console.log('LEARNING TEST: no clear learning yet')