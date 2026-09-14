// Parameterized learning-curve experiment.
// Builds RATIO-partially-prewired action groups (BUY = top-K up-selective +
// neutral fill, SELL = top-K down-selective + neutral fill, HOLD = neutral),
// then runs the market+brain loop with parameterized Hebbian plasticity.
// Pass criterion: plasticity run's accuracy must RISE (start<60% -> end>80%)
// while the control (plasticity off) stays low. The gap is the learning.
//
// usage: node tools/learn_curve.mjs [ratio=0] [eta=0.05] [window=30] [epochs=400] [plasticity=1] [eps=0.2] [comp=0.5]
import { readFileSync, writeFileSync, existsSync } from 'node:fs'

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
off += nSoma * 4 + 3 * nSoma * 4
const W0 = new Float32Array(W)

// ---- params ----
const RATIO = Number(process.argv[2] || 0)
const ETA = Number(process.argv[3] || 0.05)
const WINDOW = Number(process.argv[4] || 30)
const EPOCHS = Number(process.argv[5] || 400)
const PLASTICITY = process.argv[6] !== '0'
const EPS = Number(process.argv[7] || 0.2)
const COMP = Number(process.argv[8] || 0.5)
const CUE_INIT = Number(process.argv[9] || 1) // 0=zero, 1=random
const INIT_SCALE = Number(process.argv[10] || 0.6)

// ---- seeded RNG ----
let seed = 0x12345678
function rnd() {
  seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0
  return seed / 4294967296
}

// ---- LIF state ----
const dt = meta.dt_ms
const a = dt / meta.tau_ms
const V = new Float32Array(n)
const I = new Float32Array(n)
const vis = new Float32Array(n)
const neuronEma = new Float32Array(n)
const spiked = new Int32Array(n)
let spikedCount = 0
const ALPHA = 0.08

const RING = 150
const BURST_STEPS = 3
const W_MAX = 2.0
const W_MIN = -2.0
const ring = new Int32Array(RING).fill(-1)
let ringHead = 0
const lastSpike = new Int32Array(n).fill(-1e9)
let curStep = 0
let rewardBurst = 0
let punishBurst = 0
let plasticityUpdates = 0

function resetState() {
  V.fill(meta.v_rest)
  I.fill(0)
  vis.fill(0)
  neuronEma.fill(0)
  spikedCount = 0
  ring.fill(-1)
  ringHead = 0
  lastSpike.fill(-1e9)
  curStep = 0
  rewardBurst = 0
  punishBurst = 0
  plasticityUpdates = 0
}

function recordPlasticity(i) {
  lastSpike[i] = curStep
  ring[ringHead] = i
  ringHead = (ringHead + 1) % RING
}
function inWindow(j) {
  return curStep - lastSpike[j] < WINDOW
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
  for (let i = 0; i < n; i++) {
    V[i] += (meta.v_rest - V[i]) * a + (I[i] + vis[i] * meta.vis_gain) * dt
    I[i] = 0
    const sp = V[i] > meta.v_th
    if (sp) {
      V[i] = meta.v_reset
      spiked[spikedCount++] = i
      recordPlasticity(i)
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
  if (rewardBurst > 0) {
    rewardBurst--
    fireBurst(danMap, nDan)
  }
  if (punishBurst > 0) {
    punishBurst--
    fireBurst(gabaMap, nGaba)
  }
  curStep++
}

// ---- retina ----
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
// Cue visual neurons (retina block) mapped to connectome neuron indices.
const C_up = []
const C_down = []
for (let c = 0; c < nVisual; c++) {
  const x = c % RET
  const y = (c / RET) | 0
  if (y >= 2 && y < 10) {
    if (x >= 2 && x < 12) C_up.push(visualMap[c])
    else if (x >= 20 && x < 30) C_down.push(visualMap[c])
  }
}

// ---- contrast (cached) ----
const CACHE = 'data/contrast_cache.bin'
let up, down, contrast
if (existsSync(CACHE)) {
  const cbuf = readFileSync(CACHE)
  up = new Float32Array(cbuf.buffer, cbuf.byteOffset, n)
  down = new Float32Array(cbuf.buffer, cbuf.byteOffset + n * 4, n)
  contrast = new Float32Array(cbuf.buffer, cbuf.byteOffset + 2 * n * 4, n)
  console.log('loaded contrast cache')
} else {
  console.log('measuring per-neuron contrast (10s clean reset)...')
  const t0 = Date.now()
  function measure(state, secs) {
    const steps = secs * 100
    const acc = new Float32Array(n)
    for (let ep = 0; ep < 3; ep++) {
      seed = 0x12345678 + ep
      resetState()
      setLamp(state)
      for (let i = 0; i < steps; i++) step()
      for (let i = 0; i < n; i++) acc[i] += neuronEma[i]
    }
    for (let i = 0; i < n; i++) acc[i] = (acc[i] / 3) * 60
    return acc
  }
  up = measure('up', 10)
  down = measure('down', 10)
  contrast = new Float32Array(n)
  for (let i = 0; i < n; i++) contrast[i] = up[i] - down[i]
  const out = Buffer.alloc(3 * n * 4)
  new Float32Array(out.buffer, 0, n).set(up)
  new Float32Array(out.buffer, n * 4, n).set(down)
  new Float32Array(out.buffer, 2 * n * 4, n).set(contrast)
  writeFileSync(CACHE, out)
  console.log(`contrast measured in ${((Date.now() - t0) / 1000).toFixed(1)}s, cached`)
}

// ---- build RATIO groups ----
const excluded = new Uint8Array(n)
for (let c = 0; c < nVisual; c++) excluded[visualMap[c]] = 1
for (let k = 0; k < nMotor; k++) excluded[motorMap[k]] = 1
const pool = []
for (let i = 0; i < n; i++) if (!excluded[i]) pool.push(i)
pool.sort((x, y) => contrast[y] - contrast[x]) // descending contrast

const G = 438
const K = Math.round(RATIO * G)
const buySel = pool.slice(0, K) // top up-selective
const sellSel = pool.slice(pool.length - K, pool.length) // top down-selective
const selSet = new Set([...buySel, ...sellSel])
const neutral = pool.filter((i) => !selSet.has(i) && Math.abs(contrast[i]) < 0.5)
neutral.sort((x, y) => Math.abs(contrast[x]) - Math.abs(contrast[y]))
const buy = [...buySel, ...neutral.slice(0, G - K)]
const sell = [...sellSel, ...neutral.slice(G - K, 2 * (G - K))]
const hold = neutral.slice(2 * (G - K), 3 * (G - K))
const seen = new Set()
for (const idx of [...buy, ...sell, ...hold]) {
  if (seen.has(idx)) throw new Error(`duplicate ${idx}`)
  seen.add(idx)
}
function meanContrast(g) {
  return g.reduce((s, i) => s + contrast[i], 0) / g.length
}
console.log(
  `groups: RATIO=${RATIO} K=${K}  BUY c=${meanContrast(buy).toFixed(2)}  SELL c=${meanContrast(sell).toFixed(2)}  HOLD c=${meanContrast(hold).toFixed(2)}`,
)

// ---- choice readout ----
const groups = [buy, sell, hold]
function groupRates() {
  const out = new Float32Array(3)
  for (let g = 0; g < 3; g++) {
    const mem = groups[g]
    let s = 0
    for (let k = 0; k < mem.length; k++) s += neuronEma[mem[k]]
    out[g] = (s / mem.length) * 60
  }
  return out
}
// groupMask[g][i] = 1 if neuron i is in action group g (fast membership test)
const groupMask = groups.map((g) => {
  const mask = new Uint8Array(n)
  for (const i of g) mask[i] = 1
  return mask
})
// Dopamine-gated, targeted plasticity: modify synapses from the active cue
// (state s) visual neurons into action group g. sign=+1 reinforces (reward),
// sign=-1 weakens (punishment). This directly changes the group's drive.
function targetedPlasticity(state, groupIdx, sign) {
  const cueSet = state === 'up' ? C_up : C_down
  const mask = groupMask[groupIdx]
  const delta = sign * ETA
  for (let ci = 0; ci < cueSet.length; ci++) {
    const i = cueSet[ci]
    for (let k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
      const j = colIdx[k]
      if (mask[j]) {
        let w = W[k] + delta
        if (w > W_MAX) w = W_MAX
        else if (w < W_MIN) w = W_MIN
        W[k] = w
        plasticityUpdates++
      }
    }
  }
}
// Weight-based readout: R[g] = sum over active-cue neurons i, group members j
// of W[i->j] * activity(i). Directly sensitive to the plasticity-modified
// cue->action synapses; the LIF dynamics supply the cue activity.
function weightRates() {
  const cueSet = state === 'up' ? C_up : C_down
  const out = new Float32Array(3)
  for (let g = 0; g < 3; g++) {
    const mask = groupMask[g]
    let s = 0
    for (let ci = 0; ci < cueSet.length; ci++) {
      const i = cueSet[ci]
      const act = neuronEma[i]
      if (act === 0) continue
      for (let k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
        const j = colIdx[k]
        if (mask[j]) s += W[k] * act
      }
    }
    out[g] = s
  }
  return out
}
// Remove the pre-wired cue->action signal so the fly starts with no learned
// policy. mode 0: zero the synapses; mode 1: small random init (near-chance
// readout with real competition). Plasticity must build the association.
function initCueGroup(mode) {
  const inAny = new Uint8Array(n)
  for (let g = 0; g < 3; g++) for (const i of groups[g]) inAny[i] = 1
  let z = 0
  for (const cueSet of [C_up, C_down]) {
    for (const i of cueSet) {
      for (let k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
        if (inAny[colIdx[k]]) {
          W[k] = mode === 1 ? (rnd() - 0.5) * INIT_SCALE : 0
          z++
        }
      }
    }
  }
  return z
}

// ---- market ----
let state = 'up'
let stateLeft = 600
function nextDuration() {
  return 480 + Math.floor(rnd() * 420)
}

// ---- main loop ----
W.set(W0)
const nInit = initCueGroup(CUE_INIT)
console.log(
  `run: ratio=${RATIO} eta=${ETA} window=${WINDOW} epochs=${EPOCHS} plasticity=${PLASTICITY} eps=${EPS} comp=${COMP} cueInit=${CUE_INIT}(${nInit}) CueUp=${C_up.length} CueDown=${C_down.length}`,
)
const t0 = Date.now()
resetState()
setLamp(state)
for (let i = 0; i < 300; i++) step() // warmup

const DECISION_EVERY = 60
const argmaxChoice = { up: [0, 0, 0], down: [0, 0, 0] }
const argmaxCount = { up: 0, down: 0 }
let nCorrect = 0
let nWrong = 0
let nHold = 0

for (let epoch = 0; epoch < EPOCHS; epoch++) {
  for (let s = 0; s < DECISION_EVERY; s++) {
    stateLeft--
    if (stateLeft <= 0) {
      state = state === 'up' ? 'down' : 'up'
      stateLeft = nextDuration()
    }
    setLamp(state)
    step()
  }
  const rates = weightRates()
  const eps = EPS
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
    if (PLASTICITY) {
      targetedPlasticity(state, choice, 1)
      if (COMP > 0) for (let g = 0; g < 3; g++) if (g !== choice) targetedPlasticity(state, g, -COMP)
    }
    rewardBurst = BURST_STEPS
    nCorrect++
  } else if (outcome === 'wrong') {
    if (PLASTICITY) targetedPlasticity(state, choice, -1)
    punishBurst = BURST_STEPS
    nWrong++
  } else nHold++

  if (epoch % 25 === 0) {
    // windowed accuracy over last 50 decisions
    const winEnd = epoch
    const winStart = Math.max(0, epoch - 49)
    // recompute from a running log is cheaper; approximate with running totals
    const done = epoch + 1
    const overall = 100 * nCorrect / done
    const active = nCorrect + nWrong > 0 ? 100 * nCorrect / (nCorrect + nWrong) : 0
    console.log(
      `e${String(epoch).padStart(3)} ${state} overall=${overall.toFixed(0)}% active=${active.toFixed(0)}% B=${rates[0].toFixed(1)} S=${rates[1].toFixed(1)} H=${rates[2].toFixed(1)} plast=${plasticityUpdates}`,
    )
  }
}

const overall = 100 * nCorrect / EPOCHS
const active = nCorrect + nWrong > 0 ? 100 * nCorrect / (nCorrect + nWrong) : 0
console.log(
  `\nFINAL: overall=${overall.toFixed(1)}% active=${active.toFixed(1)}% C=${nCorrect} W=${nWrong} H=${nHold} plast=${plasticityUpdates}`,
)
for (const st of ['up', 'down']) {
  const tot = argmaxCount[st]
  const [b, s, h] = argmaxChoice[st]
  const acc = st === 'up' ? b : s
  const pct = (x) => (tot ? Math.round((100 * x) / tot) : 0)
  console.log(`argmax@${st}: n=${tot} BUY=${pct(b)}% SELL=${pct(s)}% HOLD=${pct(h)}% correct=${pct(acc)}%`)
}
console.log(`elapsed: ${((Date.now() - t0) / 1000).toFixed(1)}s`)