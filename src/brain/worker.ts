import { MAGIC, FORMAT_VERSION, MSG, type ConnectomeMeta } from './types'

// LIF network on the MaleCNS connectome graph, with dopamine-gated Hebbian
// plasticity. Binary layout v2 (little-endian):
//   header:   u32[10] = magic, version, n, m, n_visual, n_motor, n_classes,
//                              n_dan, n_gaba, n_soma
//   row_ptr:    i32[n+1]
//   col_idx:    i32[m]
//   weights:    f32[m]
//   visual_map: i32[n_visual]
//   motor_map:  i32[n_motor]
//   dan_map:    i32[n_dan]
//   gaba_map:   i32[n_gaba]
//   soma_idx:   i32[n_soma]
//   soma_xyz:   f32[3*n_soma]
//   neuron_class: u8[n]   (last; keeps i32/f32 arrays 4-byte aligned)

// plasticity knobs
const ETA = 0.1 // dopamine-gated Hebbian learning rate
const COMP = 0.5 // on reward, weaken the other action groups by ETA*COMP
const INIT_SCALE = 1.0 // random init scale for cue->action synapses
const W_MAX = 2.0
const W_MIN = -2.0
const BURST_STEPS = 3 // reward/punish burst duration (steps)

let n = 0
let m = 0
let rowPtr: Int32Array = new Int32Array(0)
let colIdx: Int32Array = new Int32Array(0)
let W: Float32Array = new Float32Array(0)
let W0: Float32Array | null = null
let neuronClass: Uint8Array = new Uint8Array(0)
let visualMap: Int32Array = new Int32Array(0)
let motorMap: Int32Array = new Int32Array(0)
let danMap: Int32Array = new Int32Array(0)
let gabaMap: Int32Array = new Int32Array(0)
let somaIdx: Int32Array = new Int32Array(0)
let nVisual = 0
let nMotor = 0
let nClasses = 0
let nDan = 0
let nGaba = 0
let nSoma = 0

let V: Float32Array = new Float32Array(0)
let I: Float32Array = new Float32Array(0)
let vis: Float32Array = new Float32Array(0)
let neuronEma: Float32Array = new Float32Array(0)
let spiked: Int32Array = new Int32Array(0)
let spikedCount = 0

let motorEma: Float32Array = new Float32Array(0)
let classEma: Float32Array = new Float32Array(0)
let classCount: Uint32Array = new Uint32Array(0)
let classNeuronEmaSum: Float32Array = new Float32Array(0)

// targeted plasticity state
let curState: 'up' | 'down' = 'up'
let cueUp: Int32Array = new Int32Array(0)
let cueDown: Int32Array = new Int32Array(0)
let groupMask: Uint8Array[] = []
let plasticityOn = true
let rewardBurst = 0
let punishBurst = 0
let plasticityUpdates = 0 // running total of weight changes (for UI)

let meta: ConnectomeMeta | null = null
let simTime = 0
let running = false
let timer: ReturnType<typeof setInterval> | null = null
let statsTimer: ReturnType<typeof setInterval> | null = null
let activityTimer: ReturnType<typeof setInterval> | null = null

const EMA_ALPHA = 0.08
let spikeCountWindow = 0
let windowSteps = 0

// deterministic PRNG so the random cue->action init (and background activity)
// reproduce the verified headless experiment; a fixed starting mapping keeps
// the plasticity-off control at a known (chance) baseline.
let rngSeed = 0x12345678
function rnd() {
  rngSeed = (Math.imul(rngSeed, 1664525) + 1013904223) >>> 0
  return rngSeed / 4294967296
}

function load(buffer: ArrayBuffer) {
  const dv = new DataView(buffer)
  const magic = dv.getUint32(0, true)
  if (magic !== MAGIC) throw new Error(`bad magic: ${magic.toString(16)}`)
  const version = dv.getUint32(4, true)
  if (version !== FORMAT_VERSION) throw new Error(`unsupported version ${version}`)
  n = dv.getUint32(8, true)
  m = dv.getUint32(12, true)
  nVisual = dv.getUint32(16, true)
  nMotor = dv.getUint32(20, true)
  nClasses = dv.getUint32(24, true)
  nDan = dv.getUint32(28, true)
  nGaba = dv.getUint32(32, true)
  nSoma = dv.getUint32(36, true)

  let off = 40
  rowPtr = new Int32Array(buffer, off, n + 1); off += (n + 1) * 4
  colIdx = new Int32Array(buffer, off, m); off += m * 4
  W = new Float32Array(buffer, off, m); off += m * 4
  W0 = new Float32Array(W)
  visualMap = new Int32Array(buffer, off, nVisual); off += nVisual * 4
  motorMap = new Int32Array(buffer, off, nMotor); off += nMotor * 4
  danMap = new Int32Array(buffer, off, nDan); off += nDan * 4
  gabaMap = new Int32Array(buffer, off, nGaba); off += nGaba * 4
  somaIdx = new Int32Array(buffer, off, nSoma); off += nSoma * 4
  off += 3 * nSoma * 4 // soma_xyz (used only by the 3D view, skip)
  neuronClass = new Uint8Array(buffer, off, n); off += n

  V = new Float32Array(n)
  I = new Float32Array(n)
  vis = new Float32Array(n)
  neuronEma = new Float32Array(n)
  spiked = new Int32Array(n)
  motorEma = new Float32Array(nMotor)
  classEma = new Float32Array(nClasses)
  classCount = new Uint32Array(nClasses)
  classNeuronEmaSum = new Float32Array(nClasses)
  V.fill(meta ? meta.v_rest : -65)
  for (let i = 0; i < n; i++) classCount[neuronClass[i]]++

  buildCues()
  buildGroupMask()
  initCueGroup()

  self.postMessage({
    type: MSG.LOADED,
    n_neurons: n,
    n_connections: m,
    n_visual: nVisual,
    n_motor: nMotor,
    n_dan: nDan,
    n_gaba: nGaba,
    n_soma: nSoma,
  })
}

function setVisual(data: Float32Array) {
  const cols = Math.min(data.length, nVisual)
  for (let c = 0; c < cols; c++) {
    const idx = visualMap[c]
    if (idx >= 0 && idx < n) vis[idx] = data[c]
  }
}

// retina blocks of the cue pattern (left = up, right = down) mapped to
// connectome neuron indices
function buildCues() {
  // visual neurons are laid out row-major over a square retina; nVisual is the
  // cell count (1024), not the side length.
  const RET = Math.round(Math.sqrt(nVisual)) || 32
  const up: number[] = []
  const down: number[] = []
  for (let c = 0; c < nVisual; c++) {
    const x = c % RET
    const y = (c / RET) | 0
    if (y >= 2 && y < 10) {
      if (x >= 2 && x < 12) up.push(visualMap[c])
      else if (x >= 20 && x < 30) down.push(visualMap[c])
    }
  }
  cueUp = Int32Array.from(up)
  cueDown = Int32Array.from(down)
}

function buildGroupMask() {
  const groups = meta ? meta.motor_groups : []
  groupMask = groups.map((g) => {
    const mask = new Uint8Array(n)
    for (const i of g) mask[i] = 1
    return mask
  })
}

// erase the pre-wired cue->action signal and re-init those synapses with a
// small random weight, so the fly starts with no learned policy (near-chance
// readout). plasticity must rebuild the association from scratch.
function initCueGroup() {
  const inAny = new Uint8Array(n)
  for (const mask of groupMask) for (let i = 0; i < n; i++) if (mask[i]) inAny[i] = 1
  for (const set of [cueUp, cueDown]) {
    for (let ci = 0; ci < set.length; ci++) {
      const i = set[ci]
      for (let k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
        if (inAny[colIdx[k]]) W[k] = (rnd() - 0.5) * INIT_SCALE
      }
    }
  }
}

// dopamine-gated, targeted plasticity: modify synapses from the active cue
// (state) visual neurons into action group. sign=+1 reinforces (reward),
// sign=-1 weakens (punishment). this directly changes the group's drive.
function targetedPlasticity(state: 'up' | 'down', groupIdx: number, sign: number) {
  const cueSet = state === 'up' ? cueUp : cueDown
  const mask = groupMask[groupIdx]
  if (!mask) return
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

function fireBurst(map: Int32Array, count: number) {
  for (let t = 0; t < count; t++) {
    const i = map[t]
    V[i] = meta ? meta.v_reset : -65
    spiked[spikedCount++] = i
    neuronEma[i] += (1 - neuronEma[i]) * EMA_ALPHA
  }
}

function step(dt: number) {
  const a = dt / (meta ? meta.tau_ms : 20)
  const vRest = meta ? meta.v_rest : -65
  const vTh = meta ? meta.v_th : -50
  const vReset = meta ? meta.v_reset : -65
  const bgRate = meta ? meta.background_rate_hz : 1
  const visGain = meta ? meta.vis_gain : 8

  // 1. propagate previous step's spikes (LIF + background + bursts)
  for (let s = 0; s < spikedCount; s++) {
    const j = spiked[s]
    const start = rowPtr[j]
    const end = rowPtr[j + 1]
    for (let k = start; k < end; k++) I[colIdx[k]] += W[k]
  }

  // 2. leak integration + visual drive, detect threshold spikes
  spikedCount = 0
  let totalSpikes = 0
  for (let i = 0; i < n; i++) {
    V[i] += (vRest - V[i]) * a + (I[i] + vis[i] * visGain) * dt
    I[i] = 0
    const sp = V[i] > vTh
    if (sp) {
      V[i] = vReset
      spiked[spikedCount++] = i
      totalSpikes++
    }
    neuronEma[i] += ((sp ? 1 : 0) - neuronEma[i]) * EMA_ALPHA
  }

  // 3. background spontaneous activity (unmodelled drive)
  const bg = bgRate * (dt / 1000) * n
  const extra = Math.floor(bg) + (rnd() < bg % 1 ? 1 : 0)
  for (let e = 0; e < extra; e++) {
    const i = (rnd() * n) | 0
    spiked[spikedCount++] = i
    neuronEma[i] += (1 - neuronEma[i]) * EMA_ALPHA
    totalSpikes++
  }

  // 4. reward / punishment bursts (DAN / GABA)
  if (rewardBurst > 0) {
    rewardBurst--
    fireBurst(danMap, nDan)
    totalSpikes += nDan
  }
  if (punishBurst > 0) {
    punishBurst--
    fireBurst(gabaMap, nGaba)
    totalSpikes += nGaba
  }

  // 5. motor + class readout
  for (let k = 0; k < nMotor; k++) {
    const i = motorMap[k]
    motorEma[k] += (neuronEma[i] - motorEma[k]) * EMA_ALPHA
  }
  for (let i = 0; i < n; i++) classNeuronEmaSum[neuronClass[i]] += neuronEma[i]
  for (let c = 0; c < nClasses; c++) {
    const mean = classCount[c] > 0 ? classNeuronEmaSum[c] / classCount[c] : 0
    classEma[c] += (mean - classEma[c]) * EMA_ALPHA
    classNeuronEmaSum[c] = 0
  }

  simTime += dt
  spikeCountWindow += totalSpikes
  windowSteps++
}

// weight-based readout: R[g] = sum over active-cue neurons i, group members j
// of W[i->j] * activity(i). directly sensitive to the plasticity-modified
// cue->action synapses; the LIF dynamics supply the cue activity.
function weightRates(): Float32Array {
  const cueSet = curState === 'up' ? cueUp : cueDown
  const out = new Float32Array(groupMask.length)
  for (let g = 0; g < groupMask.length; g++) {
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

function sendStats() {
  const hz = 60
  const motorRates = new Float32Array(nMotor)
  for (let k = 0; k < nMotor; k++) motorRates[k] = motorEma[k] * hz
  const classRates = new Float32Array(nClasses)
  for (let c = 0; c < nClasses; c++) classRates[c] = classEma[c] * hz
  const avgRate = windowSteps > 0 ? (spikeCountWindow / windowSteps) * 60 : 0
  spikeCountWindow = 0
  windowSteps = 0
  self.postMessage({
    type: MSG.STATS,
    time_ms: simTime,
    motor_rates: motorRates,
    group_rates: weightRates(),
    class_rates: classRates,
    total_spike_rate_hz: avgRate,
    plasticity_updates: plasticityUpdates,
    n_neurons: n,
    n_connections: m,
  })
}

function sendActivity() {
  // per-soma activity for the 3D brain point cloud
  const act = new Float32Array(nSoma)
  for (let k = 0; k < nSoma; k++) act[k] = neuronEma[somaIdx[k]]
  self.postMessage({ type: MSG.ACTIVITY, data: act }, [act.buffer])
}

function startLoop(dt: number) {
  if (timer) clearInterval(timer)
  if (statsTimer) clearInterval(statsTimer)
  if (activityTimer) clearInterval(activityTimer)
  timer = setInterval(() => {
    if (running) step(dt)
  }, 1000 / 60)
  statsTimer = setInterval(sendStats, 100)
  activityTimer = setInterval(sendActivity, 200)
}

self.onmessage = (e: MessageEvent) => {
  const msg = e.data
  try {
    switch (msg.type) {
      case MSG.LOAD:
        if (msg.meta) meta = msg.meta
        load(msg.buffer)
        break
      case MSG.VISUAL:
        setVisual(msg.data)
        if (msg.state === 'up' || msg.state === 'down') curState = msg.state
        break
      case MSG.STEP:
        if (meta) step(meta.dt_ms)
        break
      case MSG.REWARD: {
        const st = msg.state === 'up' || msg.state === 'down' ? msg.state : curState
        if (plasticityOn) {
          targetedPlasticity(st, msg.group, 1)
          for (let g = 0; g < groupMask.length; g++)
            if (g !== msg.group) targetedPlasticity(st, g, -COMP)
        }
        rewardBurst = BURST_STEPS
        break
      }
      case MSG.PUNISH: {
        const st = msg.state === 'up' || msg.state === 'down' ? msg.state : curState
        if (plasticityOn) targetedPlasticity(st, msg.group, -1)
        punishBurst = BURST_STEPS
        break
      }
      case 'setPlasticity':
        plasticityOn = !!msg.on
        break
      case 'setMeta':
        meta = msg.meta
        break
      case 'start':
        running = true
        if (meta) startLoop(meta.dt_ms)
        break
      case 'stop':
        running = false
        break
      case 'reset':
        if (W0) W.set(W0)
        V.fill(meta ? meta.v_rest : -65)
        I.fill(0)
        neuronEma.fill(0)
        motorEma.fill(0)
        classEma.fill(0)
        classNeuronEmaSum.fill(0)
        curState = 'up'
        simTime = 0
        rewardBurst = 0
        punishBurst = 0
        plasticityUpdates = 0
        spikeCountWindow = 0
        windowSteps = 0
        rngSeed = 0x12345678
        initCueGroup()
        break
    }
  } catch (err) {
    self.postMessage({ type: MSG.ERROR, message: String(err) })
  }
}