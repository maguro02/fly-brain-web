// Diagnostic: run the task loop (state switches, lamp on), NO plasticity,
// NO reward/punishment bursts. Log per-decision group rates to see the real
// readout distribution and quantify the persistent SELL bias.
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

const EPOCHS = 200
const DECISION_EVERY = 60
let state = 'up'
let stateLeft = 600
function nextDuration() {
  return 480 + Math.floor(rnd() * 420)
}

setLamp(state)
for (let i = 0; i < 300; i++) step()

const decisions = []
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
  const rates = groupRates()
  decisions.push({ epoch, state, buy: rates[0], sell: rates[1], hold: rates[2] })
}

// analyze: mean + std of each group rate, per state
function stats(arr) {
  const mean = arr.reduce((s, v) => s + v, 0) / arr.length
  const variance = arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length
  return { mean, std: Math.sqrt(variance) }
}
for (const st of ['up', 'down']) {
  const sub = decisions.filter((d) => d.state === st)
  const b = stats(sub.map((d) => d.buy))
  const s = stats(sub.map((d) => d.sell))
  const h = stats(sub.map((d) => d.hold))
  // argmax distribution
  let nBuy = 0, nSell = 0, nHold = 0
  for (const d of sub) {
    const mx = Math.max(d.buy, d.sell, d.hold)
    if (d.buy === mx) nBuy++
    else if (d.sell === mx) nSell++
    else nHold++
  }
  const tot = sub.length
  console.log(`state=${st} (n=${tot}):`)
  console.log(`  BUY  mean=${b.mean.toFixed(3)} std=${b.std.toFixed(3)}`)
  console.log(`  SELL mean=${s.mean.toFixed(3)} std=${s.std.toFixed(3)}`)
  console.log(`  HOLD mean=${h.mean.toFixed(3)} std=${h.std.toFixed(3)}`)
  console.log(`  argmax: BUY=${(100 * nBuy / tot).toFixed(0)}% SELL=${(100 * nSell / tot).toFixed(0)}% HOLD=${(100 * nHold / tot).toFixed(0)}%`)
}
writeFileSync('/tmp/opencode/decisions.json', JSON.stringify(decisions))
console.log('wrote /tmp/opencode/decisions.json')