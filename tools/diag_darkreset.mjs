// Does a DARK (lamp-off) period reset the SELL latch?
// 1) reach down steady state (SELL latched high)
// 2) turn lamp OFF (dark), track SELL/BUY/HOLD over 60s
// 3) then up for 60s, track
// Compare dark-decay vs up-decay of SELL.
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
const W = new Float32Array(buf.buffer, buf.byteOffset + off, m); off += m * 4
const visualMap = new Int32Array(buf.buffer, buf.byteOffset + off, nVisual); off += nVisual * 4
const motorMap = new Int32Array(buf.buffer, buf.byteOffset + off, nMotor); off += nMotor * 4
off += nDan * 4 + nGaba * 4 + nSoma * 4 + 3 * nSoma * 4

let seed = 0x12345678
function rnd() { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296 }
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
    V[i] += (meta.v_rest - V[i]) * a + (I[i] + vis[i] * meta.vis_gain) * dt
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
  else if (state === 'down') { for (let y = 2; y < 10; y++) for (let x = 20; x < 30; x++) visPat[y * RET + x] = 1 }
  // 'dark' -> stays all zero
  applyVisual()
}
const groups = meta.motor_groups
function groupRates() {
  const out = new Float32Array(groups.length)
  for (let g = 0; g < groups.length; g++) { const mem = groups[g]; let s = 0; for (let k = 0; k < mem.length; k++) s += neuronEma[mem[k]]; out[g] = (s / mem.length) * 60 }
  return out
}
const names = ['BUY', 'SELL', 'HOLD']

function phase(label, state, steps, sampleEvery = 600) {
  setLamp(state)
  let last = -sampleEvery
  for (let i = 0; i < steps; i++) {
    step()
    if (i - last >= sampleEvery) {
      last = i
      const r = groupRates()
      console.log(`  ${label} t=${((i + 1) * dt / 1000).toFixed(0)}s  ${names.map((nm, g) => `${nm}=${r[g].toFixed(2)}`).join('  ')}`)
    }
  }
}

console.log('reach down steady state (60s):')
phase('down', 'down', 6000, 1200)
console.log('turn DARK (60s):')
phase('dark', 'dark', 6000, 1200)
console.log('then UP (60s):')
phase('up', 'up', 6000, 1200)