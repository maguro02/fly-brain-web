// Build RANDOM downstream action groups for the trading readout.
// The DN/motor layer readout is broken (SELL persistent-activity attractor).
// The downstream layer carries a robust cue signal (~18k up-selective, ~18k
// down-selective neurons). We assign RANDOM downstream neurons (excluding
// visual-input and motor neurons) to BUY/SELL/HOLD so the fly starts with a
// ~random cue->action mapping; dopamine/GABA-gated Hebbian plasticity must
// teach it the correct one.
import { readFileSync, writeFileSync } from 'node:fs'

const buf = readFileSync('data/connectome.bin')
const meta = JSON.parse(readFileSync('data/metadata.json', 'utf8'))
const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
const n = dv.getUint32(8, true); const m = dv.getUint32(12, true)
const nVisual = dv.getUint32(16, true); const nMotor = dv.getUint32(20, true)
const nDan = dv.getUint32(28, true); const nGaba = dv.getUint32(32, true); const nSoma = dv.getUint32(36, true)
let off = 40
off += (n + 1) * 4 + m * 4 + m * 4 // rowPtr, colIdx, weights
const visualMap = new Int32Array(buf.buffer, buf.byteOffset + off, nVisual); off += nVisual * 4
const motorMap = new Int32Array(buf.buffer, buf.byteOffset + off, nMotor); off += nMotor * 4
off += nDan * 4 + nGaba * 4 + nSoma * 4 + 3 * nSoma * 4

const G = 438 // neurons per group (3 groups, disjoint)
const SEED = 0xfeedc0de

// exclude visual-input and motor neurons
const excluded = new Uint8Array(n)
for (let c = 0; c < nVisual; c++) excluded[visualMap[c]] = 1
for (let k = 0; k < nMotor; k++) excluded[motorMap[k]] = 1

const pool = []
for (let i = 0; i < n; i++) if (!excluded[i]) pool.push(i)
console.log(`downstream pool: ${pool.length} neurons (excluded ${nVisual} visual + ${nMotor} motor)`)

// deterministic Fisher-Yates shuffle of the pool, take first 3*G
let seed = SEED
function rnd() { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296 }
const take = 3 * G
for (let i = pool.length - 1; i > 0; i--) {
  const j = (rnd() * (i + 1)) | 0
  const t = pool[i]; pool[i] = pool[j]; pool[j] = t
}
const chosen = pool.slice(0, take)
const groups = [chosen.slice(0, G), chosen.slice(G, 2 * G), chosen.slice(2 * G, 3 * G)]

// sanity: disjoint + all valid
const seen = new Set()
for (const g of groups) for (const idx of g) {
  if (seen.has(idx)) throw new Error(`duplicate neuron ${idx} in groups`)
  seen.add(idx)
  if (excluded[idx]) throw new Error(`neuron ${idx} is excluded (visual/motor)`)
}
console.log(`groups: ${groups.map((g) => g.length).join('/')} neurons, all disjoint + valid`)

// write intermediate file (consumed by build_connectome.py)
writeFileSync('data/motor_groups.json', JSON.stringify({ groups, seed: SEED, note: 'random downstream action groups' }))

// patch metadata.json (worker reads meta.motor_groups)
meta.motor_groups = groups
writeFileSync('data/metadata.json', JSON.stringify(meta))
console.log('wrote data/motor_groups.json + patched data/metadata.json')
console.log('group0 first5:', groups[0].slice(0, 5))
console.log('group1 first5:', groups[1].slice(0, 5))
console.log('group2 first5:', groups[2].slice(0, 5))