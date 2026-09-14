// Build RATIO-partially-prewired action groups and write them to
// data/motor_groups.json + data/metadata.json (motor_groups).
// BUY = top-K up-selective + neutral fill, SELL = top-K down-selective +
// neutral fill, HOLD = neutral. usage: node tools/build_ratio_groups.mjs [ratio=0.2]
import { readFileSync, writeFileSync, existsSync } from 'node:fs'

const RATIO = Number(process.argv[2] || 0.2)
const G = 438

const buf = readFileSync('data/connectome.bin')
const meta = JSON.parse(readFileSync('data/metadata.json', 'utf8'))
const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
const n = dv.getUint32(8, true)
const m = dv.getUint32(12, true)
const nVisual = dv.getUint32(16, true)
const nMotor = dv.getUint32(20, true)
let off = 40 + (n + 1) * 4 + m * 4 + m * 4
const visualMap = new Int32Array(buf.buffer, buf.byteOffset + off, nVisual); off += nVisual * 4
const motorMap = new Int32Array(buf.buffer, buf.byteOffset + off, nMotor)

const CACHE = 'data/contrast_cache.bin'
if (!existsSync(CACHE)) throw new Error('missing contrast cache; run tools/learn_curve.mjs once to build it')
const cbuf = readFileSync(CACHE)
const contrast = new Float32Array(cbuf.buffer, cbuf.byteOffset + 2 * n * 4, n)

const excluded = new Uint8Array(n)
for (let c = 0; c < nVisual; c++) excluded[visualMap[c]] = 1
for (let k = 0; k < nMotor; k++) excluded[motorMap[k]] = 1
const pool = []
for (let i = 0; i < n; i++) if (!excluded[i]) pool.push(i)
pool.sort((x, y) => contrast[y] - contrast[x])

const K = Math.round(RATIO * G)
const buySel = pool.slice(0, K)
const sellSel = pool.slice(pool.length - K, pool.length)
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
const meanContrast = (g) => g.reduce((s, i) => s + contrast[i], 0) / g.length
console.log(
  `groups: RATIO=${RATIO} K=${K}  BUY c=${meanContrast(buy).toFixed(2)}  SELL c=${meanContrast(sell).toFixed(2)}  HOLD c=${meanContrast(hold).toFixed(2)}`,
)

writeFileSync('data/motor_groups.json', JSON.stringify({ ratio: RATIO, groups: [buy, sell, hold] }))
meta.motor_groups = [buy, sell, hold]
writeFileSync('data/metadata.json', JSON.stringify(meta))
console.log('wrote data/motor_groups.json + data/metadata.json')