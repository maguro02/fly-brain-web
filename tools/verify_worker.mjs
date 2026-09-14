// Headless verification of the SHIPPED worker (dist build): drives it exactly
// like App.tsx (VISUAL/REWARD/PUNISH once per second, stats-driven decisions)
// and prints the accuracy curve. usage: node tools/verify_worker.mjs [epochs=150] [plasticity=1]
import { readFileSync, readdirSync } from 'node:fs'

const EPOCHS = Number(process.argv[2] || 150)
const PLASTICITY = process.argv[3] !== '0'
const EPSILON = 0.2

const dist = readdirSync('dist/assets').find((f) => f.startsWith('worker-') && f.endsWith('.js'))
if (!dist) throw new Error('no built worker found; run npm run build first')

const RETINA = 32
function cuePattern(state) {
  const p = new Float32Array(RETINA * RETINA)
  if (state === 'up') for (let y = 2; y < 10; y++) for (let x = 2; x < 12; x++) p[y * RETINA + x] = 1
  else for (let y = 2; y < 10; y++) for (let x = 20; x < 30; x++) p[y * RETINA + x] = 1
  return p
}

let latestStats = null
let loaded = false
globalThis.self = {
  postMessage: (msg) => {
    if (msg.type === 'stats') latestStats = msg
    if (msg.type === 'loaded') loaded = true
  },
}
await import(`../dist/assets/${dist}`)
const post = (msg) => self.onmessage({ data: msg })

const buf = readFileSync('data/connectome.bin')
const meta = JSON.parse(readFileSync('data/metadata.json', 'utf8'))
post({ type: 'load', buffer: buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), meta })
post({ type: 'setMeta', meta })
if (!PLASTICITY) post({ type: 'setPlasticity', on: false })

// wait for LOADED
for (let i = 0; i < 200 && !loaded; i++) await new Promise((r) => setTimeout(r, 50))
// set the cue before start so the warmup runs with the visual active
post({ type: 'visual', data: cuePattern('up'), state: 'up' })
post({ type: 'start' })

// market replicated from src/sim.ts
let state = 'up'
let stateLeft = 10
let price = 100
function marketTick() {
  stateLeft--
  if (stateLeft <= 0) {
    state = state === 'up' ? 'down' : 'up'
    stateLeft = 8 + Math.floor(Math.random() * 8)
  }
  const drift = state === 'up' ? 0.004 : -0.004
  const noise = (Math.random() - 0.5) * 0.006
  price *= 1 + drift + noise
}

let nCorrect = 0
let nWrong = 0
let nHold = 0
const outcomes = []
const argmax = { up: [0, 0, 0], down: [0, 0, 0] }
const argmaxN = { up: 0, down: 0 }

const t0 = Date.now()
for (let epoch = 0; epoch < EPOCHS; epoch++) {
  await new Promise((r) => setTimeout(r, 1000))
  const gr = latestStats ? latestStats.group_rates : new Float32Array(3)
  marketTick()
  let choice
  let isArgmax = false
  if (Math.random() < EPSILON) choice = (Math.random() * 3) | 0
  else {
    choice = gr.indexOf(Math.max(gr[0], gr[1], gr[2]))
    isArgmax = true
  }
  if (isArgmax) {
    argmaxN[state]++
    argmax[state][choice]++
  }
  const outcome =
    choice === 2 ? 'hold' : (choice === 0 && state === 'up') || (choice === 1 && state === 'down') ? 'correct' : 'wrong'
  if (outcome === 'correct') {
    post({ type: 'reward', group: choice, state })
    nCorrect++
  } else if (outcome === 'wrong') {
    post({ type: 'punish', group: choice, state })
    nWrong++
  } else nHold++
  outcomes.push(outcome)
  post({ type: 'visual', data: cuePattern(state), state })

  if (epoch % 25 === 0 || epoch === EPOCHS - 1) {
    const w = outcomes.slice(-50)
    const c = w.filter((o) => o === 'correct').length
    const act = w.filter((o) => o !== 'hold').length
    const overall = 100 * nCorrect / (epoch + 1)
    const active = nCorrect + nWrong > 0 ? 100 * nCorrect / (nCorrect + nWrong) : 0
    console.log(
      `e${String(epoch).padStart(3)} ${state} overall=${overall.toFixed(0)}% active=${active.toFixed(0)}% win50=${(100 * c / w.length).toFixed(0)}% B=${gr[0].toFixed(1)} S=${gr[1].toFixed(1)} H=${gr[2].toFixed(1)} plast=${latestStats ? latestStats.plasticity_updates : 0}`,
    )
  }
}

const overall = 100 * nCorrect / EPOCHS
const active = nCorrect + nWrong > 0 ? 100 * nCorrect / (nCorrect + nWrong) : 0
console.log(`\nFINAL: overall=${overall.toFixed(1)}% active=${active.toFixed(1)}% C=${nCorrect} W=${nWrong} H=${nHold} plast=${latestStats ? latestStats.plasticity_updates : 0}`)
for (const st of ['up', 'down']) {
  const tot = argmaxN[st]
  const [b, s, h] = argmax[st]
  const acc = st === 'up' ? b : s
  const pct = (x) => (tot ? Math.round((100 * x) / tot) : 0)
  console.log(`argmax@${st}: n=${tot} BUY=${pct(b)}% SELL=${pct(s)}% HOLD=${pct(h)}% correct=${pct(acc)}%`)
}
console.log(`elapsed: ${((Date.now() - t0) / 1000).toFixed(1)}s`)
process.exit(0)