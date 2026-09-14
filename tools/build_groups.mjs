// Rebuild the 3 motor groups by per-DN left/right contrast ranking.
//
// Rationale: the old groups were LPT-balanced on mean rate, which equalized
// absolute group rates but scrambled the left/right contrast, so the state
// signal in the readout was ~0.1 Hz (buried under a ~0.25 Hz SELL bias and
// ~0.3 Hz per-decision noise). diag_contrast.mjs shows the DN layer carries
// strong contrast (top DNs up to ~30-55 Hz up-down difference). Ranking the
// groups by contrast makes the readout state-aligned by construction:
//   BUY  = top 438 up-selective DNs  (rate rises under up-lamp)
//   SELL = bottom 438 down-selective DNs (rate rises under down-lamp)
//   HOLD = middle 438 (contrast ~ 0)
//
// Inputs:
//   /tmp/opencode/dn_contrast.json  {up, down, contrast} indexed by DN-local k
//   data/metadata.json              motor_neurons[k].index = neuron idx of DN k
// Outputs:
//   data/motor_groups.json          {motor_groups, baseline_rates_hz}
//   data/metadata.json              motor_groups + motor_neurons[].group patched
// (binary connectome.bin is unchanged: it does not embed the group partition)
import { readFileSync, writeFileSync } from 'node:fs'

const dc = JSON.parse(readFileSync('/tmp/opencode/dn_contrast.json', 'utf8'))
const up = Float32Array.from(dc.up)
const down = Float32Array.from(dc.down)
const contrast = Float32Array.from(dc.contrast)
const nMotor = contrast.length

const meta = JSON.parse(readFileSync('data/metadata.json', 'utf8'))
const motorNeurons = meta.motor_neurons
if (motorNeurons.length !== nMotor) {
  throw new Error(`motor_neurons length ${motorNeurons.length} != nMotor ${nMotor}`)
}
const dnToNeuron = motorNeurons.map((mn) => mn.index)

const G = nMotor / 3
if (!Number.isInteger(G)) throw new Error(`nMotor ${nMotor} not divisible by 3`)

// rank DN-local indices by contrast, descending
const order = Array.from({ length: nMotor }, (_, k) => k).sort((a, b) => contrast[b] - contrast[a])
const buyLocal = order.slice(0, G)
const holdLocal = order.slice(G, 2 * G)
const sellLocal = order.slice(2 * G, 3 * G)

function toNeuron(locals) {
  return locals.map((k) => dnToNeuron[k])
}
const groups = [toNeuron(buyLocal), toNeuron(sellLocal), toNeuron(holdLocal)]
const names = ['BUY', 'SELL', 'HOLD']
const localSets = [buyLocal, sellLocal, holdLocal]

// verify it is a partition of all DNs
const seen = new Set()
for (const g of groups) for (const v of g) {
  if (seen.has(v)) throw new Error(`duplicate neuron ${v} in groups`)
  seen.add(v)
}
if (seen.size !== nMotor) throw new Error(`group partition size ${seen.size} != ${nMotor}`)

// per-group mean rates under each lamp + predicted argmax
function groupRate(locals, rates) {
  let s = 0
  for (const k of locals) s += rates[k]
  return s / locals.length
}
const upRates = localSets.map((ls) => groupRate(ls, up))
const downRates = localSets.map((ls) => groupRate(ls, down))
const meanRates = localSets.map((ls, i) => (upRates[i] + downRates[i]) / 2)

function argmax(arr) {
  let bi = 0
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[bi]) bi = i
  return bi
}
const predUp = argmax(upRates)
const predDown = argmax(downRates)

console.log('contrast-ranked motor groups:')
for (let g = 0; g < 3; g++) {
  console.log(`  ${names[g]}: n=${groups[g].length}  up=${upRates[g].toFixed(3)} Hz  down=${downRates[g].toFixed(3)} Hz  mean=${meanRates[g].toFixed(3)} Hz`)
}
console.log(`predicted argmax: up-lamp -> ${names[predUp]}  (correct=${predUp === 0})`)
console.log(`predicted argmax: down-lamp -> ${names[predDown]}  (correct=${predDown === 1})`)
console.log(`state signal up: BUY-SELL=${(upRates[0] - upRates[1]).toFixed(3)} Hz  BUY-HOLD=${(upRates[0] - upRates[2]).toFixed(3)} Hz`)
console.log(`state signal down: SELL-BUY=${(downRates[1] - downRates[0]).toFixed(3)} Hz  SELL-HOLD=${(downRates[1] - downRates[2]).toFixed(3)} Hz`)

// write motor_groups.json (consumed by build_connectome.py)
writeFileSync(
  'data/motor_groups.json',
  JSON.stringify({ motor_groups: groups, baseline_rates_hz: meanRates }, null, 2)
)

// patch metadata.json (what the worker actually reads)
meta.motor_groups = groups
for (let k = 0; k < nMotor; k++) {
  const neuron = dnToNeuron[k]
  motorNeurons[k].group = groups.findIndex((g) => g.includes(neuron))
}
writeFileSync('data/metadata.json', JSON.stringify(meta, null, 2))

console.log('wrote data/motor_groups.json + patched data/metadata.json')