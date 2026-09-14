import { EPSILON, type Choice, type MarketState, type Outcome } from './market'

export interface LogEntry {
  epoch: number
  state: MarketState
  choice: Choice
  outcome: Outcome
  price: number
  pnl: number
  gap: number
}

export interface SimState {
  state: MarketState
  stateLeft: number
  price: number
  priceHist: number[]
  pos: -1 | 0 | 1
  entry: number
  realized: number
  epoch: number
  lastChoice: Choice | null
  lastChoiceTime: number
  lastOutcome: Outcome | null
  nCorrect: number
  nWrong: number
  nHold: number
  outcomes: Outcome[]
  accHist: { overall: number; active: number }[]
  dopamine: number
  gaba: number
  lastNeuroTime: number
  lastNeuroKind: 'reward' | 'punish' | null
  log: LogEntry[]
  paused: boolean
  version: number
}

export const ACC_WINDOW = 50

export function makeSim(): SimState {
  return {
    state: 'up',
    stateLeft: 10,
    price: 100,
    priceHist: [100],
    pos: 0,
    entry: 100,
    realized: 0,
    epoch: 0,
    lastChoice: null,
    lastChoiceTime: -1e9,
    lastOutcome: null,
    nCorrect: 0,
    nWrong: 0,
    nHold: 0,
    outcomes: [],
    accHist: [],
    dopamine: 0,
    gaba: 0,
    lastNeuroTime: -1e9,
    lastNeuroKind: null,
    log: [],
    paused: false,
    version: 0,
  }
}

export const sim: SimState = makeSim()

export function resetSim() {
  Object.assign(sim, makeSim())
  sim.version++
}

export function marketTick(s: SimState) {
  s.stateLeft--
  if (s.stateLeft <= 0) {
    s.state = s.state === 'up' ? 'down' : 'up'
    s.stateLeft = 8 + Math.floor(Math.random() * 8)
  }
  const drift = s.state === 'up' ? 0.004 : -0.004
  const noise = (Math.random() - 0.5) * 0.006
  s.price *= 1 + drift + noise
  s.priceHist.push(s.price)
  if (s.priceHist.length > 240) s.priceHist.shift()
}

export function unrealized(s: SimState) {
  return s.pos !== 0 ? (s.price - s.entry) * s.pos : 0
}

export function makeDecision(s: SimState, groupRates: Float32Array): { choice: Choice; outcome: Outcome } {
  marketTick(s)
  s.epoch++
  const eps = EPSILON
  let choice: Choice
  if (Math.random() < eps) choice = ((Math.random() * 3) | 0) as Choice
  else choice = groupRates.indexOf(Math.max(groupRates[0], groupRates[1], groupRates[2])) as Choice

  let pnl = 0
  const close = () => {
    if (s.pos !== 0) {
      pnl += (s.price - s.entry) * s.pos
      s.realized += pnl
      s.pos = 0
    }
  }
  if (choice === 0) {
    if (s.pos !== 1) {
      close()
      s.pos = 1
      s.entry = s.price
    }
  } else if (choice === 1) {
    if (s.pos !== -1) {
      close()
      s.pos = -1
      s.entry = s.price
    }
  } else close()

  const outcome: Outcome =
    choice === 2
      ? 'hold'
      : (choice === 0 && s.state === 'up') || (choice === 1 && s.state === 'down')
        ? 'correct'
        : 'wrong'

  if (outcome === 'correct') s.nCorrect++
  else if (outcome === 'wrong') s.nWrong++
  else s.nHold++

  s.outcomes.push(outcome)
  if (s.outcomes.length > ACC_WINDOW) s.outcomes.shift()
  const w = s.outcomes
  const c = w.filter((o) => o === 'correct').length
  const act = w.filter((o) => o !== 'hold').length
  s.accHist.push({ overall: c / w.length, active: act ? c / act : 0 })
  if (s.accHist.length > 500) s.accHist.shift()

  const gap = Math.max(...groupRates) - Math.min(...groupRates)
  s.log.unshift({ epoch: s.epoch, state: s.state, choice, outcome, price: s.price, pnl, gap })
  if (s.log.length > 60) s.log.pop()

  s.lastChoice = choice
  s.lastChoiceTime = performance.now()
  s.lastOutcome = outcome
  s.version++
  return { choice, outcome }
}