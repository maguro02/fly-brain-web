import { MSG, type ConnectomeMeta, type MotorStats, type ActivityMsg } from './types'

export class BrainClient {
  private worker: Worker
  meta: ConnectomeMeta | null = null
  stats: MotorStats | null = null
  activity: Float32Array | null = null
  somaXyz: Float32Array | null = null
  loaded = false
  private listeners = new Set<() => void>()

  constructor() {
    this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
    this.worker.onmessage = (e: MessageEvent) => {
      const msg = e.data
      if (msg.type === MSG.LOADED) {
        this.loaded = true
        this.notify()
      } else if (msg.type === MSG.STATS) {
        this.stats = msg
        this.notify()
      } else if (msg.type === MSG.ACTIVITY) {
        this.activity = (msg as ActivityMsg).data
        this.notify()
      } else if (msg.type === MSG.ERROR) {
        console.error('[brain]', msg.message)
      }
    }
  }

  async load(binUrl: string, metaUrl: string) {
    const [binRes, metaRes] = await Promise.all([fetch(binUrl), fetch(metaUrl)])
    if (!binRes.ok || !metaRes.ok) throw new Error('connectome data not found: ' + binUrl)
    const buffer = await binRes.arrayBuffer()
    const meta = (await metaRes.json()) as ConnectomeMeta
    this.meta = meta
    // soma geometry is only needed on the main thread; copy it out before the
    // buffer is transferred (and detached) to the worker
    const dv = new DataView(buffer)
    const n = dv.getUint32(8, true)
    const m = dv.getUint32(12, true)
    const nVisual = dv.getUint32(16, true)
    const nMotor = dv.getUint32(20, true)
    const nDan = dv.getUint32(28, true)
    const nGaba = dv.getUint32(32, true)
    const nSoma = dv.getUint32(36, true)
    let off = 40 + (n + 1) * 4 + m * 4 + m * 4 + nVisual * 4 + nMotor * 4 + nDan * 4 + nGaba * 4 + nSoma * 4
    const xyz = new Float32Array(buffer, off, 3 * nSoma)
    this.somaXyz = Float32Array.from(xyz)
    this.worker.postMessage({ type: MSG.LOAD, buffer, meta }, [buffer])
    this.worker.postMessage({ type: 'setMeta', meta })
  }

  start() {
    this.worker.postMessage({ type: 'start' })
  }

  stop() {
    this.worker.postMessage({ type: 'stop' })
  }

  setVisual(data: Float32Array, state?: 'up' | 'down') {
    this.worker.postMessage({ type: MSG.VISUAL, data, state })
  }

  reward(group: number, state?: 'up' | 'down') {
    this.worker.postMessage({ type: MSG.REWARD, group, state })
  }

  punish(group: number, state?: 'up' | 'down') {
    this.worker.postMessage({ type: MSG.PUNISH, group, state })
  }

  tick(group: number, state?: 'up' | 'down') {
    this.worker.postMessage({ type: MSG.TICK, group, state })
  }

  setPlasticity(on: boolean) {
    this.worker.postMessage({ type: 'setPlasticity', on })
  }

  reset() {
    this.worker.postMessage({ type: 'reset' })
  }

  onChange(fn: () => void) {
    this.listeners.add(fn)
    return () => {
      this.listeners.delete(fn)
    }
  }

  private notify() {
    this.listeners.forEach((fn) => fn())
  }

  dispose() {
    this.worker.terminate()
  }
}