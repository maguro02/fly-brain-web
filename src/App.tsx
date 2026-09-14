import { useEffect, useRef, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { World } from './scene/World'
import { BrainClient } from './brain/BrainClient'
import { sim, resetSim, makeDecision, unrealized } from './sim'
import { cuePattern, EPSILON } from './market'

function fmt(n: number) {
  return n.toLocaleString('en-US')
}

function PriceChart() {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    if (!c) return
    const ctx = c.getContext('2d')!
    const w = c.width
    const h = c.height
    ctx.fillStyle = '#0d1117'
    ctx.fillRect(0, 0, w, h)
    const hist = sim.priceHist
    let lo = Math.min(...hist)
    let hi = Math.max(...hist)
    const pad = (hi - lo) * 0.15 || 1
    lo -= pad
    hi += pad
    ctx.strokeStyle = '#21262d'
    ctx.lineWidth = 1
    for (let i = 0; i <= 3; i++) {
      const y = (h * i) / 3
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(w, y)
      ctx.stroke()
    }
    const up = hist[hist.length - 1] >= hist[0]
    ctx.strokeStyle = up ? '#2ecc71' : '#e74c3c'
    ctx.lineWidth = 1.5
    ctx.beginPath()
    for (let i = 0; i < hist.length; i++) {
      const x = (w * i) / (hist.length - 1)
      const y = h * (1 - (hist[i] - lo) / (hi - lo))
      if (i) ctx.lineTo(x, y)
      else ctx.moveTo(x, y)
    }
    ctx.stroke()
  }, [sim.version])
  return <canvas ref={ref} width={300} height={110} style={{ width: '100%', height: 110, display: 'block' }} />
}

function LearningCurve() {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    if (!c) return
    const ctx = c.getContext('2d')!
    const w = c.width
    const h = c.height
    ctx.fillStyle = '#0d1117'
    ctx.fillRect(0, 0, w, h)
    ctx.strokeStyle = '#21262d'
    ctx.beginPath()
    ctx.moveTo(0, h * 0.5)
    ctx.lineTo(w, h * 0.5)
    ctx.stroke()
    const hist = sim.accHist
    if (hist.length < 2) return
    const plot = (key: 'overall' | 'active', color: string) => {
      ctx.strokeStyle = color
      ctx.lineWidth = 1.5
      ctx.beginPath()
      for (let i = 0; i < hist.length; i++) {
        const x = (w * i) / (hist.length - 1)
        const y = h * (1 - hist[i][key])
        if (i) ctx.lineTo(x, y)
        else ctx.moveTo(x, y)
      }
      ctx.stroke()
    }
    plot('overall', '#7fd0ff')
    plot('active', '#2ecc71')
  }, [sim.version])
  return <canvas ref={ref} width={300} height={80} style={{ width: '100%', height: 80, display: 'block' }} />
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ fontSize: 11, color: '#5a7a94', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6 }}>{title}</div>
      {children}
    </div>
  )
}

function Row({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', color: '#9bc' }}>
      <span>{label}</span>
      <span style={{ color: color || '#cfe' }}>{value}</span>
    </div>
  )
}

const OUTCOME_COLOR: Record<string, string> = { correct: '#2ecc71', wrong: '#e74c3c', hold: '#f1c40f' }
const CHOICE_JA = ['買い', '売り', '保持'] as const
const OUTCOME_JA: Record<string, string> = { correct: '正解', wrong: '誤り', hold: '保持' }

export default function App() {
  const brainRef = useRef<BrainClient | null>(null)
  if (!brainRef.current) brainRef.current = new BrainClient()
  const brain = brainRef.current

  const [phase, setPhase] = useState<'loading' | 'ready' | 'error'>('loading')
  const [error, setError] = useState<string | null>(null)
  const [plasticityOn, setPlasticityOn] = useState(true)
  const [, setTick] = useState(0)
  const bump = () => setTick((t) => t + 1)

  useEffect(() => {
    let alive = true
    brain
      .load(
        `${import.meta.env.BASE_URL}data/connectome.bin`,
        `${import.meta.env.BASE_URL}data/metadata.json`,
      )
      .then(() => {
        if (!alive) return
        brain.setVisual(cuePattern(sim.state), sim.state)
        brain.start()
        setPhase('ready')
      })
      .catch((e) => {
        if (!alive) return
        setError(String(e))
        setPhase('error')
      })
    const off = brain.onChange(bump)
    return () => {
      alive = false
      off()
    }
  }, [brain])

  useEffect(() => {
    const id = setInterval(() => {
      if (sim.paused || !brain.loaded || !brain.stats) return
      const { choice, outcome } = makeDecision(sim, brain.stats.group_rates)
      if (choice !== 2) {
        brain.tick(choice, sim.state)
        sim.dopamine++
        sim.lastNeuroKind = 'reward'
        sim.lastNeuroTime = performance.now()
      }
      if (outcome === 'correct') {
        brain.reward(choice, sim.state)
        sim.dopamine++
        sim.lastNeuroKind = 'reward'
        sim.lastNeuroTime = performance.now()
      } else if (outcome === 'wrong') {
        brain.punish(choice, sim.state)
        sim.gaba++
        sim.lastNeuroKind = 'punish'
        sim.lastNeuroTime = performance.now()
      }
      brain.setVisual(cuePattern(sim.state), sim.state)
      bump()
    }, 1000)
    return () => clearInterval(id)
  }, [brain])

  const togglePause = () => {
    sim.paused = !sim.paused
    if (sim.paused) brain.stop()
    else brain.start()
    bump()
  }

  const restart = () => {
    brain.reset()
    resetSim()
    brain.setVisual(cuePattern(sim.state), sim.state)
    bump()
  }

  const togglePlasticity = () => {
    brain.setPlasticity(!plasticityOn)
    setPlasticityOn(!plasticityOn)
    bump()
  }

  const stats = brain.stats
  const gr = stats ? stats.group_rates : new Float32Array(3)
  const maxGr = Math.max(1e-6, gr[0], gr[1], gr[2])
  const argmax = gr.indexOf(Math.max(gr[0], gr[1], gr[2]))
  const lastAcc = sim.accHist.length ? sim.accHist[sim.accHist.length - 1] : null
  const neuroAge = performance.now() - sim.lastNeuroTime
  const neuroFlash = neuroAge < 700 ? (sim.lastNeuroKind === 'reward' ? '#2ecc71' : '#e74c3c') : null

  return (
    <div style={{ display: 'flex', height: '100%' }}>
      <div style={{ flex: 1, position: 'relative', minWidth: 0 }}>
        <Canvas camera={{ position: [0, 1.7, 2.8], fov: 50 }} dpr={[1, 1.5]}>
          <color attach="background" args={['#0a0c12']} />
          <fog attach="fog" args={['#0a0c12', 8, 25]} />
          <World brain={brain} />
        </Canvas>
        {phase === 'loading' && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#7fd0ff' }}>
            コネクトームを読み込み中（88 MB）…
          </div>
        )}
        {phase === 'error' && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#ff7a7a' }}>
            {error}
          </div>
        )}
        <div style={{ position: 'absolute', bottom: 10, left: 12, color: '#44586a', fontSize: 11, maxWidth: 460 }}>
          MaleCNS v2 · 実コネクトーム LIF + ドーパミン制御ヘビアン可塑性 · ドラッグで回転
        </div>
      </div>

      <div style={{ width: 340, borderLeft: '1px solid #1c2430', overflowY: 'auto', padding: 14, fontSize: 12, lineHeight: 1.5, background: '#0d1017' }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#7fd0ff', marginBottom: 4 }}>ハエ脳トレーダー</div>
        <div style={{ color: '#8aa', marginBottom: 12 }}>
          {phase === 'ready' && brain.meta ? (
            <>
              {fmt(brain.meta.n_neurons)} 神経細胞 · {fmt(brain.meta.n_connections)} シナプス
            </>
          ) : (
            '…'
          )}
        </div>

        <Section title="市場">
          <Row label="状態" value={sim.state === 'up' ? '▲ 上昇' : '▼ 下落'} color={sim.state === 'up' ? '#2ecc71' : '#e74c3c'} />
          <Row label="切り替わりまで" value={`${sim.stateLeft}s`} />
          <Row label="価格" value={sim.price.toFixed(2)} />
          <Row label="建玉" value={sim.pos === 1 ? '買い' : sim.pos === -1 ? '売り' : 'なし'} color={sim.pos === 1 ? '#2ecc71' : sim.pos === -1 ? '#e74c3c' : undefined} />
          <Row label="建玉価格" value={sim.pos !== 0 ? sim.entry.toFixed(2) : '—'} />
          <Row label="確定損益" value={`${sim.realized >= 0 ? '+' : ''}${sim.realized.toFixed(2)}`} color={sim.realized >= 0 ? '#2ecc71' : '#e74c3c'} />
          <Row label="含み損益" value={`${unrealized(sim) >= 0 ? '+' : ''}${unrealized(sim).toFixed(2)}`} color={unrealized(sim) >= 0 ? '#2ecc71' : '#e74c3c'} />
          <div style={{ marginTop: 8 }}>
            <PriceChart />
          </div>
        </Section>

        <Section title="判断（DN 3グループ）">
          {([0, 1, 2] as const).map((g) => (
            <div key={g} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
              <span style={{ width: 44, color: g === argmax ? '#fff' : '#8aa' }}>{CHOICE_JA[g]}</span>
              <div style={{ flex: 1, height: 8, background: '#141c26', borderRadius: 3 }}>
                <div
                  style={{
                    width: `${(gr[g] / maxGr) * 100}%`,
                    height: '100%',
                    background: g === 0 ? '#2ecc71' : g === 1 ? '#e74c3c' : '#f1c40f',
                    borderRadius: 3,
                    opacity: g === argmax ? 1 : 0.45,
                  }}
                />
              </div>
              <span style={{ width: 48, textAlign: 'right', color: '#689' }}>{gr[g].toFixed(1)} Hz</span>
            </div>
          ))}
          <div style={{ color: '#567', fontSize: 11, marginTop: 4 }}>
            最頻選択 → {CHOICE_JA[argmax]} · ε={EPSILON.toFixed(2)}
          </div>
        </Section>

        <Section title="学習">
          <Row label="試行" value={String(sim.epoch)} />
          <Row label="正解率（直近50回）" value={lastAcc ? `${(100 * lastAcc.overall).toFixed(0)}%` : '—'} />
          <Row label="買い・売りの正解率" value={lastAcc ? `${(100 * lastAcc.active).toFixed(0)}%` : '—'} />
          <Row label="正解 / 誤り / 保持" value={`${sim.nCorrect} / ${sim.nWrong} / ${sim.nHold}`} />
          <Row label="売買（直近50秒）" value={`${sim.outcomes.filter((o) => o !== 'hold').length} / 50`} color="#f1c40f" />
          <div style={{ marginTop: 8 }}>
            <LearningCurve />
          </div>
          <div style={{ color: '#567', fontSize: 11 }}>
            <span style={{ color: '#7fd0ff' }}>—</span> 全体 · <span style={{ color: '#2ecc71' }}>—</span> 買い・売り時
          </div>
        </Section>

        <Section title="可塑性">
          <Row label="重み更新数（ΔW）" value={stats ? fmt(stats.plasticity_updates) : '—'} />
          <Row label="ドーパミン放出" value={String(sim.dopamine)} color="#2ecc71" />
          <Row label="GABA 放出" value={String(sim.gaba)} color="#e74c3c" />
          <div style={{ color: '#567', fontSize: 11, marginTop: 4 }}>
            行動報酬: 売買した瞬間にドーパミン（勝敗無視）· 損失の抑制は弱い → やめられない脳
          </div>
          <div style={{ marginTop: 6, display: 'flex', alignItems: 'center', gap: 8 }}>
            <button
              onClick={togglePlasticity}
              style={{
                padding: '4px 10px',
                background: plasticityOn ? '#1c3a2a' : '#2a1c1c',
                color: plasticityOn ? '#2ecc71' : '#e74c3c',
                border: `1px solid ${plasticityOn ? '#2ecc71' : '#e74c3c'}`,
                borderRadius: 6,
                fontFamily: 'inherit',
                fontSize: 11,
                cursor: 'pointer',
              }}
            >
              可塑性: {plasticityOn ? 'ON' : 'OFF'}
            </button>
            <span style={{ color: '#567', fontSize: 11 }}>OFF+リセットで対照実験</span>
          </div>
          <div style={{ marginTop: 6, display: 'flex', alignItems: 'center', gap: 8 }}>
            <span
              style={{
                width: 10,
                height: 10,
                borderRadius: 5,
                background: neuroFlash || '#21262d',
                boxShadow: neuroFlash ? `0 0 8px ${neuroFlash}` : 'none',
              }}
            />
            <span style={{ color: '#689' }}>{neuroFlash ? (sim.lastNeuroKind === 'reward' ? 'ドーパミン放出!' : 'GABA 放出!') : '直近の放出なし'}</span>
          </div>
        </Section>

        <Section title="実験ログ">
          {sim.log.length === 0 && <div style={{ color: '#567' }}>最初の判断を待機中…</div>}
          {sim.log.slice(0, 10).map((e) => (
            <div key={e.epoch} style={{ display: 'flex', justifyContent: 'space-between', color: '#9bc', fontSize: 11 }}>
              <span style={{ color: '#567' }}>#{e.epoch}</span>
              <span style={{ color: e.state === 'up' ? '#2ecc71' : '#e74c3c' }}>{e.state === 'up' ? '上昇' : '下落'}</span>
              <span>{CHOICE_JA[e.choice]}</span>
              <span style={{ color: OUTCOME_COLOR[e.outcome] }}>{OUTCOME_JA[e.outcome]}</span>
              <span>{e.price.toFixed(1)}</span>
            </div>
          ))}
        </Section>

        <div style={{ display: 'flex', gap: 8 }}>
          <button
            onClick={togglePause}
            style={{
              flex: 1,
              padding: '8px 0',
              background: sim.paused ? '#1c3a2a' : '#2a1c1c',
              color: sim.paused ? '#2ecc71' : '#e74c3c',
              border: `1px solid ${sim.paused ? '#2ecc71' : '#e74c3c'}`,
              borderRadius: 6,
              fontFamily: 'inherit',
              fontSize: 12,
              cursor: 'pointer',
            }}
          >
            {sim.paused ? '再開' : '一時停止'}
          </button>
          <button
            onClick={restart}
            style={{
              flex: 1,
              padding: '8px 0',
              background: '#141c26',
              color: '#7fd0ff',
              border: '1px solid #2a6f9e',
              borderRadius: 6,
              fontFamily: 'inherit',
              fontSize: 12,
              cursor: 'pointer',
            }}
          >
            リセット
          </button>
        </div>
      </div>
    </div>
  )
}