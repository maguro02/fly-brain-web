import { Suspense, useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { OrbitControls, Text, useGLTF } from '@react-three/drei'
import * as THREE from 'three'
import { sim, type SimState } from '../sim'
import { CHOICE_NAMES, type Choice } from '../market'
import type { BrainClient } from '../brain/BrainClient'
import flyUrl from '../assets/fly.glb?url'

function drawMonitor(ctx: CanvasRenderingContext2D, w: number, h: number, s: SimState) {
  ctx.fillStyle = '#0d1117'
  ctx.fillRect(0, 0, w, h)
  ctx.fillStyle = '#161b22'
  ctx.fillRect(0, 0, w, 44)
  ctx.fillStyle = '#7fd0ff'
  ctx.font = 'bold 22px monospace'
  ctx.fillText('FLY/EX  ·  dopamine market', 16, 30)

  const lamp = (cx: number, on: boolean, color: string) => {
    ctx.beginPath()
    ctx.arc(cx, 22, 10, 0, Math.PI * 2)
    ctx.fillStyle = on ? color : '#21262d'
    ctx.fill()
  }
  lamp(w - 130, s.state === 'up', '#2ecc71')
  lamp(w - 70, s.state === 'down', '#e74c3c')

  ctx.fillStyle = '#e6edf3'
  ctx.font = 'bold 40px monospace'
  ctx.fillText(s.price.toFixed(2), 16, 96)
  ctx.font = '16px monospace'
  ctx.fillStyle = s.pos === 1 ? '#2ecc71' : s.pos === -1 ? '#e74c3c' : '#8b949e'
  ctx.fillText(s.pos === 1 ? 'LONG' : s.pos === -1 ? 'SHORT' : 'FLAT', 130, 94)

  const hist = s.priceHist
  const cw = w - 32
  const ch = 140
  const cy = 112
  let lo = Math.min(...hist)
  let hi = Math.max(...hist)
  const pad = (hi - lo) * 0.15 || 1
  lo -= pad
  hi += pad
  ctx.strokeStyle = '#21262d'
  ctx.lineWidth = 1
  for (let i = 0; i <= 4; i++) {
    const y = cy + (ch * i) / 4
    ctx.beginPath()
    ctx.moveTo(16, y)
    ctx.lineTo(w - 16, y)
    ctx.stroke()
  }
  const up = hist[hist.length - 1] >= hist[0]
  ctx.strokeStyle = up ? '#2ecc71' : '#e74c3c'
  ctx.lineWidth = 2
  ctx.beginPath()
  for (let i = 0; i < hist.length; i++) {
    const x = 16 + (cw * i) / (hist.length - 1)
    const y = cy + ch * (1 - (hist[i] - lo) / (hi - lo))
    if (i) ctx.lineTo(x, y)
    else ctx.moveTo(x, y)
  }
  ctx.stroke()

  const d = performance.now() - s.lastChoiceTime
  if (s.lastChoice != null && d < 900) {
    const name = CHOICE_NAMES[s.lastChoice]
    ctx.font = 'bold 54px monospace'
    const tw = ctx.measureText(name).width
    ctx.fillStyle = s.lastOutcome === 'correct' ? '#2ecc71' : s.lastOutcome === 'wrong' ? '#e74c3c' : '#f1c40f'
    ctx.fillText(name, w / 2 - tw / 2, h - 20)
  }
}

function Monitor() {
  const canvas = useMemo(() => {
    const c = document.createElement('canvas')
    c.width = 512
    c.height = 320
    return c
  }, [])
  const texture = useMemo(() => {
    const t = new THREE.CanvasTexture(canvas)
    t.colorSpace = THREE.SRGBColorSpace
    return t
  }, [canvas])
  const ver = useRef(-1)

  useFrame(() => {
    if (ver.current !== sim.version) {
      ver.current = sim.version
      drawMonitor(canvas.getContext('2d')!, canvas.width, canvas.height, sim)
      texture.needsUpdate = true
    }
  })

  return (
    <group position={[0, 1.0, -1.05]}>
      <mesh position={[0, 0.85, 0]}>
        <boxGeometry args={[2.4, 1.35, 0.08]} />
        <meshStandardMaterial color="#101318" />
      </mesh>
      <mesh position={[0, 0.85, 0.045]}>
        <planeGeometry args={[2.2, 1.15]} />
        <meshStandardMaterial map={texture} emissive="#ffffff" emissiveMap={texture} emissiveIntensity={0.6} />
      </mesh>
      <pointLight position={[0, 0.85, 0.7]} intensity={0.8} distance={3.5} color="#88aaff" />
    </group>
  )
}

function ActionButton({ x, choice, color }: { x: number; choice: Choice; color: string }) {
  const mat = useRef<THREE.MeshStandardMaterial>(null)
  useFrame(() => {
    if (!mat.current) return
    const d = performance.now() - sim.lastChoiceTime
    const active = sim.lastChoice === choice && d < 800
    mat.current.emissiveIntensity = active ? 1.6 + Math.sin(performance.now() / 50) * 0.4 : 0.25
  })
  return (
    <group position={[x, 1.06, 0.15]}>
      <mesh>
        <boxGeometry args={[0.5, 0.07, 0.32]} />
        <meshStandardMaterial ref={mat} color="#1a2028" emissive={color} emissiveIntensity={0.25} />
      </mesh>
      <Text position={[0, 0.065, 0]} rotation={[-Math.PI / 2, 0, 0]} fontSize={0.13} color={color}>
        {CHOICE_NAMES[choice]}
      </Text>
    </group>
  )
}

function FlyModel() {
  const { scene } = useGLTF(flyUrl)
  const group = useRef<THREE.Group>(null)
  const light = useRef<THREE.PointLight>(null)

  const fit = useMemo(() => {
    const box = new THREE.Box3().setFromObject(scene)
    const size = box.getSize(new THREE.Vector3())
    const center = box.getCenter(new THREE.Vector3())
    const s = 0.55 / Math.max(size.x, size.y, size.z)
    return { s, off: center.clone().multiplyScalar(-s), bottom: (box.min.y - center.y) * s }
  }, [scene])

  useFrame(({ clock }) => {
    const g = group.current
    if (!g) return
    // bottom of the normalized model sits on the desk top (y=1.0), bob up only
    g.position.y = 1.02 - fit.bottom + (Math.sin(clock.elapsedTime * 2.2) * 0.5 + 0.5) * 0.03
    const dNeuro = performance.now() - sim.lastNeuroTime
    let inten = 0
    if (dNeuro < 700) {
      const k = 1 - dNeuro / 700
      if (light.current) {
        light.current.color.set(sim.lastNeuroKind === 'reward' ? '#2ecc71' : '#e74c3c')
        inten = 2.5 * k
      }
    }
    if (light.current) light.current.intensity = inten
    const dChoice = performance.now() - sim.lastChoiceTime
    let lean = 0
    if (dChoice < 800 && sim.lastChoice != null) {
      const k = Math.sin((1 - dChoice / 800) * Math.PI)
      // head faces -z; +y rotation turns the nose toward -x (BUY side)
      lean = (sim.lastChoice === 0 ? 0.35 : sim.lastChoice === 1 ? -0.35 : 0) * k
    }
    g.rotation.y = lean
  })

  return (
    <group ref={group} position={[0, 1.3, 0.6]}>
      <group scale={fit.s} position={fit.off}>
        <primitive object={scene} />
      </group>
      <pointLight ref={light} intensity={0} distance={2.5} />
    </group>
  )
}

function BrainHologram({ brain }: { brain: BrainClient }) {
  const soma = brain.somaXyz
  const { geo, fit } = useMemo(() => {
    if (!soma) return { geo: null, fit: { s: 1, off: new THREE.Vector3() } }
    const xyz = soma
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(xyz, 3))
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(xyz.length), 3))
    let cx = 0
    let cy = 0
    let cz = 0
    const n = xyz.length / 3
    for (let i = 0; i < n; i++) {
      cx += xyz[i * 3]
      cy += xyz[i * 3 + 1]
      cz += xyz[i * 3 + 2]
    }
    cx /= n
    cy /= n
    cz /= n
    let maxR = 0
    for (let i = 0; i < n; i++) {
      const r = Math.hypot(xyz[i * 3] - cx, xyz[i * 3 + 1] - cy, xyz[i * 3 + 2] - cz)
      if (r > maxR) maxR = r
    }
    return { geo: g, fit: { s: 0.8 / maxR, off: new THREE.Vector3(-cx, -cy, -cz) } }
  }, [soma])

  const points = useRef<THREE.Points>(null)

  useEffect(() => {
    const off = brain.onChange(() => {
      if (!geo || !brain.activity) return
      const act = brain.activity
      const col = geo.getAttribute('color') as THREE.BufferAttribute
      const arr = col.array as Float32Array
      for (let i = 0; i < act.length; i++) {
        const a = Math.min(1, act[i] * 5)
        arr[i * 3] = 0.1 + a * 0.9
        arr[i * 3 + 1] = 0.18 + a * 0.55
        arr[i * 3 + 2] = 0.45 + a * 0.55
      }
      col.needsUpdate = true
    })
    return off
  }, [brain, geo])

  useFrame(({ clock }) => {
    if (points.current) points.current.rotation.y = clock.elapsedTime * 0.15
  })

  if (!geo) return null
  return (
    <group position={[2.1, 1.9, -0.55]}>
      <group scale={fit.s} position={fit.off}>
        <points ref={points} geometry={geo}>
          <pointsMaterial size={0.02} vertexColors sizeAttenuation transparent opacity={0.95} />
        </points>
      </group>
      <mesh>
        <sphereGeometry args={[0.85, 24, 16]} />
        <meshBasicMaterial color="#2a6f9e" wireframe transparent opacity={0.08} />
      </mesh>
      <pointLight intensity={0.5} distance={3} color="#4a9fd8" />
    </group>
  )
}

function Desk() {
  return (
    <group>
      <mesh position={[0, 0.94, 0]}>
        <boxGeometry args={[5.6, 0.12, 2.8]} />
        <meshStandardMaterial color="#2a2118" roughness={0.7} />
      </mesh>
      {[
        [-2.6, -1.2],
        [2.6, -1.2],
        [-2.6, 1.2],
        [2.6, 1.2],
      ].map(([x, z], i) => (
        <mesh key={i} position={[x, 0.46, z]}>
          <boxGeometry args={[0.14, 0.92, 0.14]} />
          <meshStandardMaterial color="#1c1712" roughness={0.8} />
        </mesh>
      ))}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[40, 40]} />
        <meshStandardMaterial color="#0c0e14" roughness={0.95} />
      </mesh>
    </group>
  )
}

export function World({ brain }: { brain: BrainClient }) {
  return (
    <>
      <ambientLight intensity={0.4} />
      <directionalLight position={[4, 8, 5]} intensity={0.9} />
      <Desk />
      <Monitor />
      <ActionButton x={-0.75} choice={0} color="#2ecc71" />
      <ActionButton x={0} choice={2} color="#f1c40f" />
      <ActionButton x={0.75} choice={1} color="#e74c3c" />
      <Suspense fallback={null}>
        <FlyModel />
      </Suspense>
      <BrainHologram brain={brain} />
      <OrbitControls target={[0, 1.15, -0.3]} enablePan={false} minDistance={1.5} maxDistance={8} maxPolarAngle={1.45} />
    </>
  )
}