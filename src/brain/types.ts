export const MAGIC = 0x43594c46 // "FLYC"
export const FORMAT_VERSION = 2

export interface ConnectomeMeta {
  n_neurons: number
  n_connections: number
  n_visual: number
  n_motor: number
  n_classes: number
  n_dan: number
  n_gaba: number
  n_soma: number
  dt_ms: number
  v_rest: number
  v_th: number
  v_reset: number
  tau_ms: number
  background_rate_hz: number
  weight_scale: number
  vis_gain: number
  min_synapses: number
  classes: string[]
  nt_sign_map: Record<string, number>
  nt_default_sign: number
  source: string
  motor_neurons: { name: string; index: number; group: number }[]
  motor_groups: number[][]
  visual_columns: number
}

export interface MotorStats {
  time_ms: number
  motor_rates: Float32Array
  group_rates: Float32Array
  class_rates: Float32Array
  total_spike_rate_hz: number
  plasticity_updates: number
  n_neurons: number
  n_connections: number
}

export interface ActivityMsg {
  type: typeof MSG.ACTIVITY
  data: Float32Array
}

export const MSG = {
  LOAD: 'load',
  LOADED: 'loaded',
  VISUAL: 'visual',
  STEP: 'step',
  REWARD: 'reward',
  PUNISH: 'punish',
  STATS: 'stats',
  ACTIVITY: 'activity',
  ERROR: 'error',
} as const