#!/usr/bin/env python3
"""Build a browser-loadable LIF connectome from MaleCNS v1.0 flat-connectome.

Output:
  data/connectome.bin   CSR binary v2 (little-endian)
  data/metadata.json    LIF params + sensory/motor/plasticity neuron indices

Neuron space: dense indices over annotated bodies (body-annotations bodyId,
VFB ID space). Edges: pair-level rows with both endpoints annotated and
synapse count >= 3. Sign from the source neuron's consensus neurotransmitter.

Binary layout v2 (4-byte arrays first, u8 last for alignment):
  header:   u32[10] = magic, version, n, m, n_visual, n_motor, n_classes,
                              n_dan, n_gaba, n_soma
  row_ptr:  i32[n+1]
  col_idx:  i32[m]
  weights:  f32[m]
  visual_map: i32[n_visual]
  motor_map:  i32[n_motor]
  dan_map:    i32[n_dan]
  gaba_map:   i32[n_gaba]
  soma_idx:   i32[n_soma]
  soma_xyz:   f32[3*n_soma]   (normalized to ~[-1,1]^3)
  neuron_class: u8[n]
"""
import json
import struct
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

RAW = "data/raw"
OUT = "data"
MAGIC = 0x43594C46  # "FLYC"
VERSION = 2
MIN_SYNAPSES = 3
RETINA_COLS = 1024  # 32x32
N_MOTOR_GROUPS = 3
GABA_CAP = 2000

# LIF params
DT_MS = 10.0
TAU_MS = 20.0
V_REST = -65.0
V_TH = -50.0
V_RESET = -65.0
BG_RATE_HZ = 1.0
WEIGHT_SCALE = 0.05
VIS_GAIN = 4.0

# Sign convention (per flybrain.online): ACh excitatory, GABA/glutamate inhibitory.
NT_SIGN = {
    "acetylcholine": 1.0,
    "histamine": 1.0,
    "glutamate": -1.0,
    "gaba": -1.0,
    "octopamine": -1.0,
    "dopamine": 0.0,
    "serotonin": 0.0,
}
NT_DEFAULT_SIGN = 1.0  # 'unclear' / unmodelled -> excitatory


def log(msg):
    print(f"[build] {msg}", flush=True)


def main():
    t0 = time.time()

    # ---- annotations: neuron space + per-neuron metadata ----
    log("loading annotations")
    ann = feather.read_table(f"{RAW}/body-annotations.feather")
    vfb_sorted = np.sort(ann["bodyId"].to_numpy())
    n = len(vfb_sorted)
    log(f"neurons: {n}")

    def dense_idx(vfb_arr):
        return np.searchsorted(vfb_sorted, vfb_arr)

    class_col = ann["class"].to_pandas().fillna("unknown")
    superclass_col = ann["superclass"].to_pandas().fillna("unknown")
    type_col = ann["type"].to_pandas().fillna("")
    vfb_arr = ann["bodyId"].to_numpy()
    pos_of = np.empty(n, dtype=np.int64)
    pos_of[dense_idx(vfb_arr)] = np.arange(n)
    class_sorted = class_col.to_numpy()[pos_of]
    superclass_sorted = superclass_col.to_numpy()[pos_of]
    type_sorted = type_col.to_numpy()[pos_of]

    class_names, class_ids = np.unique(class_sorted, return_inverse=True)
    n_classes = len(class_names)
    neuron_class = class_ids.astype(np.uint8)
    log(f"classes: {n_classes}")

    # ---- edges ----
    log("loading connectome")
    con = feather.read_table(f"{RAW}/connectome-weights.feather")
    annset = pa.array(vfb_sorted)
    both = pc.and_(
        pc.is_in(con["body_pre"], value_set=annset),
        pc.is_in(con["body_post"], value_set=annset),
    )
    sub = con.filter(both)
    keep = pc.greater_equal(sub["weight"], MIN_SYNAPSES)
    sub = sub.filter(keep)
    pre_vfb = sub["body_pre"].to_numpy()
    post_vfb = sub["body_post"].to_numpy()
    syn = sub["weight"].to_numpy()
    m = len(pre_vfb)
    log(f"edges (both annotated, syn>={MIN_SYNAPSES}): {m}")

    pre_idx = dense_idx(pre_vfb).astype(np.int32)
    post_idx = dense_idx(post_vfb).astype(np.int32)

    # ---- NT sign per source neuron ----
    log("loading neurotransmitters")
    nt = feather.read_table(f"{RAW}/body-neurotransmitters.feather").to_pandas()
    nt = nt[nt["consensus_nt"].notna() & (nt["consensus_nt"] != "unclear")]
    vc = (
        nt.groupby(["body", "consensus_nt"])
        .size()
        .reset_index(name="c")
        .sort_values(["body", "c"], ascending=[True, False])
        .drop_duplicates("body")
    )
    body_mode = vc.set_index("body")["consensus_nt"].to_dict()
    nt_counter = {k: 0 for k in NT_SIGN}
    nt_counter["default"] = 0
    sign_by_vfb = np.full(n, NT_DEFAULT_SIGN, dtype=np.float32)
    gaba_vfbs = []
    for body, label in body_mode.items():
        s = NT_SIGN.get(label, NT_DEFAULT_SIGN)
        sign_by_vfb[dense_idx(np.int64(body))] = s
        nt_counter[label if label in NT_SIGN else "default"] += 1
        if label == "gaba":
            gaba_vfbs.append(int(body))
    log(f"nt sign distribution: {nt_counter}")

    edge_sign = sign_by_vfb[pre_idx]
    weights = (edge_sign * syn.astype(np.float32) * WEIGHT_SCALE).astype(np.float32)

    # ---- CSR by source ----
    log("building CSR")
    order = np.argsort(pre_idx, kind="stable")
    col_idx = post_idx[order]
    weights = weights[order]
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    np.add.at(row_ptr, pre_idx + 1, 1)
    np.cumsum(row_ptr, out=row_ptr)

    # ---- sensory / motor maps ----
    visual_idx = np.where(class_sorted == "visual")[0].astype(np.int32)
    visual_map = visual_idx[:RETINA_COLS]
    motor_idx = np.where(superclass_sorted == "descending_neuron")[0].astype(np.int32)
    motor_map = motor_idx
    log(f"visual input neurons: {len(visual_map)} (of {len(visual_idx)})")
    log(f"motor (descending) neurons: {len(motor_map)}")

    # motor groups: one per action (BUY / SELL / HOLD). Prefer the rate-balanced
    # partition from tools/balance_groups.mjs; fall back to contiguous slices.
    import os
    groups_path = f"{OUT}/motor_groups.json"
    if os.path.exists(groups_path):
        with open(groups_path) as f:
            motor_groups = json.load(f)["motor_groups"]
        log(f"motor groups (balanced): {[len(g) for g in motor_groups]}")
    else:
        n_groups = N_MOTOR_GROUPS
        base = len(motor_map) // n_groups
        motor_groups = []
        for g in range(n_groups):
            lo = g * base
            hi = len(motor_map) if g == n_groups - 1 else (g + 1) * base
            motor_groups.append([int(i) for i in motor_map[lo:hi]])
        log(f"motor groups (contiguous): {[len(g) for g in motor_groups]}")

    # ---- DAN (dopamine) + GABA (punishment) maps ----
    dan_idx = np.where(class_sorted == "DAN")[0].astype(np.int32)
    gaba_sorted = np.sort(dense_idx(np.array(gaba_vfbs, dtype=np.int64)))
    gaba_map = gaba_sorted[:GABA_CAP].astype(np.int32)
    log(f"DAN neurons: {len(dan_idx)}, GABAergic (capped {GABA_CAP}): {len(gaba_map)}")

    # ---- soma positions (real 3D) ----
    log("extracting soma positions")
    soma_list = ann["somaLocation"].to_pylist()
    vfb_to_soma = {}
    for vfb, t in zip(vfb_arr, soma_list):
        if t:
            vfb_to_soma[int(vfb)] = t
    soma_vfbs = np.array(sorted(vfb_to_soma.keys()), dtype=np.int64)
    soma_idx = dense_idx(soma_vfbs).astype(np.int32)
    xyz = np.array([vfb_to_soma[int(v)] for v in soma_vfbs], dtype=np.float32)
    # normalize to ~[-1, 1]^3
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    rng = (hi - lo).max()
    xyz = (xyz - (lo + hi) / 2.0) / (rng / 2.0)
    log(f"soma positions: {len(soma_idx)}")

    # ---- write binary ----
    log("writing binary")
    bin_path = f"{OUT}/connectome.bin"
    with open(bin_path, "wb") as f:
        f.write(struct.pack("<IIIIIIIIII", MAGIC, VERSION, n, m, len(visual_map), len(motor_map), n_classes, len(dan_idx), len(gaba_map), len(soma_idx)))
        f.write(row_ptr.tobytes())
        f.write(col_idx.tobytes())
        f.write(weights.tobytes())
        f.write(visual_map.tobytes())
        f.write(motor_map.tobytes())
        f.write(dan_idx.tobytes())
        f.write(gaba_map.tobytes())
        f.write(soma_idx.tobytes())
        f.write(xyz.tobytes())
        f.write(neuron_class.tobytes())

    # ---- metadata ----
    group_of = {}
    for g, members in enumerate(motor_groups):
        for i in members:
            group_of[int(i)] = g
    motor_neurons = [
        {"name": str(type_sorted[i]), "index": int(i), "group": group_of.get(int(i), -1)}
        for i in motor_map
    ]
    meta = {
        "n_neurons": int(n),
        "n_connections": int(m),
        "n_visual": int(len(visual_map)),
        "n_motor": int(len(motor_map)),
        "n_classes": int(n_classes),
        "n_dan": int(len(dan_idx)),
        "n_gaba": int(len(gaba_map)),
        "n_soma": int(len(soma_idx)),
        "dt_ms": DT_MS,
        "v_rest": V_REST,
        "v_th": V_TH,
        "v_reset": V_RESET,
        "tau_ms": TAU_MS,
        "background_rate_hz": BG_RATE_HZ,
        "weight_scale": WEIGHT_SCALE,
        "vis_gain": VIS_GAIN,
        "min_synapses": MIN_SYNAPSES,
        "classes": [str(c) for c in class_names],
        "nt_sign_map": NT_SIGN,
        "nt_default_sign": NT_DEFAULT_SIGN,
        "visual_columns": RETINA_COLS,
        "motor_neurons": motor_neurons,
        "motor_groups": motor_groups,
        "source": "MaleCNS v1.0 (HHMI Janelia x Google Research, CC-BY, 2026-06-08)",
    }
    with open(f"{OUT}/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    log(f"done in {time.time()-t0:.1f}s")
    log(f"  {bin_path}")
    log(f"  {OUT}/metadata.json")


if __name__ == "__main__":
    sys.exit(main())