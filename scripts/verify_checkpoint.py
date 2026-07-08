"""Stage-by-stage verification of an OpenPI Orbax checkpoint.

Stages:
  0  on-disk structure & metadata
  1  sharding file (saved topology)
  2  array shapes/dtypes from _METADATA (no data load)
  3  actual array load + finite/absmax (optional, needs RAM/GPU)
  4  verdict

Stages 0-2 always work (metadata-only). Stage 3 is opt-in via --load.

Usage:
  /home/xudi_ge/openpi/.venv/bin/python scripts/verify_checkpoint.py
  /home/xudi_ge/openpi/.venv/bin/python scripts/verify_checkpoint.py --load
  /home/xudi_ge/openpi/.venv/bin/python scripts/verify_checkpoint.py \
      /path/to/ckpt/12000 --load
"""
from __future__ import annotations

import json
import os
import sys
import traceback

os.environ.setdefault("JAX_PLATFORMS", "cpu")

DEFAULT_CKPT = (
    "/data/gxdcheckpoint/dataV5_final_v3src_pad_conti/"
    "dataV5_final_v3src_pad_conti/12000"
)
ABS_MAX_HUGE_THRESHOLD = 1e6


def hr(title):
    print(f"\n===== {title} =====", flush=True)


def stage0(root):
    hr("Stage 0: on-disk structure")
    for sub in ("params", "train_state"):
        p = os.path.join(root, sub)
        if not os.path.isdir(p):
            print(f"  {sub}/: MISSING")
            continue
        print(f"  {sub}/: {sorted(os.listdir(p))}")
    meta_path = os.path.join(root, "_CHECKPOINT_METADATA")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        print("  metadata item_handlers:", list(meta.get("item_handlers", {}).keys()))


def stage1(root):
    hr("Stage 1: sharding file (saved topology)")
    for sub in ("params", "train_state"):
        sf = os.path.join(root, sub, "_sharding")
        if not os.path.exists(sf):
            print(f"  {sub}: no _sharding file")
            continue
        s = json.load(open(sf))
        k = next(iter(s))
        v = json.loads(s[k]) if isinstance(s[k], str) else s[k]
        print(f"  {sub}: n_arrays={len(s)}  example_mesh={v}")


def _walk_metadata(tree_metadata, prefix=""):
    """Yield (path, value_metadata) for each leaf in the tree_metadata dict."""
    for k, v in tree_metadata.items():
        if isinstance(v, dict) and "value_metadata" in v:
            yield prefix + str(k), v["value_metadata"]
        elif isinstance(v, dict):
            yield from _walk_metadata(v, prefix + str(k) + ".")


def stage2(root):
    hr("Stage 2: array shapes/dtypes from _METADATA (no data load)")
    import numpy as np
    for sub in ("params", "train_state"):
        mp = os.path.join(root, sub, "_METADATA")
        if not os.path.exists(mp):
            print(f"  {sub}: no _METADATA")
            continue
        m = json.load(open(mp))
        tm = m["tree_metadata"]
        leaves = list(_walk_metadata(tm))
        total_bytes = 0
        biggest = (0, None, None)
        n_nu = 0
        n_mu = 0
        for path, vm in leaves:
            shape = tuple(vm.get("write_shape", []))
            dtype_s = vm.get("value_type", "?")
            dtypes = {"DT_FLOAT": 4, "DT_BFLOAT16": 2, "DT_INT32": 4, "DT_DOUBLE": 8}
            itemsize = dtypes.get(dtype_s, 4)
            size = int(np.prod(shape)) * itemsize if shape else 0
            total_bytes += size
            if size > biggest[0]:
                biggest = (size, path, shape)
            if ".nu" in path or path.endswith("nu"):
                n_nu += 1
            if ".mu" in path or path.endswith("mu"):
                n_mu += 1
        print(f"  [{sub}] n_arrays={len(leaves)} total={total_bytes/2**30:.2f} GiB")
        print(f"  [{sub}] biggest: {biggest[1]} shape={biggest[2]} ({biggest[0]/2**20:.1f} MiB)")
        print(f"  [{sub}] n_nu={n_nu} n_mu={n_mu}")


def stage3(root):
    hr("Stage 3: load arrays & check finite/absmax (opt-in)")
    import jax
    import numpy as np
    import orbax.checkpoint as ocp
    from orbax.checkpoint import PyTreeCheckpointer

    cpu_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    def restore_one(subdir):
        ck = PyTreeCheckpointer()
        struct = ck.restore(os.path.join(root, subdir),
                            args=ocp.args.PyTreeRestoreArgs(restore_args=None))
        flat, treedef = jax.tree_util.tree_flatten(struct)
        if not flat or not isinstance(flat[0], jax.ShapeDtypeStruct):
            return struct
        shardings = jax.tree_util.tree_unflatten(treedef, [cpu_sharding] * len(flat))
        return ck.restore(os.path.join(root, subdir),
                          args=ocp.args.PyTreeRestoreArgs(restore_args=shardings))

    def summarize(name, tree):
        flat = jax.tree_util.tree_flatten_with_path(tree)[0]
        n_bad = n_huge = 0
        worst = (0.0, None)
        for kp, arr in flat:
            a = np.asarray(arr)
            if not np.isfinite(a).all(): n_bad += 1
            am = float(np.abs(a).max()) if a.size else 0.0
            if am > ABS_MAX_HUGE_THRESHOLD: n_huge += 1
            if am > worst[0]: worst = (am, jax.tree_util.keystr(kp))
        print(f"  [{name}] n_arrays={len(flat)} n_nonfinite={n_bad} n_absmax_gt_1e6={n_huge}")
        print(f"  [{name}] worst absmax = {worst[0]:.6e} at {worst[1]}")
        return {"n_nonfinite": n_bad, "n_huge": n_huge, "worst": worst}

    out = {}
    try:
        out["params"] = summarize("params", restore_one("params"))
    except Exception:
        print("  params load failed:"); traceback.print_exc()
    try:
        ts = restore_one("train_state")
        print("  train_state keys:", list(ts.keys()))
        if "opt_state" in ts:
            out["opt_state"] = summarize("opt_state", ts["opt_state"])
            def find_moments(prefix, t, acc):
                if isinstance(t, dict):
                    for k, v in t.items(): find_moments(f"{prefix}.{k}", v, acc)
                elif isinstance(t, (list, tuple)):
                    for i, v in enumerate(t): find_moments(f"{prefix}[{i}]", v, acc)
                else:
                    if "nu" in prefix or "mu" in prefix: acc.append((prefix, t))
            moments = []; find_moments("opt_state", ts["opt_state"], moments)
            for moment in ("mu", "nu"):
                leaves = [(p, a) for p, a in moments if f".{moment}" in p]
                if not leaves: continue
                worst = (0.0, None); n_bad = 0
                for path, arr in leaves:
                    a = np.asarray(arr)
                    if not np.isfinite(a).all(): n_bad += 1
                    am = float(np.abs(a).max()) if a.size else 0.0
                    if am > worst[0]: worst = (am, path)
                print(f"  [{moment}] n_leaves={len(leaves)} n_nonfinite={n_bad} worst_absmax={worst[0]:.6e} at {worst[1]}")
                out[moment] = {"n_nonfinite": n_bad, "worst": worst}
        if "step" in ts:
            print("  train_state.step =", ts["step"])
            out["step"] = ts["step"]
    except Exception:
        print("  train_state load failed:"); traceback.print_exc()
    return out


def stage4(s23):
    hr("Stage 4: verdict")
    if not s23:
        print("  Stage 3 was skipped (--load not passed) or failed.")
        print("  Verdict requires Stage 3 numeric check; rerun with --load when GPU/RAM is free.")
        return
    issues = []
    p = s23.get("params", {})
    o = s23.get("opt_state", {})
    nu = s23.get("nu", {})
    mu = s23.get("mu", {})
    if p.get("n_nonfinite", 0) > 0: issues.append(f"params has {p['n_nonfinite']} non-finite arrays")
    if o.get("n_nonfinite", 0) > 0: issues.append(f"opt_state has {o['n_nonfinite']} non-finite arrays")
    if nu.get("n_nonfinite", 0) > 0: issues.append(f"nu has {nu['n_nonfinite']} non-finite leaves (Adam second moment corrupt)")
    if nu.get("worst", (0.0, None))[0] > ABS_MAX_HUGE_THRESHOLD:
        issues.append(f"nu worst absmax = {nu['worst'][0]:.3e} (>1e6): optimizer second moment exploded")
    if mu.get("worst", (0.0, None))[0] > ABS_MAX_HUGE_THRESHOLD:
        issues.append(f"mu worst absmax = {mu['worst'][0]:.3e} (>1e6): optimizer first moment exploded")
    if not issues:
        print("  OK: checkpoint numerics look healthy.")
    else:
        print("  PROBLEM:")
        for i in issues: print(f"    - {i}")


def main():
    args = [a for a in sys.argv[1:] if a != "--load"]
    do_load = "--load" in sys.argv[1:]
    root = args[0] if args else DEFAULT_CKPT
    print(f"Verifying checkpoint: {root}  (load={do_load})")
    if not os.path.isdir(root):
        print(f"ERROR: not a directory: {root}"); return 1
    stage0(root)
    stage1(root)
    stage2(root)
    s23 = stage3(root) if do_load else {}
    stage4(s23)
    return 0


if __name__ == "__main__":
    sys.exit(main())
