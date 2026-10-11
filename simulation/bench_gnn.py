"""Latency of the numpy graph models (software/python/gnn.py) on this machine."""

import json
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "software" / "python"))
sys.path.insert(0, str(HERE))

import gnn  # noqa: E402
from farm import Farm  # noqa: E402

HIDDEN = 32
FAST = {"kind": "fast", "n_features": 6, "classes": ["ok", "shade", "soiling", "obstruction", "debris_or_diode", "dead"]}
SLOW = {"kind": "slow", "n_features": 14, "classes": ["ok", "soiling", "degradation", "recurring_shade", "fault"]}


def random_npz(spec, path):
    rng = np.random.default_rng(0)
    h, f, c = HIDDEN, spec["n_features"], len(spec["classes"])
    shapes = {"encode.0.weight": (h, f), "encode.0.bias": (h,),
              "head.0.weight": (h, h), "head.0.bias": (h,), "head.2.weight": (c, h), "head.2.bias": (c,)}
    for name, out in (("cell.gates", 2 * h), ("cell.cand", h)):
        shapes.update({f"{name}.self_lin.weight": (out, 2 * h), f"{name}.self_lin.bias": (out,),
                       f"{name}.geo_lin.weight": (out, 2 * h), f"{name}.ele_lin.weight": (out, 2 * h),
                       f"{name}.geo_attr.weight": (out, 3), f"{name}.geo_attr.bias": (out,)})
    arrays = {k: (rng.standard_normal(s) * 0.1).astype(np.float32) for k, s in shapes.items()}
    meta = {**spec, "hidden": h, "peak_decay": 0.999, "dist_scale": 5.0, "ema_days": 7, "current_peak_decay": 0.995}
    np.savez(path, __meta__=np.array(json.dumps(meta)), **arrays)


def plant(n_target):
    if n_target <= 2:
        return Farm(rows=1, strings_per_row=1, modules_per_string=2)
    mps = 20
    rows = max(1, round(n_target / (2 * mps)))
    return Farm(rows=rows, strings_per_row=2, modules_per_string=mps, row_pitch=2.0)


def timed(fn, repeats):
    fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats * 1000


def main():
    tmp = Path(tempfile.mkdtemp())
    random_npz(FAST, tmp / "fast.npz")
    random_npz(SLOW, tmp / "slow.npz")
    print(f"numpy {np.__version__}, hidden {HIDDEN}")
    print(f"{'modules':>8} {'fast step (0.5 s cycle)':>24} {'slow step (1 per day)':>22}")
    for n in (2, 120, 480, 720):
        farm = plant(n)
        nodes = sorted(farm.layout, key=lambda s: int(s[1:]))
        edges = [tuple(e) for e in farm.edges]
        fast = gnn.FastModel(tmp / "fast.npz", nodes, farm.layout, edges, farm.strings)
        slow = gnn.SlowModel(tmp / "slow.npz", nodes, farm.layout, edges, farm.strings)
        v = np.array([random.uniform(3, 40) for _ in nodes], dtype=np.float32)
        hv = np.tile(v, (8, 1))
        cur = np.linspace(4, 11, 8, dtype=np.float32)
        t_fast = timed(lambda: fast.step(v), 50)
        t_slow = timed(lambda: slow.step(hv, cur), 20)
        print(f"{len(nodes):>8} {t_fast:>21.1f} ms {t_slow:>19.1f} ms")


if __name__ == "__main__":
    main()
