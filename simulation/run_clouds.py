"""Run the REAL SolarMesh twin (../software/python) against simulated clouds and score it.

Usage:  python3 run_clouds.py [--cols 25 --rows 20 --scenarios 12 --seed 1]
"""

import argparse
import json
import math
import random
import statistics
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".deps"))
sys.path.insert(0, str(HERE.parent / "software" / "python"))

import diagnosis  # noqa: E402
import link  # noqa: E402
import twin  # noqa: E402
from simulator import Cloud, Simulator  # noqa: E402

DT = 0.2
MIN_DURATION = 40.0
FAULT_AT = 3.0
LEAD_BUCKETS = [(0, 5), (5, 15), (15, 60)]


def angle_error(a, b):
    return abs((a - b + 180) % 360 - 180)


def make_scenarios(n, cols, rows, seed):
    rng = random.Random(seed)
    layout = Simulator(cols, rows).layout
    scenarios = []
    for i in range(n):
        heading = i * 360 / n + rng.uniform(-10, 10)
        speed = rng.choice([2, 4, 6, 8, 10, 12]) * rng.uniform(0.9, 1.1)
        radius = rng.uniform(30, 80)
        fault = rng.choice(sorted(layout))
        scenarios.append({
            "id": i, "heading": round(heading % 360, 1), "speed": round(speed, 2),
            "radius": round(radius, 1), "fault": fault, "cols": cols, "rows": rows, "seed": seed + i,
        })
    return scenarios


def run(sc):
    layout_sim = Simulator(sc["cols"], sc["rows"]).layout
    cloud = Cloud(sc["heading"], sc["speed"], sc["radius"], layout_sim)
    sim = Simulator(sc["cols"], sc["rows"], cloud, {sc["fault"]: [(FAULT_AT, 0.25)]}, seed=sc["seed"])
    state = twin.new_twin()

    truth, predictions = {}, {}
    speeds, headings = [], []
    false_tech, false_inspect = set(), set()
    fault_detected = None
    steps = 0
    started = time.perf_counter()

    duration = max(cloud.duration, MIN_DURATION)
    t = 0.0
    while t < duration:
        payload = twin.update(state, link.parse_mesh(sim.mesh_json(t)), t)
        steps += 1

        for node, light in sim.lights(t).items():
            if node not in truth and light < diagnosis.ANOMALY:
                truth[node] = t

        c = payload["cloud"]
        if c["active"]:
            speeds.append(c["speed_mps"])
            headings.append(c["heading_deg"])
            for node, eta in c["eta"].items():
                predictions.setdefault(node, []).append((t, t + eta))

        for node in payload["technician"]["modules"]:
            if node == sc["fault"]:
                fault_detected = fault_detected or t
            else:
                false_tech.add(node)
        for p in payload["panels"]:
            if p["action"] == "inspect" and p["id"] != sc["fault"]:
                false_inspect.add(p["id"])
        t += DT

    errors = {b: [] for b in LEAD_BUCKETS}
    warned, leads = 0, []
    for node, t_true in truth.items():
        if node == sc["fault"]:
            continue
        early = [(made, arr) for made, arr in predictions.get(node, []) if made < t_true]
        if early:
            warned += 1
            leads.append(t_true - early[0][0])
        for made, arr in early:
            lead = t_true - made
            for lo, hi in LEAD_BUCKETS:
                if lo <= lead < hi:
                    errors[(lo, hi)].append(abs(arr - t_true))

    shaded = len([n for n in truth if n != sc["fault"]])
    heading_est = None
    if headings:
        sx = sum(math.cos(math.radians(h)) for h in headings)
        sy = sum(math.sin(math.radians(h)) for h in headings)
        heading_est = math.degrees(math.atan2(sy, sx)) % 360

    return {
        **sc,
        "steps": steps,
        "sec_per_step": (time.perf_counter() - started) / steps,
        "speed_est": statistics.median(speeds) if speeds else None,
        "heading_est": heading_est,
        "heading_err": angle_error(heading_est, sc["heading"]) if heading_est is not None else None,
        "shaded": shaded,
        "warned_pct": warned / shaded if shaded else None,
        "lead_median_s": statistics.median(leads) if leads else None,
        "eta_mae": {f"{lo}-{hi}s": (statistics.mean(e) if e else None) for (lo, hi), e in errors.items()},
        "fault_detect_s": (fault_detected - FAULT_AT) if fault_detected else None,
        "false_technician": sorted(false_tech),
        "false_inspect": len(false_inspect),
    }


def fmt(v, spec):
    return "–" if v is None else format(v, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=25)
    ap.add_argument("--rows", type=int, default=20)
    ap.add_argument("--scenarios", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    scenarios = make_scenarios(args.scenarios, args.cols, args.rows, args.seed)
    print(f"Plant {args.cols}x{args.rows} = {args.cols * args.rows} modules, {len(scenarios)} clouds, dt={DT}s", flush=True)
    with Pool(args.workers) as pool:
        results = pool.map(run, scenarios)

    print(f"\n{'#':>2} {'head':>5} {'est':>5} {'err°':>5} {'v m/s':>6} {'est':>6} {'R m':>5} "
          f"{'warned':>7} {'lead s':>6} {'MAE<5s':>7} {'5-15s':>6} {'15-60s':>7} {'fault s':>7} {'falseT':>6} {'inspect':>7}")
    for r in results:
        e = r["eta_mae"]
        print(f"{r['id']:>2} {r['heading']:>5.0f} {fmt(r['heading_est'], '5.0f')} {fmt(r['heading_err'], '5.1f')} "
              f"{r['speed']:>6.1f} {fmt(r['speed_est'], '6.2f')} {r['radius']:>5.0f} "
              f"{fmt(r['warned_pct'], '7.0%')} {fmt(r['lead_median_s'], '6.1f')} "
              f"{fmt(e['0-5s'], '7.2f')} {fmt(e['5-15s'], '6.2f')} {fmt(e['15-60s'], '7.2f')} "
              f"{fmt(r['fault_detect_s'], '7.1f')} {len(r['false_technician']):>6} {r['false_inspect']:>7}")

    def avg(key, sub=None):
        vals = [(r[key][sub] if sub else r[key]) for r in results]
        vals = [v for v in vals if v is not None]
        return statistics.mean(vals) if vals else None

    speed_err = [abs(r["speed_est"] - r["speed"]) / r["speed"] for r in results if r["speed_est"]]
    print("\nSummary")
    print(f"  heading error (mean)        {fmt(avg('heading_err'), '.1f')}°")
    print(f"  speed error (mean)          {fmt(statistics.mean(speed_err) if speed_err else None, '.1%')}")
    print(f"  modules warned before shade {fmt(avg('warned_pct'), '.0%')}  (median lead {fmt(avg('lead_median_s'), '.1f')} s)")
    for b in ("0-5s", "5-15s", "15-60s"):
        print(f"  ETA error, lead {b:<6}      {fmt(avg('eta_mae', b), '.2f')} s")
    print(f"  fault → technician after    {fmt(avg('fault_detect_s'), '.1f')} s")
    print(f"  false technician alerts     {sum(len(r['false_technician']) for r in results)}")
    print(f"  false 'inspect' (cloud)     {sum(r['false_inspect'] for r in results)} module-scenarios")
    print(f"  compute                     {fmt(avg('sec_per_step'), '.3f')} s per step")

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    path = out / f"clouds_{args.cols}x{args.rows}_seed{args.seed}.json"
    path.write_text(json.dumps(results, indent=1))
    print(f"\nSaved {path.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
