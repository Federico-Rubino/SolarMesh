"""Final testbench: the REAL SolarMesh software (../software/python) against a
realistic 1 MWp farm (farm.py), with ground truth for every module and step.

Usage:  python3 testbench.py [--workers 4] [--duration 600] [--quick]
Output: results/testbench.json and results/TESTBENCH.md
"""

import argparse
import json
import math
import random
import statistics
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".deps"))
sys.path.insert(0, str(HERE.parent / "software" / "python"))

import diagnosis  # noqa: E402
import link  # noqa: E402
import twin  # noqa: E402
from farm import Farm  # noqa: E402

DT = 0.5
ACTIONS = ["none", "clean", "inspect", "technician"]
TRUTH_ACTION = {
    "ok": "none", "shade": "none", "soiling": "clean", "debris": "clean",
    "obstruction": "inspect", "diode": "technician", "dead": "technician",
}
CONFIGS = {
    "demo": {"PERSIST_S": 20.0, "SUDDEN_S": 3.0},
    "field": {"PERSIST_S": 300.0, "SUDDEN_S": 3.0},
}
LEAD_BUCKETS = [(0, 5), (5, 15), (15, 60)]


# ---------- scenarios ----------

def add_cumulus(farm, rng, heading, speed, duration):
    clouds, start = [], 20.0
    while start < duration - 60:
        c = farm.add_cloud(
            start,
            heading + rng.uniform(-10, 10),
            speed * rng.uniform(0.85, 1.15),
            rng.uniform(100, 600),
            rng.uniform(0.25, 0.5),
        )
        clouds.append(c)
        start += rng.uniform(80, 160)
    return clouds


def add_faults(farm, rng, taken):
    nodes = [n for n in sorted(farm.layout) if n not in taken]
    picks = rng.sample(nodes, 3)
    farm.add_event("diode", [picks[0]], 60.0, sub=rng.randrange(3))
    farm.add_event("dead", [picks[1]], 90.0)
    farm.add_event("debris", [picks[2]], 120.0, sub=rng.randrange(3), transmittance=0.3)
    taken.update(picks)


def patch(farm, row, col, rows, cols):
    per_row = len(farm.strings[0]) * 2
    return [f"m{r * per_row + c + 1}" for r in range(row, row + rows) for c in range(col, col + cols)]


def build(name, seed, duration):
    rng = random.Random(seed)
    farm = Farm(seed=seed)
    taken = set()
    if name.startswith("cumulus"):
        heading = float(name.split("_")[1])
        add_cumulus(farm, rng, heading, rng.uniform(5, 15), duration)
        add_faults(farm, rng, taken)
    elif name == "overcast":
        farm.set_overcast(60.0, 120.0, 0.3)
        add_faults(farm, rng, taken)
    elif name == "tree_shadow":
        nodes = patch(farm, 10, 20, 2, 3)
        farm.add_event("obstruction", nodes, 30.0, transmittance=0.3)
        taken.update(nodes)
        add_cumulus(farm, rng, rng.uniform(0, 360), rng.uniform(5, 15), duration)
    elif name == "local_soiling":
        nodes = patch(farm, 0, 0, 2, 52)
        farm.add_event("soiling", nodes, 0.0, rate=0.3 / duration, max_loss=0.3)
        taken.update(nodes)
    elif name == "clear_sky":
        pass
    return farm, taken


SCENARIOS = [f"cumulus_{h}" for h in (0, 60, 120, 180, 240, 300)] + [
    "overcast", "tree_shadow", "local_soiling", "clear_sky",
]


# ---------- one run ----------

def run(job):
    name, config, seed, duration = job
    for k, v in CONFIGS[config].items():
        setattr(diagnosis, k, v)
    farm, _ = build(name, seed, duration)
    state = twin.new_twin()

    confusion = Counter()
    tech_alerts = Counter()
    first_action = {}
    events = {n: e for e in farm.events for n in e["nodes"] if e["kind"] in ("diode", "dead", "debris", "obstruction", "soiling")}
    onset_truth, predictions = {}, {}
    radar_err = []
    compute = []

    t = 0.0
    while t < duration:
        mesh = link.parse_mesh(farm.mesh_json(t))
        tick = time.perf_counter()
        payload = twin.update(state, mesh, t)
        compute.append(time.perf_counter() - tick)
        truth = farm.truth(t)

        for p in payload["panels"]:
            node = p["id"]
            ta = TRUTH_ACTION[truth[node]]
            confusion[(ta, p["action"])] += 1
            if p["action"] == "technician" and truth[node] in ("ok", "shade"):
                tech_alerts[node] += 1
            if node in events and node not in first_action and p["action"] == TRUTH_ACTION[events[node]["kind"]]:
                first_action[node] = t

        for node, (x, y) in farm.layout.items():
            if node not in onset_truth and node not in events:
                if any(c.factor(x, y, t) < 0.85 for c in farm.clouds):
                    onset_truth[node] = t

        c = payload["cloud"]
        if c["active"]:
            active = [cl for cl in farm.clouds if 0 <= t - cl.start <= cl.duration]
            if active:
                ref = min(active, key=lambda cl: abs(t - cl.start - cl.duration / 2))
                herr = abs((c["heading_deg"] - ref.heading_deg + 180) % 360 - 180)
                radar_err.append((herr, abs(c["speed_mps"] - ref.speed) / ref.speed))
            for node, eta in c["eta"].items():
                lst = predictions.setdefault(node, [])
                if len(lst) < 40:
                    lst.append((t, t + eta))
        t += DT

    eta_err = {b: [] for b in LEAD_BUCKETS}
    warned = 0
    for node, t_true in onset_truth.items():
        early = [(m, a) for m, a in predictions.get(node, []) if m < t_true]
        if early:
            warned += 1
        for made, arr in early:
            lead = t_true - made
            for lo, hi in LEAD_BUCKETS:
                if lo <= lead < hi:
                    eta_err[(lo, hi)].append(abs(arr - t_true))

    final = {p["id"]: p["action"] for p in payload["panels"]}
    event_rows = []
    for node, e in sorted(events.items()):
        expected = TRUTH_ACTION[e["kind"]]
        event_rows.append({
            "node": node, "kind": e["kind"], "expected": expected, "final": final[node],
            "correct": final[node] == expected,
            "delay_s": (first_action[node] - e["start"]) if node in first_action else None,
        })

    return {
        "scenario": name, "config": config, "seed": seed, "duration": duration,
        "modules": len(farm.layout), "clouds": len(farm.clouds),
        "confusion": {f"{a}|{b}": n for (a, b), n in confusion.items()},
        "events": event_rows,
        "false_technician_modules": len(tech_alerts),
        "false_technician_module_minutes": sum(tech_alerts.values()) * DT / 60,
        "radar_heading_err": statistics.median([h for h, _ in radar_err]) if radar_err else None,
        "radar_speed_err": statistics.median([s for _, s in radar_err]) if radar_err else None,
        "shaded_modules": len(onset_truth),
        "warned_pct": warned / len(onset_truth) if onset_truth else None,
        "eta_mae": {f"{lo}-{hi}s": (statistics.mean(v) if v else None) for (lo, hi), v in eta_err.items()},
        "eta_n": {f"{lo}-{hi}s": len(v) for (lo, hi), v in eta_err.items()},
        "compute_ms": {"mean": 1000 * statistics.mean(compute), "p95": 1000 * sorted(compute)[int(0.95 * len(compute))]},
    }


# ---------- report ----------

def pct(v):
    return "–" if v is None else f"{v:.0%}"


def num(v, spec=".2f"):
    return "–" if v is None else format(v, spec)


def summarize(results, config):
    rs = [r for r in results if r["config"] == config]
    conf = Counter()
    for r in rs:
        for k, n in r["confusion"].items():
            conf[tuple(k.split("|"))] += n
    events = [e for r in rs for e in r["events"]]
    by_kind = {}
    for e in events:
        by_kind.setdefault(e["kind"], []).append(e)
    tech_truth = [e for e in events if e["expected"] == "technician"]
    tech_hit = [e for e in tech_truth if e["final"] == "technician"]
    tech_called = [e for e in events if e["final"] == "technician"]

    lines = [f"## Configuration `{config}` (PERSIST_S = {CONFIGS[config]['PERSIST_S']:.0f} s)", ""]
    lines += ["### Human-needed predictor", ""]
    lines += ["| Injected event | Expected action | Correct at end of run | Median time to correct action |", "|---|---|---|---|"]
    for kind, es in sorted(by_kind.items()):
        ok = sum(e["correct"] for e in es)
        delays = [e["delay_s"] for e in es if e["delay_s"] is not None]
        lines.append(f"| {kind} | {es[0]['expected']} | {ok}/{len(es)} | {num(statistics.median(delays), '.0f') + ' s' if delays else '–'} |")
    lines.append("")
    total_minutes = sum(r["duration"] for r in rs) / 60
    false_mod = sum(r["false_technician_modules"] for r in rs)
    lines += [
        f"- **Technician recall** (real faults → technician): {len(tech_hit)}/{len(tech_truth)}",
        f"- **Technician precision** (technician calls that were real faults, at end of run): "
        f"{sum(e['expected'] == 'technician' for e in tech_called)}/{len(tech_called)} injected-event modules",
        f"- **False technician alerts on healthy modules** (truth ok or cloud shade): {false_mod} module(s) over "
        f"{total_minutes:.0f} simulated minutes × {rs[0]['modules']} modules",
        "",
        "Per-step action confusion (rows = truth, columns = system), all modules × all steps:",
        "",
        "| truth \\ system | " + " | ".join(ACTIONS) + " | recall |",
        "|---|" + "---|" * (len(ACTIONS) + 1),
    ]
    for a in ACTIONS:
        row = [conf[(a, b)] for b in ACTIONS]
        tot = sum(row)
        lines.append(f"| {a} | " + " | ".join(f"{n:,}" for n in row) + f" | {pct(conf[(a, a)] / tot) if tot else '–'} |")
    lines.append("")

    cum = [r for r in rs if r["scenario"].startswith("cumulus") or r["scenario"] == "tree_shadow"]
    lines += ["### Cloud tracker", "", "| Scenario | Clouds | Heading err (median) | Speed err (median) | Modules warned before shade | ETA MAE <5 s | 5–15 s | 15–60 s |", "|---|---|---|---|---|---|---|---|"]
    for r in cum:
        e = r["eta_mae"]
        lines.append(f"| {r['scenario']} | {r['clouds']} | {num(r['radar_heading_err'], '.1f')}° | {pct(r['radar_speed_err'])} | "
                     f"{pct(r['warned_pct'])} | {num(e['0-5s'])} s | {num(e['5-15s'])} s | {num(e['15-60s'])} s |")
    lines += ["", "### Compute (real `twin.update` on the UNO Q)", ""]
    lines.append(f"- mean {statistics.mean(r['compute_ms']['mean'] for r in rs):.0f} ms, "
                 f"worst p95 {max(r['compute_ms']['p95'] for r in rs):.0f} ms per 0.5 s cycle")
    lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--quick", action="store_true", help="2 scenarios, demo config only")
    args = ap.parse_args()

    scenarios = ["cumulus_0", "clear_sky"] if args.quick else SCENARIOS
    configs = ["demo"] if args.quick else list(CONFIGS)
    jobs = [(s, c, args.seed + i, args.duration) for c in configs for i, s in enumerate(scenarios)]
    print(f"{len(jobs)} runs × {args.duration:.0f} s simulated, dt={DT}s, 1820 modules", flush=True)

    started = time.time()
    with Pool(args.workers) as pool:
        results = []
        for r in pool.imap_unordered(run, jobs):
            results.append(r)
            print(f"  done {r['config']:<5} {r['scenario']:<14} ({len(results)}/{len(jobs)}, {time.time() - started:.0f} s)", flush=True)

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "testbench.json").write_text(json.dumps(results, indent=1))
    order = {s: i for i, s in enumerate(SCENARIOS)}
    results.sort(key=lambda r: order[r["scenario"]])
    lines = ["# SolarMesh testbench results", "",
             f"Realistic 1 MWp farm: 1,820 × 550 W modules, 70 strings of 26, 35 rows (59 × 172 m). "
             f"{len(scenarios)} scenarios × {args.duration:.0f} s, sampled every {DT} s. Seed {args.seed}.", ""]
    for c in configs:
        lines += summarize(results, c)
    (out / "TESTBENCH.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
