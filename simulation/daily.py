"""Daily synthetic data for the SolarMesh SLOW model (days -> weeks)."""

import argparse
import json
import math
import random
import sys
import time
from array import array
from multiprocessing import Pool
from pathlib import Path

from dataset import random_plant
from farm import Farm, ISC_STC, G_CLEAR

HOURS = [9, 10, 11, 12, 13, 14, 15, 16]
SLOW_RAW_CLASSES = ["ok", "soiling", "degradation", "recurring_shade", "fault"]
SOILING_LABEL = 0.05
DEGRADATION_LABEL = 0.05


class DailyFarm(Farm):
    """Farm whose light per module is set per (day, hour) by the daily scenario."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.light = {n: 1.0 for n in self.layout}
        self.shorted = {}
        self.dead = set()

    def substrings(self, node, t):
        g = self.light[node]
        sub = self.shorted.get(node)
        iph = [0.0 if node in self.dead else ISC_STC * G_CLEAR / 1000.0 * g] * 3
        shorted = [i == sub for i in range(3)]
        return iph, shorted, g


def hour_profile(hour):
    return max(0.15, math.sin(math.pi * (hour - 6) / 12))


def build_scenario(rng, farm, days):
    nodes = sorted(farm.layout)
    xs = sorted({x for x, _ in farm.layout.values()})
    ys = sorted({y for _, y in farm.layout.values()})

    rate = {n: 0.0 for n in nodes}
    base = rng.uniform(0.0005, 0.004)
    for n in nodes:
        rate[n] = base * rng.uniform(0.8, 1.2)
    if rng.random() < 0.5:
        edge_rows = set(ys[:rng.randint(1, max(1, len(ys) // 3))])
        boost = rng.uniform(2, 5)
        for n, (x, y) in farm.layout.items():
            if y in edge_rows:
                rate[n] *= boost
    birds = rng.sample(nodes, rng.randint(0, 4))
    for n in birds:
        rate[n] += rng.uniform(0.005, 0.02)

    degr = {n: rng.uniform(0.003, 0.01) / 365 for n in nodes}
    for _ in range(rng.choice([0, 1, 1, 2])):
        string = rng.choice(farm.strings)
        for n in string[: rng.randint(2, max(2, len(string) // 3))]:
            degr[n] = rng.uniform(0.1, 0.4) / 365

    shade = None
    if rng.random() < 0.5:
        h0 = rng.choice(HOURS[:-2])
        width = rng.randint(1, min(3, len(xs)))
        row = rng.randrange(len(ys))
        col = rng.randrange(max(1, len(xs) - width - 3))
        shade = {"start_hour": h0, "hours": rng.randint(2, 3), "row": ys[row], "col": col, "width": width,
                 "transmittance": rng.uniform(0.2, 0.5)}

    faults = []
    for kind in ("diode", "dead"):
        if rng.random() < 0.3:
            faults.append({"kind": kind, "node": rng.choice(nodes), "day": rng.randrange(days // 4, days),
                           "sub": rng.randrange(3)})

    weather = rng.choices(["sunny", "partly", "overcast"], [0.6, 0.25, 0.15], k=days)
    rain = [rng.random() < (0.35 if w == "overcast" else 0.08) for w in weather]
    return {"rate": rate, "degr": degr, "shade": shade, "faults": faults, "weather": weather,
            "rain": rain, "birds": birds, "xs": xs}


def shaded_now(farm, shade, hour, xs):
    if not shade or not (shade["start_hour"] <= hour < shade["start_hour"] + shade["hours"]):
        return set()
    c0 = shade["col"] + (hour - shade["start_hour"])
    cols = set(xs[c0:c0 + shade["width"]])
    return {n for n, (x, y) in farm.layout.items() if y == shade["row"] and x in cols}


def generate(job):
    index, seed, days, out = job
    rng = random.Random(seed)
    plant = random_plant(rng)
    farm = DailyFarm(seed=seed, **plant)
    sc = build_scenario(rng, farm, days)
    nodes = sorted(farm.layout, key=lambda n: int(n[1:]))
    soil = {n: 0.0 for n in nodes}
    ever_shaded = set()
    season0 = rng.uniform(0, 365)

    volts, current, labels = array("f"), array("f"), array("B")
    for d in range(days):
        if sc["rain"][d]:
            wash = rng.uniform(0.7, 1.0)
            for n in nodes:
                soil[n] *= 1 - wash if n not in sc["birds"] else 1 - wash * 0.5
        for n in nodes:
            soil[n] = min(soil[n] + sc["rate"][n], 0.5)
        degr = {n: sc["degr"][n] * d for n in nodes}
        farm.shorted = {f["node"]: f["sub"] for f in sc["faults"] if f["kind"] == "diode" and d >= f["day"]}
        farm.dead = {f["node"] for f in sc["faults"] if f["kind"] == "dead" and d >= f["day"]}
        farm.t_ambient = 15 + 12 * math.sin(2 * math.pi * (d + season0) / 365) + rng.gauss(0, 3)

        for hour in HOURS:
            w = sc["weather"][d]
            sky = 1.0 if w == "sunny" else rng.uniform(0.3, 1.0) if w == "partly" else rng.uniform(0.2, 0.45)
            sun = hour_profile(hour) * sky
            shaded = shaded_now(farm, sc["shade"], hour, sc["xs"])
            ever_shaded |= shaded
            for n in nodes:
                g = sun * (1 - soil[n]) * (1 - degr[n])
                farm.light[n] = g * (sc["shade"]["transmittance"] if n in shaded else 1.0)
            mesh = json.loads(farm.mesh_json(0))
            volts.extend(mesh["nodes"][n] for n in nodes)
            current.append(mesh["current"])

        median_degr = sorted(degr.values())[len(nodes) // 2]
        for n in nodes:
            if n in farm.dead or n in farm.shorted:
                label = "fault"
            elif sc["shade"] and n in ever_shaded:
                label = "recurring_shade"
            elif soil[n] >= SOILING_LABEL:
                label = "soiling"
            elif degr[n] - median_degr >= DEGRADATION_LABEL:
                label = "degradation"
            else:
                label = "ok"
            labels.append(SLOW_RAW_CLASSES.index(label))

    pos = {n: i for i, n in enumerate(nodes)}
    name = f"day_{index:05d}"
    meta = {
        "nodes": nodes,
        "pos": [list(farm.layout[n]) for n in nodes],
        "strings": [[pos[n] for n in s] for s in farm.strings],
        "geo_edges": [[pos[a], pos[b]] for a, b, _ in farm.edges],
        "days": days,
        "hours": HOURS,
        "raw_classes": SLOW_RAW_CLASSES,
        "plant": plant,
        "weather": sc["weather"],
        "rain": sc["rain"],
        "shade": sc["shade"],
        "faults": [{**f, "node": pos[f["node"]]} for f in sc["faults"]],
        "seed": seed,
    }
    (out / f"{name}.json").write_text(json.dumps(meta))
    for suffix, arr in (("volts.f32", volts), ("current.f32", current), ("labels.u8", labels)):
        with open(out / f"{name}.{suffix}", "wb") as f:
            arr.tofile(f)
    return name, len(nodes), [labels.count(i) for i in range(len(SLOW_RAW_CLASSES))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "ml" / "daily"))
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if sys.byteorder != "little":
        sys.exit("expects a little-endian machine")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(i, args.seed + i, args.days, out) for i in range(args.start, args.start + args.episodes)]
    totals = [0] * len(SLOW_RAW_CLASSES)
    started = time.time()
    with Pool(args.workers) as pool:
        for k, (name, n, counts) in enumerate(pool.imap_unordered(generate, jobs), 1):
            totals = [a + b for a, b in zip(totals, counts)]
            print(f"{name}: {n:4d} modules ({k}/{len(jobs)}, {time.time() - started:.0f} s)", flush=True)
    total = sum(totals)
    print("label share:", ", ".join(f"{c} {v / total:.2%}" for c, v in zip(SLOW_RAW_CLASSES, totals)))


if __name__ == "__main__":
    main()
