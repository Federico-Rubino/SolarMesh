"""Synthetic training data for the SolarMesh graph model (sequence of graphs)."""

import argparse
import json
import random
import sys
import time
from array import array
from multiprocessing import Pool
from pathlib import Path

from farm import Farm

DT = 0.5
RAW_CLASSES = ["ok", "shade", "soiling", "obstruction", "debris", "diode", "dead"]


MIN_MODULES, MAX_MODULES = 250, 700


def random_plant(rng):
    """Commercial / hospital / neighbourhood rooftops: 250-700 modules."""
    while True:
        mps = rng.randint(14, 22)
        spr = rng.randint(1, 3)
        rows = round(rng.randint(MIN_MODULES, MAX_MODULES) / (mps * spr))
        if rows >= 2 and MIN_MODULES <= rows * spr * mps <= MAX_MODULES:
            break
    return {
        "rows": rows,
        "strings_per_row": spr,
        "modules_per_string": mps,
        "row_pitch": round(rng.uniform(1.2, 3.5), 2),
        "blocks": min(rows, rng.choice([1, 1, 2, 3])),
        "block_gap": round(rng.uniform(6.0, 25.0), 1),
        "noise": round(rng.uniform(0.02, 0.1), 3),
        "tolerance": round(rng.uniform(0.005, 0.015), 4),
    }


def add_weather(farm, rng, duration):
    kind = rng.choices(["cumulus", "overcast", "clear"], [0.7, 0.15, 0.15])[0]
    if kind == "cumulus":
        heading, speed = rng.uniform(0, 360), rng.uniform(3, 15)
        start = rng.uniform(0, 40)
        while start < duration - 20:
            farm.add_cloud(start, heading + rng.uniform(-15, 15), speed * rng.uniform(0.8, 1.2),
                           rng.uniform(50, 600), rng.uniform(0.2, 0.6))
            start += rng.uniform(40, 150)
    elif kind == "overcast":
        farm.set_overcast(rng.uniform(10, duration / 2), rng.uniform(30, 180), rng.uniform(0.2, 0.5))
    return kind


def patch(farm, rng):
    rows = sorted({y for _, y in farm.layout.values()})
    cols = sorted({x for x, _ in farm.layout.values()})
    h, w = rng.randint(1, min(3, len(rows))), rng.randint(1, min(4, len(cols)))
    r0, c0 = rng.randrange(len(rows) - h + 1), rng.randrange(len(cols) - w + 1)
    keep = {(x, y) for x in cols[c0:c0 + w] for y in rows[r0:r0 + h]}
    return [n for n, p in farm.layout.items() if p in keep]


def add_events(farm, rng, duration):
    nodes = sorted(farm.layout)
    start = lambda: rng.uniform(10, duration - 30)  # noqa: E731
    for _ in range(rng.choice([0, 1, 1, 2])):
        farm.add_event("diode", [rng.choice(nodes)], start(), sub=rng.randrange(3))
    for _ in range(rng.choice([0, 0, 1])):
        farm.add_event("dead", [rng.choice(nodes)], start())
    for _ in range(rng.choice([0, 1, 2])):
        s = start()
        end = s + rng.uniform(20, 200) if rng.random() < 0.3 else None
        farm.add_event("debris", [rng.choice(nodes)], s, end, sub=rng.randrange(3),
                       transmittance=rng.uniform(0.1, 0.5))
    if rng.random() < 0.4:
        farm.add_event("obstruction", patch(farm, rng), start(), transmittance=rng.uniform(0.15, 0.5))
    if rng.random() < 0.4:
        if rng.random() < 0.5:
            targets = rng.choice(farm.strings)
        else:
            targets = rng.sample(nodes, min(len(nodes), rng.randint(1, 8)))
        farm.add_event("soiling", targets, rng.uniform(0, duration / 2),
                       rate=rng.uniform(0.1, 0.4) / duration, max_loss=rng.uniform(0.1, 0.4))


def generate(job):
    index, seed, duration, out = job
    rng = random.Random(seed)
    plant = random_plant(rng)
    farm = Farm(seed=seed, **plant)
    weather = add_weather(farm, rng, duration)
    add_events(farm, rng, duration)

    ids = sorted(farm.layout, key=lambda n: int(n[1:]))
    pos = {n: i for i, n in enumerate(ids)}
    volts, labels = array("f"), array("B")
    frames = int(duration / DT)
    for k in range(frames):
        t = k * DT
        mesh = json.loads(farm.mesh_json(t))
        truth = farm.truth(t)
        volts.extend(mesh["nodes"][n] for n in ids)
        labels.extend(RAW_CLASSES.index(truth[n]) for n in ids)

    name = f"ep_{index:05d}"
    meta = {
        "nodes": ids,
        "pos": [list(farm.layout[n]) for n in ids],
        "strings": [[pos[n] for n in s] for s in farm.strings],
        "geo_edges": [[pos[a], pos[b]] for a, b, _ in farm.edges],
        "dt": DT,
        "frames": frames,
        "raw_classes": RAW_CLASSES,
        "plant": plant,
        "weather": weather,
        "events": [{"kind": e["kind"], "nodes": sorted(pos[n] for n in e["nodes"]),
                    "start": e["start"], "end": e["end"]} for e in farm.events],
        "seed": seed,
    }
    (out / f"{name}.json").write_text(json.dumps(meta))
    with open(out / f"{name}.volts.f32", "wb") as f:
        volts.tofile(f)
    with open(out / f"{name}.labels.u8", "wb") as f:
        labels.tofile(f)
    counts = [labels.count(i) for i in range(len(RAW_CLASSES))]
    return name, len(ids), weather, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "ml" / "data"))
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--start", type=int, default=0, help="first episode index (to extend a dataset)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if sys.byteorder != "little":
        sys.exit("expects a little-endian machine")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(i, args.seed + i, args.duration, out) for i in range(args.start, args.start + args.episodes)]
    totals = [0] * len(RAW_CLASSES)
    started = time.time()
    with Pool(args.workers) as pool:
        for k, (name, n, weather, counts) in enumerate(pool.imap_unordered(generate, jobs), 1):
            totals = [a + b for a, b in zip(totals, counts)]
            print(f"{name}: {n:4d} modules, {weather:<8} ({k}/{len(jobs)}, {time.time() - started:.0f} s)", flush=True)
    total = sum(totals)
    print("label share:", ", ".join(f"{c} {v / total:.2%}" for c, v in zip(RAW_CLASSES, totals)))


if __name__ == "__main__":
    main()
