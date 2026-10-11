"""Live simulation: drives the real SolarMesh twin with a simulated plant and serves the GUI."""

import argparse
import json
import random
import sys
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "software" / "assets"
sys.path.insert(0, str(HERE / ".deps"))
sys.path.insert(0, str(HERE.parent / "software" / "python"))

import link  # noqa: E402
import twin  # noqa: E402
from simulator import Cloud, Simulator  # noqa: E402

PERIOD = 0.5
latest = {"json": b"{}"}


def soiling(rate):
    return lambda dt: max(0.55, 1.0 - rate * dt)


def build(args):
    rng = random.Random(args.seed)
    sim = Simulator(args.cols, args.rows, seed=args.seed)
    nodes = sorted(sim.layout)
    sim.faults = {
        rng.choice(nodes): [(args.fault_at, 0.25)],
        rng.choice(nodes): [(0.0, soiling(0.004))],
    }
    print("faults:", {n: "fault" if not callable(f[0][1]) else "soiling" for n, f in sim.faults.items()}, flush=True)
    return sim, rng


def schedule_clouds(sim, rng, t):
    last_start, last = sim.clouds[-1] if sim.clouds else (-1e9, None)
    if last is None or t >= last_start + last.duration * 0.7:
        cloud = Cloud(rng.uniform(0, 360), rng.uniform(2, 10), rng.uniform(25, 70), sim.layout)
        start = t + rng.uniform(5, 15)
        sim.add_cloud(start, cloud)
        print(f"cloud at t={start:.0f}s heading {cloud.heading_deg:.0f}° {cloud.speed:.1f} m/s R={cloud.radius:.0f} m", flush=True)


def simulate(args):
    sim, rng = build(args)
    state = twin.new_twin()
    t0 = time.monotonic()
    while True:
        tick = time.monotonic()
        t = (tick - t0) * args.speedup
        schedule_clouds(sim, rng, t)
        payload = twin.update(state, link.parse_mesh(sim.mesh_json(t)), t)
        latest["json"] = json.dumps(payload).encode()
        time.sleep(max(0.0, PERIOD - (time.monotonic() - tick)))


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] == "/twin.json":
            body = latest["json"]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=25)
    ap.add_argument("--rows", type=int, default=20)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--speedup", type=float, default=1.0)
    ap.add_argument("--fault-at", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    threading.Thread(target=simulate, args=(args,), daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), partial(Handler, directory=str(ASSETS)))
    print(f"{args.cols * args.rows} modules → http://<board-ip>:{args.port}/?source=/twin.json", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
