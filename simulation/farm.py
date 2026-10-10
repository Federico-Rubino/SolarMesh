"""Realistic 1 MWp fixed-tilt PV farm for benchmarking SolarMesh.

Simulation-only: never imported by the real app in ../software/python.

Physics, per module:
  - 550 W module, 3 bypass-diode substrings of 24 cells (single-diode model per substring)
  - strings of 26 modules in series; each string's inverter MPPT picks the
    current that maximises string power, so weak substrings get bypassed
    (their voltage goes to about -0.7 V) - this is how shading really shows up
    in per-module voltage
  - cloud shadows with sharp edges, partial transmittance and edge brightening
  - events: bypass-diode short, dead module, debris, static obstruction, soiling

Units: metres, seconds, volts, amps. Parameters and sources: ../docs/simulation-model.md
"""

import json
import math
import random

# Module (550 W class, 2278 x 1134 mm)
MODULE_W = 1.134
MODULE_GAP = 0.02
ISC_STC = 13.9
NVT_SUB = 0.74            # n * cells_per_substring * Vt at 25 °C
I0_SUB = 2.6e-9           # gives Voc ~ 16.6 V per substring (49.7 V per module)
RS_SUB = 0.035            # gives Vmp ~ 42 V per module
BETA_V = -0.0026          # voltage temperature coefficient (1/°C)
V_BYPASS = -0.7
NOCT = 45.0

# Plant
MODULES_PER_STRING = 26
STRINGS_PER_ROW = 2
ROWS = 35
ROW_PITCH = 5.06          # 2.278 m collector / GCR 0.45

# Site / weather
G_CLEAR = 900.0
T_AMBIENT = 25.0
EDGE_M = 50.0
EDGE_BOOST = 0.10
NOISE_V = 0.05


def substring_voltage(iph, current, temp):
    if current >= iph:
        return V_BYPASS
    v = NVT_SUB * math.log((iph - current) / I0_SUB + 1.0) - current * RS_SUB
    return v * (1.0 + BETA_V * (temp - 25.0))


class Cloud:
    def __init__(self, start, heading_deg, speed, radius, transmittance, center):
        self.start = start
        self.heading_deg = heading_deg
        self.speed = speed
        self.radius = radius
        self.transmittance = transmittance
        h = math.radians(heading_deg)
        self.vx, self.vy = speed * math.cos(h), speed * math.sin(h)
        lead = radius + EDGE_M + 200.0
        self.x0 = center[0] - lead * math.cos(h)
        self.y0 = center[1] - lead * math.sin(h)
        self.duration = 2 * lead / speed

    def factor(self, x, y, t):
        dt = t - self.start
        if dt < 0 or dt > self.duration:
            return 1.0
        d = math.hypot(x - (self.x0 + self.vx * dt), y - (self.y0 + self.vy * dt)) - self.radius
        if d >= EDGE_M:
            return 1.0
        if d <= 0:
            return self.transmittance
        ramp = d / EDGE_M
        boost = EDGE_BOOST * math.sin(math.pi * ramp)
        return self.transmittance + (1.0 - self.transmittance) * ramp + boost


class Farm:
    def __init__(self, rows=ROWS, strings_per_row=STRINGS_PER_ROW, seed=0,
                 modules_per_string=MODULES_PER_STRING, row_pitch=ROW_PITCH,
                 noise=NOISE_V, tolerance=0.0):
        self.rng = random.Random(seed)
        self.noise = noise
        self.layout, self.edges, self.strings = {}, [], []
        cols = strings_per_row * modules_per_string
        for r in range(rows):
            for c in range(cols):
                node = f"m{r * cols + c + 1}"
                self.layout[node] = (round(c * (MODULE_W + MODULE_GAP), 3), round(r * row_pitch, 3))
                if c > 0:
                    wired = int(c % modules_per_string != 0)
                    self.edges.append([f"m{r * cols + c}", node, wired])
                if r > 0:
                    self.edges.append([f"m{(r - 1) * cols + c + 1}", node, 0])
            for s in range(strings_per_row):
                base = r * cols + s * modules_per_string
                self.strings.append([f"m{base + i + 1}" for i in range(modules_per_string)])
        xs = [p[0] for p in self.layout.values()]
        ys = [p[1] for p in self.layout.values()]
        self.center = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
        self.gain = {n: 1.0 + self.rng.gauss(0, tolerance) for n in self.layout}
        self.clouds = []
        self.overcast = None
        self.events = []

    # ---------- scenario building ----------

    def add_cloud(self, start, heading_deg, speed, radius, transmittance):
        cloud = Cloud(start, heading_deg, speed, radius, transmittance, self.center)
        self.clouds.append(cloud)
        return cloud

    def set_overcast(self, start, ramp_s, level):
        self.overcast = (start, ramp_s, level)

    def add_event(self, kind, nodes, start, end=None, **params):
        self.events.append({"kind": kind, "nodes": set(nodes), "start": start, "end": end, **params})

    # ---------- truth ----------

    def sky(self, x, y, t):
        f = 1.0
        for cloud in self.clouds:
            f *= cloud.factor(x, y, t)
        if self.overcast:
            start, ramp, level = self.overcast
            if t >= start:
                f *= 1.0 - (1.0 - level) * min((t - start) / ramp, 1.0)
        return f

    def active_events(self, node, t):
        return [e for e in self.events
                if node in e["nodes"] and t >= e["start"] and (e["end"] is None or t < e["end"])]

    def truth(self, t):
        labels = {}
        for node, (x, y) in self.layout.items():
            kinds = [e["kind"] for e in self.active_events(node, t)]
            for k in ("dead", "diode", "debris", "obstruction", "soiling"):
                if k in kinds:
                    labels[node] = k
                    break
            else:
                labels[node] = "shade" if self.sky(x, y, t) < 0.85 else "ok"
        return labels

    # ---------- electrical model ----------

    def substrings(self, node, t):
        x, y = self.layout[node]
        light = self.sky(x, y, t)
        g = [light, light, light]
        shorted = [False, False, False]
        dead = False
        for e in self.active_events(node, t):
            age = t - e["start"]
            if e["kind"] == "soiling":
                loss = min(e["rate"] * age, e.get("max_loss", 0.5))
                g = [v * (1 - loss) for v in g]
            elif e["kind"] == "debris":
                g[e.get("sub", 0)] *= e.get("transmittance", 0.3)
            elif e["kind"] == "obstruction":
                g = [v * e.get("transmittance", 0.3) for v in g]
            elif e["kind"] == "diode":
                shorted[e.get("sub", 1)] = True
            elif e["kind"] == "dead":
                dead = True
        iph = [0.0 if dead else ISC_STC * G_CLEAR / 1000.0 * v for v in g]
        return iph, shorted, light

    def string_voltages(self, nodes, t):
        subs = []
        lights = []
        for node in nodes:
            iph, shorted, light = self.substrings(node, t)
            subs.append((iph, shorted))
            lights.append(light)
        temp = T_AMBIENT + (NOCT - 20.0) / 800.0 * G_CLEAR * (sum(lights) / len(lights))

        levels = sorted({round(i, 2) for iph, sh in subs for i, s in zip(iph, sh) if not s and i > 0})
        best_i, best_p = 0.0, -1.0
        for level in levels:
            for frac in (0.90, 0.94, 0.97):
                current = level * frac
                v = sum(
                    0.0 if s else substring_voltage(i, current, temp)
                    for iph, sh in subs for i, s in zip(iph, sh)
                )
                if current * v > best_p:
                    best_i, best_p = current, current * v

        volts = {}
        for node, (iph, sh) in zip(nodes, subs):
            v = sum(0.0 if s else substring_voltage(i, best_i, temp) for i, s in zip(iph, sh))
            volts[node] = v * self.gain[node] + self.rng.gauss(0, self.noise)
        return volts, best_i, temp

    def mesh_json(self, t):
        nodes, currents, temps = {}, [], []
        for string in self.strings:
            volts, current, temp = self.string_voltages(string, t)
            nodes.update({n: round(v, 3) for n, v in volts.items()})
            currents.append(current)
            temps.append(temp)
        return json.dumps({
            "nodes": nodes,
            "temp": round(sum(temps) / len(temps), 1),
            "current": round(sum(currents) / len(currents), 3),
            "edges": self.edges,
            "layout": {n: list(p) for n, p in self.layout.items()},
        })
