"""Synthetic SolarMesh plant. Produces the same get_mesh JSON the STM32 sends.

Simulation-only: never imported by the real app in ../software/python.
Units: 1 grid cell = 1 m, time in seconds.
Assumption: module voltage scales linearly with the light it receives
(the same proxy the optimizer uses). Real open-circuit voltage drops far
less with shade, so check what the STM32 actually measures.
"""

import json
import math
import random

V_REF = 3.0
CURRENT = 0.45
TRANSMITTANCE = 0.3
EDGE_M = 2.0
NOISE_V = 0.01


def build_plant(cols, rows):
    layout, edges = {}, []
    for y in range(rows):
        for x in range(cols):
            node = f"n{y * cols + x + 1}"
            layout[node] = (x, y)
            if x > 0:
                edges.append([f"n{y * cols + x}", node, 1])
            if y > 0:
                edges.append([f"n{(y - 1) * cols + x + 1}", node, 0])
    return layout, edges


class Cloud:
    def __init__(self, heading_deg, speed, radius, layout, transmittance=TRANSMITTANCE):
        xs = [p[0] for p in layout.values()]
        ys = [p[1] for p in layout.values()]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        half_diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) / 2

        self.heading_deg = heading_deg
        self.speed = speed
        self.radius = radius
        self.transmittance = transmittance
        self.vx = speed * math.cos(math.radians(heading_deg))
        self.vy = speed * math.sin(math.radians(heading_deg))
        start = radius + half_diag + EDGE_M + 1
        self.x0 = cx - start * math.cos(math.radians(heading_deg))
        self.y0 = cy - start * math.sin(math.radians(heading_deg))
        self.duration = 2 * start / speed

    def light(self, x, y, t):
        cx, cy = self.x0 + self.vx * t, self.y0 + self.vy * t
        d = math.hypot(x - cx, y - cy) - self.radius
        coverage = min(max(0.5 - d / EDGE_M, 0.0), 1.0)
        return 1.0 - (1.0 - self.transmittance) * coverage


class Simulator:
    def __init__(self, cols, rows, cloud=None, faults=None, seed=0):
        self.layout, self.edges = build_plant(cols, rows)
        self.clouds = [(0.0, cloud)] if cloud else []
        self.faults = faults or {}
        self.rng = random.Random(seed)

    def add_cloud(self, start, cloud):
        self.clouds.append((start, cloud))

    def active_clouds(self, t):
        return [(s, c) for s, c in self.clouds if 0 <= t - s <= c.duration]

    def lights(self, t):
        active = self.active_clouds(t)
        out = {}
        for n, (x, y) in self.layout.items():
            light = 1.0
            for start, cloud in active:
                light *= cloud.light(x, y, t - start)
            out[n] = light
        return out

    def voltages(self, t):
        out = {}
        for node, light in self.lights(t).items():
            factor = 1.0
            for start, f in self.faults.get(node, []):
                if t >= start:
                    factor *= f(t - start) if callable(f) else f
            v = V_REF * light * factor + self.rng.gauss(0, NOISE_V)
            out[node] = round(max(v, 0.0), 4)
        return out

    def mesh_json(self, t):
        lights = self.lights(t)
        return json.dumps({
            "nodes": self.voltages(t),
            "temp": 25.0,
            "current": round(CURRENT * max(lights.values()), 4),
            "edges": self.edges,
            "layout": {n: list(p) for n, p in self.layout.items()},
        })
