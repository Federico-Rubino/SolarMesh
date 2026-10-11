import math
import time
from pathlib import Path

import diagnosis
import radar
import topology
from optimizer import optimize

try:
    import gnn
except ImportError:  # numpy missing (e.g. host Python): run with rules only
    gnn = None

PITCH_M = 1.0
MODEL_PATH = Path(__file__).resolve().parent / "models" / "gnn.pt"
AI_ACTIONS = {
    "ok": "none", "shade": "none", "soiling": "clean", "obstruction": "inspect",
    "debris_or_diode": "inspect", "dead": "technician",
}

ALERT_TEXT = {
    "technician": "send technician",
    "clean": "schedule cleaning",
    "inspect": "inspect for obstruction",
}


def new_twin():
    return {"trackers": {}, "issues": []}


def grid_layout(graph):
    rows = topology.strings(graph)
    placed = {n for row in rows for n in row}
    rest = sorted(n for n in graph.nodes if n not in placed)
    if rest:
        cols = max(1, math.ceil(math.sqrt(len(rest))))
        rows += [rest[i:i + cols] for i in range(0, len(rest), cols)]
    return {node: (x, y) for y, row in enumerate(rows) for x, node in enumerate(row)}


def positions(mesh, graph):
    layout = mesh["layout"]
    if layout and all(n in layout for n in graph.nodes):
        return layout
    return grid_layout(graph)


def module_alerts(diag, ages):
    alerts = []
    for node, d in sorted(diag.items()):
        text = ALERT_TEXT.get(d["action"])
        if text:
            alerts.append(f"{node}: {d['diagnosis']} for {ages[node]:.0f} s → {text}")
    return alerts


def cloud_alert(cloud):
    if not cloud["active"] or cloud["next_eta"] is None:
        return []
    return [
        f"Cloud approaching: {len(cloud['eta'])} modules shaded within "
        f"{max(cloud['eta'].values()):.0f} s, first in {cloud['next_eta']:.1f} s "
        f"({cloud['speed_mps']:.1f} m/s, heading {cloud['heading_deg']:.0f}°)"
    ]


def analyse_topology(twin, mesh):
    key = (tuple(sorted(mesh["nodes"])), tuple(mesh["edges"]), tuple(sorted(mesh["layout"].items())))
    if twin.get("topology_key") != key:
        graph = topology.build_graph(mesh["nodes"], mesh["edges"])
        twin["topology_key"] = key
        twin["topology"] = {
            "graph": graph,
            "issues": topology.validate(graph),
            "positions": positions(mesh, graph),
            "neighbours": {n: topology.neighbours(graph, n) for n in graph.nodes},
            "dict": topology.to_dict(graph),
        }
        twin["topology"]["model"] = build_model(mesh, twin["topology"])
    return twin["topology"]


def build_model(mesh, topo):
    if gnn is None or not MODEL_PATH.exists():
        return None
    try:
        return gnn.FastModel(MODEL_PATH, list(topo["graph"].nodes), topo["positions"],
                             mesh["edges"], topo["dict"]["strings"])
    except Exception as e:  # a bad model file must not stop the twin
        print(f"AI model not loaded: {e}", flush=True)
        return None


def run_model(model, mesh):
    if model is None:
        return {}, None
    nodes = list(model.nodes)
    tick = time.perf_counter()
    cls, probs = model.step([mesh["nodes"].get(n, 0.0) for n in nodes])
    ms = (time.perf_counter() - tick) * 1000
    return {n: (model.classes[c], float(probs[i, c])) for i, (n, c) in enumerate(zip(nodes, cls))}, ms


def update(twin, mesh, now):
    topo = analyse_topology(twin, mesh)
    graph = topo["graph"]
    twin["issues"] = topo["issues"]
    pos = topo["positions"]

    current = mesh["current"] if mesh["current"] is not None else 1.0
    result = optimize(mesh["nodes"], None, mesh["current"])
    health = result["health"]
    v_best = result["reference"]

    neighbours = topo["neighbours"]
    diag = diagnosis.diagnose(twin["trackers"], health, neighbours, now)
    ai, ai_ms = run_model(topo["model"], mesh)
    cloud = radar.track(twin["trackers"], pos, now, PITCH_M, topo["dict"]["strings"])
    ages = {n: (now - d["since"]) if d["since"] is not None else 0.0 for n, d in diag.items()}

    panels = []
    for node in graph.nodes:
        v = mesh["nodes"].get(node)
        d = diag.get(node, {"diagnosis": "unknown", "action": "none"})
        panels.append({
            "id": node,
            "x": pos[node][0],
            "y": pos[node][1],
            "voltage": v,
            "health": health.get(node),
            "diagnosis": d["diagnosis"],
            "action": d["action"],
            "age": round(ages.get(node, 0.0), 1),
            "eta": cloud["eta"].get(node),
            "loss_w": current * max(v_best - v, 0.0) if v is not None else None,
            "ai_diagnosis": ai[node][0] if node in ai else None,
            "ai_action": AI_ACTIONS.get(ai[node][0]) if node in ai else None,
            "ai_confidence": round(ai[node][1], 3) if node in ai else None,
        })

    technicians = diagnosis.technician_needed(diag)
    # Every module's voltage is measured under load and bypassed modules don't
    # stop their string, so plant power = current x sum of module voltages.
    p_actual = current * sum(mesh["nodes"].values())
    return {
        "time": time.time() * 1000,
        "temp": mesh["temp"],
        "current": mesh["current"],
        "cleaner": mesh.get("cleaner"),
        "sim": mesh.get("sim"),
        "sim_time": mesh.get("time"),
        "panels": panels,
        "graph": topo["dict"],
        "metrics": {
            "reference": v_best,
            "mse": result["mse"],
            "power_actual": p_actual,
            "power_ideal": result["power_ideal"],
            "power_lost": result["power_ideal"] - p_actual,
            "efficiency": p_actual / result["power_ideal"] if result["power_ideal"] > 0 else 0.0,
        },
        "technician": {"needed": bool(technicians), "modules": technicians},
        "ai": {
            "enabled": topo["model"] is not None,
            "technician": sorted(n for n, (c, _) in ai.items() if AI_ACTIONS[c] == "technician"),
            "inference_ms": round(ai_ms, 1) if ai_ms is not None else None,
            "model": {k: topo["model"].meta.get(k) for k in ("epoch", "val_macro_f1")} if topo["model"] else None,
        },
        "cloud": cloud,
        "alerts": twin["issues"] + cloud_alert(cloud) + module_alerts(diag, ages),
    }
