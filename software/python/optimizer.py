DEAD_THRESHOLD = 0.05


def measured_voltages(voltages, active):
    if active is None:
        return dict(voltages)
    return {n: v for n, v in voltages.items() if n in active}


def best_voltage(measured):
    return max(measured.values(), default=0.0)


def compute_health(voltages, measured, v_best):
    health = {}
    for node in voltages:
        if node not in measured:
            health[node] = None
        elif v_best > 0:
            health[node] = max(measured[node] / v_best, 0.0)
        else:
            health[node] = 0.0
    return health


def compute_mse(measured, v_best):
    if not measured:
        return 0.0
    return sum((v_best - v) ** 2 for v in measured.values()) / len(measured)


def rank_nodes(health, dead_threshold):
    alive = [n for n, h in health.items() if h is not None and h > dead_threshold]
    return sorted(alive, key=lambda n: (-health[n], n))


def string_power(nodes, measured, health, current):
    if not nodes:
        return 0.0
    i_string = current * min(health[n] for n in nodes)
    v_string = sum(measured[n] for n in nodes)
    return i_string * v_string


def ideal_power(v_best, n_nodes, current):
    return current * v_best * n_nodes


def sweep_configurations(ranked, measured, health, current):
    sweep = [{"k": 0, "node": None, "power": 0.0}]
    v_sum = 0.0
    for k, node in enumerate(ranked, start=1):
        v_sum += measured[node]
        sweep.append({"k": k, "node": node, "power": current * health[node] * v_sum})
    return sweep


def best_configuration(ranked, sweep):
    best = max(sweep, key=lambda s: (s["power"], s["k"]))
    return set(ranked[: best["k"]]), best["power"]


def gain(p_new, p_old):
    return (p_new - p_old) / p_old if p_old > 0 else None


def optimize(voltages, active=None, current=None, dead_threshold=DEAD_THRESHOLD):
    i_ref = current if current is not None else 1.0
    measured = measured_voltages(voltages, active)
    v_best = best_voltage(measured)
    health = compute_health(voltages, measured, v_best)
    ranked = rank_nodes(health, dead_threshold)

    sweep = sweep_configurations(ranked, measured, health, i_ref)
    on, p_with = best_configuration(ranked, sweep)
    known = [n for n, h in health.items() if h is not None]
    p_without = string_power(known, measured, health, i_ref)
    p_diode = sweep[-1]["power"]
    p_ideal = ideal_power(v_best, len(voltages), i_ref)

    return {
        "states": {n: int(n in on) for n in voltages},
        "health": health,
        "reference": v_best,
        "current": current,
        "mse": compute_mse(measured, v_best),
        "power_with": p_with,
        "power_without": p_without,
        "power_diode": p_diode,
        "power_ideal": p_ideal,
        "gain": gain(p_with, p_without),
        "gain_vs_diode": gain(p_with, p_diode),
        "efficiency": p_with / p_ideal if p_ideal > 0 else 0.0,
        "sweep": sweep,
    }
