DEAD_THRESHOLD = 0.05


def measured_voltages(voltages, active):
    if active is None:
        return dict(voltages)
    return {n: v for n, v in voltages.items() if n in active}


def best_voltage(measured):
    return max(measured.values(), default=0.0)


def ideal_voltage(v_best, n_nodes):
    return v_best * n_nodes


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


def best_configuration(ranked, measured):
    best_k, best_v = 0, 0.0
    sweep = [{"k": 0, "node": None, "voltage": 0.0}]
    for k, node in enumerate(ranked, start=1):
        v = k * measured[node]
        sweep.append({"k": k, "node": node, "voltage": v})
        if v >= best_v:
            best_k, best_v = k, v
    return set(ranked[:best_k]), best_v, sweep


def optimize(voltages, active=None, dead_threshold=DEAD_THRESHOLD):
    measured = measured_voltages(voltages, active)
    v_best = best_voltage(measured)
    v_ideal = ideal_voltage(v_best, len(voltages))
    health = compute_health(voltages, measured, v_best)
    on, v_expected, sweep = best_configuration(rank_nodes(health, dead_threshold), measured)

    return {
        "states": {n: int(n in on) for n in voltages},
        "health": health,
        "reference": v_best,
        "ideal_voltage": v_ideal,
        "expected_voltage": v_expected,
        "mse": compute_mse(measured, v_best),
        "efficiency": v_expected / v_ideal if v_ideal > 0 else 0.0,
        "sweep": sweep,
    }
