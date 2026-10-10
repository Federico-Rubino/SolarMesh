import math

WINDOW_S = 30.0
HORIZON_S = 60.0
MIN_SLOWNESS = 1e-3
MIN_ONSETS = 5
MAX_RESIDUAL_S = 1.0
REARM_S = 120.0


def recent_onsets(trackers, positions, now, groups=None):
    first = {}
    for n, t in trackers.items():
        if (
            n in positions
            and t["since"] is not None
            and t["drop_s"] is not None
            and now - t["since"] <= WINDOW_S
            and (t.get("recovered") is None or t["since"] - t["recovered"] > REARM_S)
        ):
            key = groups.get(n, n) if groups else n
            if key not in first or t["since"] < first[key][2]:
                first[key] = (*positions[n], t["since"])
    return list(first.values())


def string_groups(strings, min_size=3):
    return {n: i for i, s in enumerate(strings) if len(s) >= min_size for n in s}


def is_planar(positions):
    pts = [(x, y, 0.0) for x, y in positions.values()]
    return fit_moments(pts)["det_ratio"] > 1e-3


def fit_moments(points):
    n = len(points)
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    syy = sum((p[1] - my) ** 2 for p in points)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    det = sxx * syy - sxy ** 2
    return {"det_ratio": det / (sxx + syy) ** 2 if sxx + syy else 0.0}


def fit_front(points, allow_1d=True):
    n = len(points)
    if n < 2:
        return None
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    mt = sum(p[2] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    syy = sum((p[1] - my) ** 2 for p in points)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    sxt = sum((p[0] - mx) * (p[2] - mt) for p in points)
    syt = sum((p[1] - my) * (p[2] - mt) for p in points)
    if sxx + syy == 0:
        return None

    det = sxx * syy - sxy ** 2
    if det > 1e-6 * (sxx + syy) ** 2:
        sx = (sxt * syy - syt * sxy) / det
        sy = (syt * sxx - sxt * sxy) / det
    elif not allow_1d:
        return None
    else:
        angle = 0.5 * math.atan2(2 * sxy, sxx - syy)
        ux, uy = math.cos(angle), math.sin(angle)
        ss = sum(((p[0] - mx) * ux + (p[1] - my) * uy) ** 2 for p in points)
        st = sum(((p[0] - mx) * ux + (p[1] - my) * uy) * (p[2] - mt) for p in points)
        k = st / ss
        sx, sy = k * ux, k * uy

    slowness = math.hypot(sx, sy)
    if slowness < MIN_SLOWNESS:
        return None
    residual = math.sqrt(sum((p[2] - mt - sx * (p[0] - mx) - sy * (p[1] - my)) ** 2 for p in points) / n)
    if residual > MAX_RESIDUAL_S:
        return None
    return {"origin": (mx, my, mt), "slowness": (sx, sy), "residual": residual}


def arrival_time(front, x, y):
    mx, my, mt = front["origin"]
    sx, sy = front["slowness"]
    return mt + sx * (x - mx) + sy * (y - my)


def track(trackers, positions, now, pitch_m=1.0, strings=None):
    onsets = recent_onsets(trackers, positions, now, string_groups(strings or []))
    if len(onsets) < min(MIN_ONSETS, len(positions)):
        return {"active": False, "eta": {}}
    front = fit_front(onsets, allow_1d=not is_planar(positions))
    if front is None:
        return {"active": False, "eta": {}}

    sx, sy = front["slowness"]
    s2 = sx * sx + sy * sy
    vx, vy = sx / s2, sy / s2

    eta = {}
    for node, (x, y) in positions.items():
        if trackers.get(node, {}).get("since") is not None:
            continue
        dt = arrival_time(front, x, y) - now
        if 0 < dt <= HORIZON_S:
            eta[node] = round(dt, 1)

    return {
        "active": True,
        "velocity": (vx, vy),
        "speed_mps": math.hypot(vx, vy) * pitch_m,
        "heading_deg": math.degrees(math.atan2(vy, vx)),
        "eta": eta,
        "next_eta": min(eta.values()) if eta else None,
    }
