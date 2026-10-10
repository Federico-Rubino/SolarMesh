from collections import deque

ANOMALY = 0.85
RECOVER = 0.90
BASELINE = 0.95
DEAD = 0.05
SUDDEN_S = 3.0
PERSIST_S = 20.0
HISTORY_S = 600.0

ACTIONS = {
    "ok": "none",
    "shade": "none",
    "obstruction": "inspect",
    "soiling": "clean",
    "fault": "technician",
    "dead": "technician",
}


def new_tracker():
    return {"history": deque(), "since": None, "drop_s": None, "recovered": None}


def update_tracker(tracker, now, health):
    history = tracker["history"]
    history.append((now, health))
    while history and now - history[0][0] > HISTORY_S:
        history.popleft()

    if tracker["since"] is None and health < ANOMALY:
        tracker["since"] = now
        tracker["drop_s"] = drop_duration(history, now)
    elif tracker["since"] is not None and health > RECOVER:
        tracker["since"] = None
        tracker["drop_s"] = None
        tracker["recovered"] = now


def drop_duration(history, onset):
    for t, h in reversed(history):
        if h >= BASELINE:
            return onset - t
    return None


def is_isolated(node, neighbours, trackers):
    if not neighbours:
        return True
    anomalous = sum(1 for n in neighbours if trackers.get(n, {}).get("since") is not None)
    return anomalous / len(neighbours) < 0.5


def classify(tracker, health, now, isolated):
    if tracker["since"] is None:
        return "ok"
    duration = now - tracker["since"]
    if duration < PERSIST_S:
        return "shade"
    if health <= DEAD:
        return "dead"
    sudden = tracker["drop_s"] is not None and tracker["drop_s"] <= SUDDEN_S
    if not sudden:
        return "soiling"
    return "fault" if isolated else "obstruction"


def diagnose(trackers, health, neighbours, now):
    for node, h in health.items():
        if h is None:
            continue
        update_tracker(trackers.setdefault(node, new_tracker()), now, h)

    result = {}
    for node, h in health.items():
        tracker = trackers.get(node)
        if h is None or tracker is None:
            result[node] = {"diagnosis": "unknown", "action": "none", "since": None}
            continue
        isolated = is_isolated(node, neighbours.get(node, []), trackers)
        label = classify(tracker, h, now, isolated)
        result[node] = {"diagnosis": label, "action": ACTIONS[label], "since": tracker["since"]}
    return result


def technician_needed(result):
    return sorted(n for n, r in result.items() if r["action"] == "technician")
