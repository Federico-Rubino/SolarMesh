
import torch

from model import Graph, SolarMeshGNN  # noqa: F401  (re-exported for convenience)

SLOW_CLASSES = ["ok", "soiling", "degradation", "recurring_shade", "fault"]
SLOW_ACTIONS = ["none", "clean", "monitor", "inspect", "technician"]
SLOW_RAW_TO_CLASS = [0, 1, 2, 3, 4]
HOURS = 8
N_SLOW_FEATURES = HOURS + 6
EMA_DAYS = 7
CURRENT_PEAK_DECAY = 0.995


def per_graph_median(v, g):
    if g.num_graphs <= 1:
        return v.median().expand_as(v)
    med = torch.stack([v[g.graph_id == i].median() for i in range(g.num_graphs)])
    return med[g.graph_id]


def daily_feature_step(volts, current, g, state=None):
    """volts: [HOURS, N], current: [HOURS, num_graphs] -> ([N, N_SLOW_FEATURES], state)."""
    rel = torch.stack([volts[h] / per_graph_median(volts[h], g).clamp(min=1e-3) for h in range(volts.shape[0])])
    rel = rel.clamp(-0.5, 1.5)
    mean, minimum = rel.mean(0), rel.min(0).values
    bypass = (rel < 0.05).float().mean(0)

    noon = current.max(0).values[g.graph_id]
    if state is None:
        state = {"ema": mean, "prev": mean, "cur_peak": noon.clamp(min=1e-3)}
    alpha = 2.0 / (EMA_DAYS + 1)
    cur_peak = torch.maximum(state["cur_peak"] * CURRENT_PEAK_DECAY, noon).clamp(min=1e-3)
    feats = torch.cat([
        rel.T,
        torch.stack([mean, minimum, mean - state["ema"], mean - state["prev"], bypass, noon / cur_peak], dim=1),
    ], dim=1)
    new_state = {"ema": (1 - alpha) * state["ema"] + alpha * mean, "prev": mean, "cur_peak": cur_peak}
    return feats, new_state


def daily_sequence_features(volts, current, g, state=None):
    """volts: [D, HOURS, N], current: [D, HOURS, num_graphs] -> ([D, N, F], state)."""
    out = []
    for d in range(volts.shape[0]):
        f, state = daily_feature_step(volts[d], current[d], g, state)
        out.append(f)
    return torch.stack(out), state
