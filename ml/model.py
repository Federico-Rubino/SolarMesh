"""SolarMesh spatio-temporal graph model (architecture D: recurrent graph GRU)."""

import torch
import torch.nn as nn

CLASSES = ["ok", "shade", "soiling", "obstruction", "debris_or_diode", "dead"]
RAW_TO_CLASS = [0, 1, 2, 3, 4, 4, 5]          # dataset RAW_CLASSES -> CLASSES
ACTIONS = ["none", "none", "clean", "inspect", "inspect", "technician"]
N_FEATURES = 6
PEAK_DECAY = 0.999
DIST_SCALE = 5.0


# ---------- topology ----------

class Graph:
    """Fixed topology as tensors. Several graphs can be merged into one batch."""

    def __init__(self, num_nodes, geo_src, geo_dst, geo_attr, node_string, string_size, graph_id):
        self.num_nodes = num_nodes
        self.geo_src, self.geo_dst, self.geo_attr = geo_src, geo_dst, geo_attr
        indeg = torch.zeros(num_nodes).index_add_(0, geo_dst, torch.ones(len(geo_dst)))
        self.geo_norm = 1.0 / indeg.clamp(min=1)[geo_dst]
        self.node_string = node_string
        self.string_size = string_size
        self.graph_id = graph_id
        self.num_graphs = int(graph_id.max()) + 1 if num_nodes else 0

    @staticmethod
    def build(pos, geo_edges, strings):
        n = len(pos)
        pos = torch.tensor(pos, dtype=torch.float32)
        e = torch.tensor(geo_edges, dtype=torch.long).reshape(-1, 2)
        src = torch.cat([e[:, 0], e[:, 1]])
        dst = torch.cat([e[:, 1], e[:, 0]])
        d = (pos[src] - pos[dst]) / DIST_SCALE
        attr = torch.cat([d, d.norm(dim=1, keepdim=True)], dim=1)

        node_string = torch.arange(n)
        sizes = [1] * n
        for k, members in enumerate(strings):
            for m in members:
                node_string[m] = n + k
        size_t = torch.tensor(sizes + [len(s) for s in strings], dtype=torch.float32)
        return Graph(n, src, dst, attr, node_string, size_t, torch.zeros(n, dtype=torch.long))

    @staticmethod
    def union(graphs):
        offs, string_offs = 0, 0
        parts = {k: [] for k in ("src", "dst", "attr", "ns", "size", "gid")}
        for i, g in enumerate(graphs):
            parts["src"].append(g.geo_src + offs)
            parts["dst"].append(g.geo_dst + offs)
            parts["attr"].append(g.geo_attr)
            parts["ns"].append(g.node_string + string_offs)
            parts["size"].append(g.string_size)
            parts["gid"].append(torch.full((g.num_nodes,), i, dtype=torch.long))
            offs += g.num_nodes
            string_offs += len(g.string_size)
        cat = {k: torch.cat(v) for k, v in parts.items()}
        return Graph(offs, cat["src"], cat["dst"], cat["attr"], cat["ns"], cat["size"], cat["gid"])

    def to(self, device):
        for k in ("geo_src", "geo_dst", "geo_attr", "geo_norm", "node_string", "string_size", "graph_id"):
            setattr(self, k, getattr(self, k).to(device))
        return self


# ---------- features (identical in training and streaming inference) ----------

def graph_median(v, g):
    if g.num_graphs <= 1:
        return v.median().expand_as(v)
    med = torch.stack([v[g.graph_id == i].median() for i in range(g.num_graphs)])
    return med[g.graph_id]


def feature_step(v, g, state=None):
    """v: [N] voltages of one frame. Returns ([N, N_FEATURES], new state)."""
    med = graph_median(v, g).clamp(min=1e-3)
    if state is None:
        state = {"peak": v.clamp(min=1e-3), "plant_peak": med, "prev": v}
    peak = torch.maximum(state["peak"] * PEAK_DECAY, v).clamp(min=1e-3)
    plant_peak = torch.maximum(state["plant_peak"] * PEAK_DECAY, med)
    rel = v / med
    step = ((rel - 2 / 3).abs() < 0.08) | ((rel - 1 / 3).abs() < 0.08)
    feats = torch.stack([
        rel.clamp(-0.5, 1.5),
        (v / peak).clamp(-0.5, 1.5),
        ((v - state["prev"]) / med).clamp(-1.5, 1.5),
        (rel < 0.05).float(),
        med / plant_peak,
        step.float(),
    ], dim=1)
    return feats, {"peak": peak, "plant_peak": plant_peak, "prev": v}


def sequence_features(volts, g, state=None):
    """volts: [T, N] -> ([T, N, N_FEATURES], final state)."""
    out = []
    for t in range(volts.shape[0]):
        f, state = feature_step(volts[t], g, state)
        out.append(f)
    return torch.stack(out), state


# ---------- layers ----------

class RelConv(nn.Module):
    """Relational graph convolution: self + geometric neighbours + same-string modules."""

    def __init__(self, din, dout):
        super().__init__()
        self.self_lin = nn.Linear(din, dout)
        self.geo_lin = nn.Linear(din, dout, bias=False)
        self.geo_attr = nn.Linear(3, dout)
        self.ele_lin = nn.Linear(din, dout, bias=False)

    def forward(self, h, g):
        out = self.self_lin(h)
        msg = self.geo_lin(h)[g.geo_src] + self.geo_attr(g.geo_attr)
        out = out.index_add(0, g.geo_dst, msg * g.geo_norm[:, None])

        he = self.ele_lin(h)
        sums = torch.zeros(len(g.string_size), he.shape[1], device=h.device).index_add(0, g.node_string, he)
        size = g.string_size[g.node_string][:, None]
        others = (sums[g.node_string] - he) / (size - 1).clamp(min=1)
        return out + others


class GraphGRUCell(nn.Module):
    def __init__(self, din, hidden):
        super().__init__()
        self.gates = RelConv(din + hidden, 2 * hidden)
        self.cand = RelConv(din + hidden, hidden)

    def forward(self, x, h, g):
        z, r = torch.sigmoid(self.gates(torch.cat([x, h], dim=1), g)).chunk(2, dim=1)
        c = torch.tanh(self.cand(torch.cat([x, r * h], dim=1), g))
        return (1 - z) * h + z * c


class SolarMeshGNN(nn.Module):
    def __init__(self, hidden=32, n_features=N_FEATURES, n_classes=len(CLASSES)):
        super().__init__()
        self.hidden = hidden
        self.encode = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU())
        self.cell = GraphGRUCell(hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def init_state(self, g, device=None):
        return torch.zeros(g.num_nodes, self.hidden, device=device)

    def step(self, feats, g, h):
        h = self.cell(self.encode(feats), h, g)
        return self.head(h), h

    def forward(self, feats, g, h=None):
        """feats: [T, N, F] -> logits [T, N, C], final hidden state."""
        if h is None:
            h = self.init_state(g, feats.device)
        logits = []
        for t in range(feats.shape[0]):
            out, h = self.step(feats[t], g, h)
            logits.append(out)
        return torch.stack(logits), h
