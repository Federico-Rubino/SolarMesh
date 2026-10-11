"""numpy inference for the SolarMesh graph models."""

import collections
import json
import pickle
import zipfile

import numpy as np

DEFAULTS = {"peak_decay": 0.999, "dist_scale": 5.0, "ema_days": 7, "current_peak_decay": 0.995}
STORAGE_DTYPES = {
    "FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
    "LongStorage": np.int64, "IntStorage": np.int32, "BoolStorage": np.bool_,
}


def _rebuild_tensor(storage, offset, size, stride, *_):
    if not size:
        return storage[offset].copy()
    strides = [s * storage.itemsize for s in stride]
    return np.lib.stride_tricks.as_strided(storage[offset:], shape=size, strides=strides).copy()


class _CheckpointUnpickler(pickle.Unpickler):
    """Reads a torch.save() zip checkpoint into numpy; only tensor/dict types are allowed."""

    def __init__(self, file, archive, prefix):
        super().__init__(file)
        self.archive, self.prefix = archive, prefix

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor
        if module == "torch" and name in STORAGE_DTYPES:
            return name
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        raise pickle.UnpicklingError(f"unexpected object in checkpoint: {module}.{name}")

    def persistent_load(self, pid):
        _, storage_type, key, _, _ = pid
        name = storage_type if isinstance(storage_type, str) else getattr(storage_type, "__name__", "")
        data = self.archive.read(f"{self.prefix}/data/{key}")
        return np.frombuffer(data, dtype=STORAGE_DTYPES[name])


def load_checkpoint(path):
    """torch.save() checkpoint -> (weights dict, meta dict), numpy only."""
    with zipfile.ZipFile(path) as archive:
        pkl = next(n for n in archive.namelist() if n.endswith("data.pkl"))
        prefix = pkl.rsplit("/", 1)[0]
        with archive.open(pkl) as f:
            ckpt = _CheckpointUnpickler(f, archive, prefix).load()
    weights = {k: np.asarray(v, dtype=np.float32) for k, v in ckpt["state_dict"].items()}
    n_features = weights["encode.0.weight"].shape[1]
    meta = {**DEFAULTS, "kind": ckpt.get("kind", "slow" if n_features > 6 else "fast"),
            "hidden": ckpt["hidden"], "classes": ckpt["classes"], "n_features": n_features,
            "epoch": ckpt.get("epoch"), "val_macro_f1": ckpt.get("val_macro_f1")}
    return weights, meta


def load_model_file(path):
    if str(path).endswith(".pt"):
        return load_checkpoint(path)
    data = np.load(path)
    meta = {**DEFAULTS, **json.loads(str(data["__meta__"]))}
    return {k: data[k] for k in data.files if k != "__meta__"}, meta


def lower_median(v):
    """torch.median semantics: the lower of the two middle values."""
    return np.partition(v, (len(v) - 1) // 2)[(len(v) - 1) // 2]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class GraphTensors:
    """Fixed topology in index form, built once per plant layout."""

    def __init__(self, nodes, positions, edges, strings, dist_scale):
        index = {n: i for i, n in enumerate(nodes)}
        self.n = len(nodes)
        pos = np.array([positions[n] for n in nodes], dtype=np.float32)
        pairs = np.array([(index[a], index[b]) for a, b, *_ in edges if a in index and b in index],
                         dtype=np.int64).reshape(-1, 2)
        self.src = np.concatenate([pairs[:, 0], pairs[:, 1]])
        self.dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
        d = (pos[self.src] - pos[self.dst]) / dist_scale
        self.attr = np.concatenate([d, np.linalg.norm(d, axis=1, keepdims=True)], axis=1).astype(np.float32)
        indeg = np.bincount(self.dst, minlength=self.n).astype(np.float32)
        self.norm = (1.0 / np.maximum(indeg, 1.0))[self.dst][:, None]

        self.node_string = np.arange(self.n)
        sizes = [1.0] * self.n
        for k, members in enumerate(strings):
            for m in members:
                if m in index:
                    self.node_string[index[m]] = self.n + k
        self.string_size = np.array(sizes + [len(s) for s in strings], dtype=np.float32)
        self.others_div = np.maximum(self.string_size[self.node_string] - 1, 1.0)[:, None]


class RelConv:
    def __init__(self, w, prefix, g):
        self.w_self, self.b_self = w[f"{prefix}.self_lin.weight"].T, w[f"{prefix}.self_lin.bias"]
        self.w_geo = w[f"{prefix}.geo_lin.weight"].T
        self.w_ele = w[f"{prefix}.ele_lin.weight"].T
        attr_msg = (g.attr @ w[f"{prefix}.geo_attr.weight"].T + w[f"{prefix}.geo_attr.bias"]) * g.norm
        self.attr_term = np.zeros((g.n, attr_msg.shape[1]), dtype=np.float32)
        np.add.at(self.attr_term, g.dst, attr_msg)
        self.g = g

    def __call__(self, h):
        g = self.g
        out = h @ self.w_self + self.b_self + self.attr_term
        np.add.at(out, g.dst, (h @ self.w_geo)[g.src] * g.norm)
        he = h @ self.w_ele
        sums = np.zeros((len(g.string_size), he.shape[1]), dtype=np.float32)
        np.add.at(sums, g.node_string, he)
        return out + (sums[g.node_string] - he) / g.others_div


class GraphGRU:
    """Streaming SolarMeshGNN: call step() once per frame."""

    def __init__(self, path, nodes, positions, edges, strings):
        w, self.meta = load_model_file(path)
        self.nodes = list(nodes)
        self.classes = self.meta["classes"]
        self.hidden = self.meta["hidden"]
        self.g = GraphTensors(nodes, positions, edges, strings, self.meta["dist_scale"])
        self.enc_w, self.enc_b = w["encode.0.weight"].T, w["encode.0.bias"]
        self.gates = RelConv(w, "cell.gates", self.g)
        self.cand = RelConv(w, "cell.cand", self.g)
        self.h1_w, self.h1_b = w["head.0.weight"].T, w["head.0.bias"]
        self.h2_w, self.h2_b = w["head.2.weight"].T, w["head.2.bias"]
        self.h = np.zeros((self.g.n, self.hidden), dtype=np.float32)
        self.feat_state = None

    def step_features(self, feats):
        x = np.maximum(feats @ self.enc_w + self.enc_b, 0.0)
        zr = sigmoid(self.gates(np.concatenate([x, self.h], axis=1)))
        z, r = zr[:, :self.hidden], zr[:, self.hidden:]
        c = np.tanh(self.cand(np.concatenate([x, r * self.h], axis=1)))
        self.h = (1 - z) * self.h + z * c
        logits = np.maximum(self.h @ self.h1_w + self.h1_b, 0.0) @ self.h2_w + self.h2_b
        return logits

    def predict(self, logits):
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        return probs.argmax(axis=1), probs


class FastModel(GraphGRU):
    """One step per 0.5 s reading; input = module voltages in node order."""

    def features(self, v):
        decay = self.meta["peak_decay"]
        med = max(float(lower_median(v)), 1e-3)
        st = self.feat_state or {"peak": np.maximum(v, 1e-3), "plant_peak": med, "prev": v}
        peak = np.maximum(np.maximum(st["peak"] * decay, v), 1e-3)
        plant_peak = max(st["plant_peak"] * decay, med)
        rel = v / med
        step = (np.abs(rel - 2 / 3) < 0.08) | (np.abs(rel - 1 / 3) < 0.08)
        feats = np.stack([
            np.clip(rel, -0.5, 1.5),
            np.clip(v / peak, -0.5, 1.5),
            np.clip((v - st["prev"]) / med, -1.5, 1.5),
            (rel < 0.05).astype(np.float32),
            np.full_like(v, med / plant_peak),
            step.astype(np.float32),
        ], axis=1).astype(np.float32)
        self.feat_state = {"peak": peak, "plant_peak": plant_peak, "prev": v}
        return feats

    def step(self, volts):
        v = np.asarray(volts, dtype=np.float32)
        return self.predict(self.step_features(self.features(v)))


class SlowModel(GraphGRU):
    """One step per day; input = [hours x nodes] voltages and [hours] plant current."""

    def features(self, volts, current):
        rel = np.stack([volts[h] / max(float(lower_median(volts[h])), 1e-3) for h in range(volts.shape[0])])
        rel = np.clip(rel, -0.5, 1.5)
        mean, minimum = rel.mean(0), rel.min(0)
        bypass = (rel < 0.05).astype(np.float32).mean(0)
        noon = float(np.max(current))
        st = self.feat_state or {"ema": mean, "prev": mean, "cur_peak": max(noon, 1e-3)}
        alpha = 2.0 / (self.meta["ema_days"] + 1)
        cur_peak = max(st["cur_peak"] * self.meta["current_peak_decay"], noon, 1e-3)
        feats = np.concatenate([
            rel.T,
            np.stack([mean, minimum, mean - st["ema"], mean - st["prev"], bypass,
                      np.full_like(mean, noon / cur_peak)], axis=1),
        ], axis=1).astype(np.float32)
        self.feat_state = {"ema": (1 - alpha) * st["ema"] + alpha * mean, "prev": mean, "cur_peak": cur_peak}
        return feats

    def step(self, volts, current):
        return self.predict(self.step_features(self.features(np.asarray(volts, np.float32),
                                                             np.asarray(current, np.float32))))
