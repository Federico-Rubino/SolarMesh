"""Load episodes written by simulation/dataset.py and sample training windows.

Episodes are memory-mapped: only the frames of a requested window are read,
so thousands of episodes fit on disk without being held in RAM.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch

from model import RAW_TO_CLASS, Graph, sequence_features

RAW_MAP = np.array(RAW_TO_CLASS, dtype=np.int64)


class Episode:
    def __init__(self, meta_path):
        meta_path = Path(meta_path)
        self.meta = json.loads(meta_path.read_text())
        stem = str(meta_path.with_suffix(""))
        self.nodes, self.frames = len(self.meta["nodes"]), self.meta["frames"]
        shape = (self.frames, self.nodes)
        self.volts = np.memmap(f"{stem}.volts.f32", dtype="<f4", mode="r", shape=shape)
        self.raw = np.memmap(f"{stem}.labels.u8", dtype=np.uint8, mode="r", shape=shape)
        self._graph = None

    @property
    def graph(self):
        if self._graph is None:
            self._graph = Graph.build(self.meta["pos"], self.meta["geo_edges"], self.meta["strings"])
        return self._graph

    def window(self, start, length):
        """Features computed from frame 0 (as in streaming), returned for [start, start+length)."""
        end = start + length
        volts = torch.from_numpy(np.array(self.volts[:end]))
        feats, _ = sequence_features(volts, self.graph)
        labels = torch.from_numpy(RAW_MAP[self.raw[start:end]])
        return feats[start:end], labels

    def class_counts(self, n_classes):
        return np.bincount(RAW_MAP[np.asarray(self.raw)].ravel(), minlength=n_classes)


def load_episodes(folder):
    return [Episode(p) for p in sorted(Path(folder).glob("ep_*.json"))]


def split(episodes, val_fraction=0.15, seed=0):
    eps = episodes[:]
    random.Random(seed).shuffle(eps)
    k = max(1, int(len(eps) * val_fraction))
    return eps[k:], eps[:k]


def batch(episodes, length, rng):
    """Random window from each episode, merged into one disjoint graph."""
    feats, labels, graphs = [], [], []
    for ep in episodes:
        start = rng.randint(0, ep.frames - length)
        f, y = ep.window(start, length)
        feats.append(f)
        labels.append(y)
        graphs.append(ep.graph)
    return torch.cat(feats, dim=1), torch.cat(labels, dim=1), Graph.union(graphs)


def class_counts(episodes, n_classes):
    counts = np.zeros(n_classes, dtype=np.int64)
    for ep in episodes:
        counts += ep.class_counts(n_classes)
    return torch.from_numpy(counts).float()
