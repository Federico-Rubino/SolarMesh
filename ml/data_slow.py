
import json
from pathlib import Path

import numpy as np
import torch

from model import Graph
from slow import SLOW_RAW_TO_CLASS, daily_sequence_features

RAW_MAP = np.array(SLOW_RAW_TO_CLASS, dtype=np.int64)


class DailyEpisode:
    def __init__(self, meta_path):
        meta_path = Path(meta_path)
        self.meta = json.loads(meta_path.read_text())
        stem = str(meta_path.with_suffix(""))
        self.volts_path = f"{stem}.volts.f32"
        self.current_path = f"{stem}.current.f32"
        self.labels_path = f"{stem}.labels.u8"
        self.nodes, self.frames, self.hours = len(self.meta["nodes"]), self.meta["days"], len(self.meta["hours"])
        self._graph = None

    @property
    def graph(self):
        if self._graph is None:
            self._graph = Graph.build(self.meta["pos"], self.meta["geo_edges"], self.meta["strings"])
        return self._graph

    def read_labels(self, start, end):
        data = np.fromfile(self.labels_path, dtype=np.uint8, count=(end - start) * self.nodes,
                           offset=start * self.nodes)
        return data.reshape(end - start, self.nodes)

    def window(self, start, length):
        """Features computed from day 0 (as in streaming), returned for days [start, start+length)."""
        end = start + length
        volts = np.fromfile(self.volts_path, dtype="<f4", count=end * self.hours * self.nodes)
        current = np.fromfile(self.current_path, dtype="<f4", count=end * self.hours)
        volts = torch.from_numpy(volts.reshape(end, self.hours, self.nodes))
        current = torch.from_numpy(current.reshape(end, self.hours, 1))
        feats, _ = daily_sequence_features(volts, current, self.graph)
        labels = torch.from_numpy(RAW_MAP[self.read_labels(start, end)])
        return feats[start:end], labels

    def class_counts(self, n_classes):
        return np.bincount(RAW_MAP[self.read_labels(0, self.frames)].ravel(), minlength=n_classes)


def load_episodes(folder):
    return [DailyEpisode(p) for p in sorted(Path(folder).glob("day_*.json"))]
