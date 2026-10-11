"""Check that the board's numpy inference (software/python/gnn.py) matches PyTorch."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "software" / "python"))

import gnn  # noqa: E402
from data import load_episodes  # noqa: E402
from data_slow import load_episodes as load_daily  # noqa: E402
from model import Graph, SolarMeshGNN, sequence_features  # noqa: E402
from slow import daily_sequence_features  # noqa: E402


def topology(meta):
    names = meta["nodes"]
    positions = {n: tuple(p) for n, p in zip(names, meta["pos"])}
    edges = [(names[a], names[b], 1) for a, b in meta["geo_edges"]]
    strings = [[names[i] for i in s] for s in meta["strings"]]
    return names, positions, edges, strings


def main():
    ckpt_path, folder = Path(sys.argv[1]), sys.argv[2]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    kind = ckpt.get("kind", "fast")
    episodes = load_daily(folder) if kind == "slow" else load_episodes(folder)
    ep = episodes[0]
    names, positions, edges, strings = topology(ep.meta)

    torch_model = SolarMeshGNN(hidden=ckpt["hidden"], n_features=ckpt["state_dict"]["encode.0.weight"].shape[1],
                               n_classes=len(ckpt["classes"]))
    torch_model.load_state_dict(ckpt["state_dict"])
    torch_model.eval()
    g = Graph.build(ep.meta["pos"], ep.meta["geo_edges"], ep.meta["strings"])

    np_cls = gnn.SlowModel if kind == "slow" else gnn.FastModel
    np_model = np_cls(ckpt_path.with_suffix(".npz"), names, positions, edges, strings)

    if kind == "slow":
        volts = np.fromfile(ep.volts_path, "<f4").reshape(ep.frames, ep.hours, ep.nodes)
        current = np.fromfile(ep.current_path, "<f4").reshape(ep.frames, ep.hours)
        feats, _ = daily_sequence_features(torch.from_numpy(volts), torch.from_numpy(current[:, :, None]), g)
        steps = [(volts[d], current[d]) for d in range(ep.frames)]
    else:
        volts = ep.read_volts(ep.frames)
        feats, _ = sequence_features(torch.from_numpy(volts.copy()), g)
        steps = [(volts[t],) for t in range(ep.frames)]

    with torch.no_grad():
        torch_logits, _ = torch_model(feats, g)
    worst, agree, total = 0.0, 0, 0
    for t, args in enumerate(steps):
        np_model.step(*args)
        logits = np.maximum(np_model.h @ np_model.h1_w + np_model.h1_b, 0.0) @ np_model.h2_w + np_model.h2_b
        ref = torch_logits[t].numpy()
        worst = max(worst, float(np.abs(logits - ref).max()))
        agree += int((logits.argmax(1) == ref.argmax(1)).sum())
        total += len(ref)
    print(f"{kind} model, {ep.nodes} modules, {len(steps)} steps: max |logit diff| = {worst:.2e}, "
          f"same class {agree / total:.4%}")
    print("OK" if worst < 1e-3 and agree / total > 0.999 else "MISMATCH - send me this output")


if __name__ == "__main__":
    main()
