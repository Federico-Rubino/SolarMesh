
import json
import sys
from pathlib import Path

import numpy as np
import torch

from model import N_FEATURES, PEAK_DECAY, DIST_SCALE
from slow import CURRENT_PEAK_DECAY, EMA_DAYS, N_SLOW_FEATURES


def main():
    src = Path(sys.argv[1])
    ckpt = torch.load(src, map_location="cpu")
    kind = ckpt.get("kind", "fast")
    arrays = {k: v.detach().numpy().astype(np.float32) for k, v in ckpt["state_dict"].items()}
    meta = {
        "kind": kind,
        "hidden": ckpt["hidden"],
        "classes": ckpt["classes"],
        "n_features": N_SLOW_FEATURES if kind == "slow" else N_FEATURES,
        "peak_decay": PEAK_DECAY,
        "dist_scale": DIST_SCALE,
        "ema_days": EMA_DAYS,
        "current_peak_decay": CURRENT_PEAK_DECAY,
        "epoch": ckpt.get("epoch"),
        "val_macro_f1": ckpt.get("val_macro_f1"),
    }
    out = src.with_suffix(".npz")
    np.savez(out, __meta__=np.array(json.dumps(meta)), **arrays)
    print(f"{out}: {kind} model, {sum(a.size for a in arrays.values())} weights, classes {meta['classes']}")


if __name__ == "__main__":
    main()
