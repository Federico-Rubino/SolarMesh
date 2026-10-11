
import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import data
import data_slow
from data import batch, class_counts, split
from model import ACTIONS as FAST_ACTIONS
from model import CLASSES as FAST_CLASSES
from model import N_FEATURES, Graph, SolarMeshGNN
from slow import N_SLOW_FEATURES, SLOW_ACTIONS, SLOW_CLASSES

KINDS = {
    "fast": {"classes": FAST_CLASSES, "actions": FAST_ACTIONS, "features": N_FEATURES,
             "load": data.load_episodes, "window": 160, "burn_in": 40, "out": "checkpoints/gnn.pt"},
    "slow": {"classes": SLOW_CLASSES, "actions": SLOW_ACTIONS, "features": N_SLOW_FEATURES,
             "load": data_slow.load_episodes, "window": 60, "burn_in": 14, "out": "checkpoints/gnn_slow.pt"},
}
CLASSES, ACTIONS = FAST_CLASSES, FAST_ACTIONS


def device_auto():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def evaluate(model, episodes, burn_in, device):
    model.eval()
    conf = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.long)
    for ep in episodes:
        feats, labels = ep.window(0, ep.frames)
        g = Graph.union([ep.graph]).to(device)
        logits, _ = model(feats.to(device), g)
        pred = logits.argmax(-1).cpu()[burn_in:].flatten()
        true = labels[burn_in:].flatten()
        conf += torch.bincount(true * len(CLASSES) + pred, minlength=len(CLASSES) ** 2).reshape(len(CLASSES), -1)
    model.train()
    return conf


def report(conf):
    tp = conf.diag().float()
    recall = tp / conf.sum(1).clamp(min=1)
    precision = tp / conf.sum(0).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-9)
    present = conf.sum(1) > 0
    lines = [f"{'class':<16}{'support':>10}{'recall':>8}{'prec':>8}{'f1':>8}   action"]
    for i, c in enumerate(CLASSES):
        lines.append(f"{c:<16}{int(conf[i].sum()):>10}{recall[i]:>8.2f}{precision[i]:>8.2f}{f1[i]:>8.2f}   {ACTIONS[i]}")
    macro = f1[present].mean().item()
    lines.append(f"macro-F1 {macro:.3f}")
    lines.append("confusion (rows = truth, cols = predicted): " + " ".join(f"{c[:5]:>6}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        lines.append(f"  {c[:14]:<14}" + " ".join(f"{int(v):>6}" for v in conf[i]))
    return macro, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=list(KINDS), default="fast")
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=100, help="optimizer steps per epoch")
    ap.add_argument("--batch", type=int, default=4, help="episodes per step")
    ap.add_argument("--window", type=int, default=None, help="frames per window (fast: 160 x 0.5 s, slow: 60 days)")
    ap.add_argument("--burn-in", type=int, default=None, help="frames without loss at window start (fast 40, slow 14)")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    global CLASSES, ACTIONS
    kind = KINDS[args.kind]
    CLASSES, ACTIONS = kind["classes"], kind["actions"]
    args.window = args.window or kind["window"]
    args.burn_in = args.burn_in if args.burn_in is not None else kind["burn_in"]
    args.out = args.out or kind["out"]

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = device_auto()
    episodes = kind["load"](args.data)
    train_eps, val_eps = split(episodes)
    print(f"{len(train_eps)} train / {len(val_eps)} val episodes, device {device}", flush=True)

    counts = class_counts(train_eps, len(CLASSES))
    weights = (counts.sum() / counts.clamp(min=1)).sqrt()
    weights = (weights / weights.mean()).to(device)
    print("class frames:", {c: int(n) for c, n in zip(CLASSES, counts)}, flush=True)

    model = SolarMeshGNN(hidden=args.hidden, n_features=kind["features"], n_classes=len(CLASSES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best = -1.0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        started, total = time.time(), 0.0
        for _ in range(args.steps):
            eps = rng.sample(train_eps, min(args.batch, len(train_eps)))
            feats, labels, g = batch(eps, args.window, rng)
            logits, _ = model(feats.to(device), g.to(device))
            loss = F.cross_entropy(
                logits[args.burn_in:].reshape(-1, len(CLASSES)),
                labels[args.burn_in:].reshape(-1).to(device),
                weight=weights,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sched.step()

        macro, text = report(evaluate(model, val_eps, args.burn_in, device))
        print(f"epoch {epoch:3d}  loss {total / args.steps:.4f}  val macro-F1 {macro:.3f}  ({time.time() - started:.0f} s)", flush=True)
        if macro > best:
            best = macro
            torch.save({"state_dict": model.state_dict(), "kind": args.kind, "hidden": args.hidden, "classes": CLASSES,
                        "epoch": epoch, "val_macro_f1": macro}, args.out)
            print(text, flush=True)
    print(f"best val macro-F1 {best:.3f} → {args.out}")


if __name__ == "__main__":
    main()
