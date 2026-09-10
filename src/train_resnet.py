"""ResNet-18 classification baselines for the VLM benchmark tasks.

The VLMs answer image-level classification questions, so the fair CNN
baseline is an image-level classifier, not a YOLO detector. This script
trains ResNet-18 on any of the three tasks using EXACTLY the same
image-level ground truth as the VLM benchmark: labels are derived through
src.data_loading.load_task (majority box class for single-label YOLO
tasks, box-class set for defects, COCO votes for freshness). That means
the head-to-head tables compare against identical labels.

  donation_type  single-label, 3 classes, cross-entropy
  freshness      single-label, 3 classes, cross-entropy
  defects        multi-label, 6 classes, BCE-with-logits (pos_weight)

Protocol (matches the locked freshness ResNet run): ResNet-18 ImageNet
weights, 224x224, ImageNet normalization, Adam, seed 42, stratified
70/15/15 train/val/test split, best checkpoint selected on val macro-F1,
metrics reported on the held-out test split only.

Per task, everything lands in {output_dir}/resnet_{task}/:

  best.pt                     best-val-macro-F1 checkpoint
  class_map.json              index -> class name
  splits.json                 file_name lists for train/val/test
  test_manifest.txt           test-split basenames — point the VLM
                              benchmark's subset_manifest here to score
                              the VLMs on the same held-out images
  training_history.json       per-epoch train/val metrics (Fig 7/8 style)
  test_predictions.json       per-image prediction vs ground truth
  test_metrics.json           accuracy, macro-F1, per-class P/R/F1,
                              confusion matrix (single-label) or
                              per-defect sensitivity/specificity/F1 +
                              subset accuracy (defects)
  resnet_{task}_summary.csv   one row per class, same column names as
                              the VLM analyzer summaries

Usage:
  python -m src.train_resnet config.yaml --task donation_type
  python -m src.train_resnet config.yaml --task defects --epochs 40
  python -m src.train_resnet config.yaml --task donation_type --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from .data_loading import TASK_CLASSES, load_task

SEED_DEFAULT = 42
SPLIT_FRACS = (0.70, 0.15, 0.15)  # train / val / test


# ----------------------------------------------------------------
# Stratified split on Example lists (no torch needed)
# ----------------------------------------------------------------
def _iterative_stratify(examples, classes, seed):
    """Greedy iterative stratification (Sechidis et al., 2011) so every
    defect class is represented in train/val/test roughly proportionally
    despite tiny positive counts and label co-occurrence."""
    rng = np.random.RandomState(seed)
    n = len(examples)
    desired_total = [f * n for f in SPLIT_FRACS]
    desired = {c: [f * sum(1 for e in examples if c in e.labels)
                   for f in SPLIT_FRACS] for c in classes}
    assigned = [-1] * n
    placed = [0, 0, 0]
    remaining = set(range(n))

    def place(i, s):
        assigned[i] = s
        placed[s] += 1
        remaining.discard(i)
        for c in examples[i].labels:
            desired[c][s] -= 1

    while any(sum(1 for i in remaining if c in examples[i].labels)
              for c in classes):
        # label with the fewest remaining unassigned positives
        cands = [(sum(1 for i in remaining if c in examples[i].labels), c)
                 for c in classes]
        cands = [(k, c) for k, c in cands if k > 0]
        _, lab = min(cands)
        for i in sorted(i for i in remaining if lab in examples[i].labels):
            best = max(range(3), key=lambda s: (
                desired[lab][s],
                desired_total[s] - placed[s],
                rng.rand()))
            place(i, best)
    # defect-free images: fill remaining capacity
    for i in sorted(remaining):
        best = max(range(3), key=lambda s: (desired_total[s] - placed[s],
                                            rng.rand()))
        place(i, best)
    return ([examples[i] for i in range(n) if assigned[i] == 0],
            [examples[i] for i in range(n) if assigned[i] == 1],
            [examples[i] for i in range(n) if assigned[i] == 2])


def stratified_split(examples, task: str, seed: int):
    if task == "defects":
        return _iterative_stratify(examples, TASK_CLASSES[task], seed)

    from sklearn.model_selection import train_test_split
    keys = [e.label for e in examples]
    key_counts = Counter(keys)
    keys = [k if key_counts[k] >= 3 else "_rare" for k in keys]

    idx = np.arange(len(examples))
    try:
        tr, rest = train_test_split(idx, test_size=1 - SPLIT_FRACS[0],
                                    random_state=seed,
                                    stratify=[keys[i] for i in idx])
        rest_keys = [keys[i] for i in rest]
        rk = Counter(rest_keys)
        rest_keys = [k if rk[k] >= 2 else "_rare" for k in rest_keys]
        va, te = train_test_split(rest, test_size=SPLIT_FRACS[2] /
                                  (SPLIT_FRACS[1] + SPLIT_FRACS[2]),
                                  random_state=seed, stratify=rest_keys)
    except ValueError:
        # degenerate class counts: fall back to a plain shuffled split
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(examples))
        n_tr = int(SPLIT_FRACS[0] * len(idx))
        n_va = int(SPLIT_FRACS[1] * len(idx))
        tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    return ([examples[i] for i in sorted(tr)],
            [examples[i] for i in sorted(va)],
            [examples[i] for i in sorted(te)])


def _dist(examples, task):
    if task == "defects":
        pos = Counter(c for e in examples for c in e.labels)
        neg = sum(1 for e in examples if not e.labels)
        return {**dict(pos), "defect_free": neg}
    return dict(Counter(e.label for e in examples))


# ----------------------------------------------------------------
# Torch bits (imported lazily so --dry-run works without torch)
# ----------------------------------------------------------------
def build_dataset(examples, classes, task, train: bool, augment: bool):
    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision import transforms

    tfs = [transforms.Resize((224, 224))]
    if train and augment:
        tfs += [transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1)]
    tfs += [transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])]
    tf = transforms.Compose(tfs)
    c2i = {c: i for i, c in enumerate(classes)}
    multi = task == "defects"

    class _DS(Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, i):
            e = examples[i]
            img = tf(Image.open(e.path).convert("RGB"))
            if multi:
                y = torch.zeros(len(classes))
                for c in e.labels:
                    y[c2i[c]] = 1.0
            else:
                y = c2i[e.label]
            return img, y, e.file_name

    return _DS()


def single_label_metrics(y_true, y_pred, classes):
    from sklearn.metrics import (confusion_matrix, f1_score,
                                 precision_recall_fscore_support)
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0)),
        "per_class": {classes[i]: {"precision": float(p[i]),
                                   "recall": float(r[i]),
                                   "f1": float(f[i]),
                                   "support": int(s[i])}
                      for i in labels},
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels).tolist(),
        "confusion_matrix_order": classes,
    }


def multi_label_metrics(y_true, y_pred, classes):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    per = {}
    f1s = []
    for i, c in enumerate(classes):
        t, p = y_true[:, i], y_pred[:, i]
        tp = int(((t == 1) & (p == 1)).sum())
        tn = int(((t == 0) & (p == 0)).sum())
        fp = int(((t == 0) & (p == 1)).sum())
        fn = int(((t == 1) & (p == 0)).sum())
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
        per[c] = {"sensitivity": sens, "specificity": spec,
                  "precision": prec, "f1": f1,
                  "support": int((t == 1).sum())}
        f1s.append(f1)
    return {
        "subset_accuracy": float((y_true == y_pred).all(axis=1).mean()),
        "macro_f1": float(np.mean(f1s)),
        "per_defect": per,
    }


def train(args, cfg, examples, classes, splits):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import models

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    tr, va, te = splits
    task, multi = args.task, args.task == "defects"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    loaders = {}
    for name, exs, is_tr in (("train", tr, True), ("val", va, False),
                             ("test", te, False)):
        ds = build_dataset(exs, classes, task, is_tr, args.augment)
        loaders[name] = DataLoader(ds, batch_size=args.batch_size,
                                   shuffle=is_tr, num_workers=args.workers,
                                   pin_memory=device.type == "cuda")

    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)

    if multi:
        # pos_weight offsets heavy negative skew per defect
        pos = np.array([sum(1 for e in tr if c in e.labels)
                        for c in classes], dtype=float)
        neg = len(tr) - pos
        w = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1, 50),
                         dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=w)
        print("pos_weight:", dict(zip(classes, np.round(
            w.cpu().numpy(), 1).tolist())))
    else:
        weight = None
        if args.class_weights:
            cnt = Counter(e.label for e in tr)
            weight = torch.tensor(
                [len(tr) / (len(classes) * cnt[c]) for c in classes],
                dtype=torch.float32, device=device)
            print("class weights:", dict(zip(classes, np.round(
                weight.cpu().numpy(), 2).tolist())))
        criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def run_epoch(loader, train_mode):
        model.train(train_mode)
        losses, ys, ps = [], [], []
        ctx = torch.enable_grad() if train_mode else torch.no_grad()
        with ctx:
            for x, y, _ in loader:
                x = x.to(device)
                y = y.to(device)
                out = model(x)
                loss = criterion(out, y if not multi else y.float())
                if train_mode:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                losses.append(loss.item() * len(x))
                if multi:
                    ps.extend((torch.sigmoid(out) >= 0.5).long()
                              .cpu().numpy().tolist())
                    ys.extend(y.long().cpu().numpy().tolist())
                else:
                    ps.extend(out.argmax(1).cpu().numpy().tolist())
                    ys.extend(y.cpu().numpy().tolist())
        m = (multi_label_metrics(ys, ps, classes) if multi
             else single_label_metrics(ys, ps, classes))
        m["loss"] = float(np.sum(losses) / len(loader.dataset))
        return m

    out_dir = Path(cfg["paths"]["output_dir"]) / f"resnet_{task}"
    out_dir.mkdir(parents=True, exist_ok=True)
    acc_key = "subset_accuracy" if multi else "accuracy"

    history, best_f1, best_epoch = [], -1.0, -1
    for epoch in range(1, args.epochs + 1):
        tr_m = run_epoch(loaders["train"], True)
        va_m = run_epoch(loaders["val"], False)
        history.append({"epoch": epoch,
                        "train_loss": tr_m["loss"],
                        "train_acc": tr_m[acc_key],
                        "train_macro_f1": tr_m["macro_f1"],
                        "val_loss": va_m["loss"],
                        "val_acc": va_m[acc_key],
                        "val_macro_f1": va_m["macro_f1"]})
        flag = ""
        if va_m["macro_f1"] > best_f1:
            best_f1, best_epoch = va_m["macro_f1"], epoch
            torch.save({"state_dict": model.state_dict(),
                        "classes": classes, "task": task,
                        "epoch": epoch, "seed": args.seed},
                       out_dir / "best.pt")
            flag = "  <- best"
        print(f"epoch {epoch:02d} | train loss {tr_m['loss']:.4f} "
              f"acc {tr_m[acc_key]:.3f} | val loss {va_m['loss']:.4f} "
              f"acc {va_m[acc_key]:.3f} F1 {va_m['macro_f1']:.3f}{flag}")

    (out_dir / "training_history.json").write_text(
        json.dumps({"task": task, "seed": args.seed,
                    "best_epoch": best_epoch, "epochs": history},
                   indent=2))

    # ---- held-out test evaluation with the best checkpoint ----
    ckpt = torch.load(out_dir / "best.pt", map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ys, ps, files, probs = [], [], [], []
    with torch.no_grad():
        for x, y, fn in loaders["test"]:
            out = model(x.to(device))
            if multi:
                pr = torch.sigmoid(out).cpu().numpy()
                ps.extend((pr >= 0.5).astype(int).tolist())
                ys.extend(y.long().numpy().tolist())
            else:
                pr = torch.softmax(out, 1).cpu().numpy()
                ps.extend(out.argmax(1).cpu().numpy().tolist())
                ys.extend(y.numpy().tolist())
            probs.extend(pr.tolist())
            files.extend(fn)

    metrics = (multi_label_metrics(ys, ps, classes) if multi
               else single_label_metrics(ys, ps, classes))
    metrics.update({"task": task, "n_test": len(files),
                    "best_epoch": best_epoch, "seed": args.seed,
                    "epochs_trained": args.epochs, "lr": args.lr,
                    "batch_size": args.batch_size,
                    "checkpoint_selection": "val_macro_f1"})
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics,
                                                          indent=2))

    preds = []
    for i, fn in enumerate(files):
        if multi:
            preds.append({
                "file_name": fn,
                "predicted": [classes[j] for j, v in enumerate(ps[i]) if v],
                "ground_truth": [classes[j] for j, v in enumerate(ys[i])
                                 if v],
                "probabilities": {classes[j]: round(probs[i][j], 4)
                                  for j in range(len(classes))}})
        else:
            preds.append({
                "file_name": fn,
                "predicted": classes[ps[i]],
                "ground_truth": classes[ys[i]],
                "probabilities": {classes[j]: round(probs[i][j], 4)
                                  for j in range(len(classes))}})
    (out_dir / "test_predictions.json").write_text(json.dumps(preds,
                                                              indent=2))

    # analyzer-style CSV: one row per class
    import csv
    with open(out_dir / f"resnet_{task}_summary.csv", "w",
              newline="") as f:
        wtr = csv.writer(f)
        if multi:
            wtr.writerow(["model", "task", "class", "sensitivity",
                          "specificity", "precision", "f1", "support"])
            for c, m in metrics["per_defect"].items():
                wtr.writerow(["resnet18", task, c,
                              f"{m['sensitivity']:.4f}",
                              f"{m['specificity']:.4f}",
                              f"{m['precision']:.4f}", f"{m['f1']:.4f}",
                              m["support"]])
        else:
            wtr.writerow(["model", "task", "class", "precision", "recall",
                          "f1", "support", "accuracy", "macro_f1"])
            for c, m in metrics["per_class"].items():
                wtr.writerow(["resnet18", task, c, f"{m['precision']:.4f}",
                              f"{m['recall']:.4f}", f"{m['f1']:.4f}",
                              m["support"], f"{metrics['accuracy']:.4f}",
                              f"{metrics['macro_f1']:.4f}"])

    print(f"\nTEST ({len(files)} images, best epoch {best_epoch}):")
    if multi:
        print(f"  subset accuracy {metrics['subset_accuracy']:.3f} | "
              f"macro-F1 {metrics['macro_f1']:.3f}")
        for c, m in metrics["per_defect"].items():
            print(f"  {c:<14} sens {m['sensitivity']:.3f} "
                  f"spec {m['specificity']:.3f} f1 {m['f1']:.3f} "
                  f"(n={m['support']})")
    else:
        print(f"  accuracy {metrics['accuracy']:.3f} | "
              f"macro-F1 {metrics['macro_f1']:.3f}")
        for c, m in metrics["per_class"].items():
            print(f"  {c:<14} P {m['precision']:.3f} R {m['recall']:.3f} "
                  f"F1 {m['f1']:.3f} (n={m['support']})")
    print(f"\nArtifacts in {out_dir}/")
    print(f"Head-to-head: set subset_manifest for the '{task}' dataset in "
          f"config.yaml to {out_dir / 'test_manifest.txt'} and re-run the "
          f"VLM benchmark into a fresh output_dir.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--task", required=True,
                    choices=list(TASK_CLASSES))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--augment", action="store_true",
                    help="horizontal flip + light color jitter on train")
    ap.add_argument("--class-weights", action="store_true",
                    help="inverse-frequency CE weights (single-label)")
    ap.add_argument("--dry-run", action="store_true",
                    help="load data, build splits, write manifests, exit "
                         "(no torch required)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    classes = TASK_CLASSES[args.task]
    examples = load_task(args.task, cfg)
    tr, va, te = stratified_split(examples, args.task, args.seed)

    print(f"task {args.task} | {len(examples)} images | classes {classes}")
    for name, exs in (("train", tr), ("val", va), ("test", te)):
        print(f"  {name:<5} {len(exs):>4}  {_dist(exs, args.task)}")

    out_dir = Path(cfg["paths"]["output_dir"]) / f"resnet_{args.task}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "class_map.json").write_text(
        json.dumps({i: c for i, c in enumerate(classes)}, indent=2))
    (out_dir / "splits.json").write_text(json.dumps(
        {"seed": args.seed, "fractions": SPLIT_FRACS,
         "train": [e.file_name for e in tr],
         "val": [e.file_name for e in va],
         "test": [e.file_name for e in te]}, indent=2))
    (out_dir / "test_manifest.txt").write_text(
        "\n".join(e.path.name for e in te) + "\n")

    if args.dry_run:
        print(f"dry run: splits + manifests written to {out_dir}/")
        return
    train(args, cfg, examples, classes, (tr, va, te))


if __name__ == "__main__":
    main()
