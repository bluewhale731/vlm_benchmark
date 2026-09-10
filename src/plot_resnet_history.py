"""Plot ResNet training curves from training_history.json.

For each task, produces two figures in results/resnet_{task}/:
  resnet_{task}_accuracy.png/.pdf   train vs. val accuracy per epoch
  resnet_{task}_f1.png/.pdf         train vs. val macro-F1 per epoch
The epoch selected as the best checkpoint (val macro-F1) is marked.

Usage (from the vlm_benchmark root):
  python -m src.plot_resnet_history config.yaml
  python -m src.plot_resnet_history config.yaml --tasks donation_type
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

TITLES = {
    "donation_type": "Donation Type Classification (ResNet-18)",
    "defects": "Defect Detection (ResNet-18, multi-label)",
    "freshness": "Produce Freshness Classification (ResNet-18)",
}

ACC_LABEL = {
    "donation_type": "Accuracy",
    "freshness": "Accuracy",
    "defects": "Subset accuracy",
}


def plot_task(task: str, out_dir: Path) -> bool:
    hist_path = out_dir / "training_history.json"
    if not hist_path.exists():
        print(f"[skip] {task}: {hist_path} not found (train first)")
        return False
    h = json.loads(hist_path.read_text())
    epochs = [e["epoch"] for e in h["epochs"]]
    best = h.get("best_epoch")
    title = TITLES.get(task, task)

    def one_figure(train_key, val_key, ylabel, stem):
        tr = [e[train_key] for e in h["epochs"]]
        va = [e[val_key] for e in h["epochs"]]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(epochs, tr, label=f"Train {ylabel.lower()}",
                color="#1f77b4", linewidth=1.8)
        ax.plot(epochs, va, label=f"Validation {ylabel.lower()}",
                color="#d62728", linewidth=1.8)
        if best in epochs:
            i = epochs.index(best)
            ax.axvline(best, color="gray", linestyle="--", linewidth=1,
                       alpha=0.7)
            ax.scatter([best], [va[i]], color="#d62728", zorder=5, s=35)
            ax.annotate(f"best epoch {best}\n(val {va[i]:.3f})",
                        xy=(best, va[i]), xytext=(8, -14),
                        textcoords="offset points", fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"resnet_{task}_{stem}.{ext}", dpi=300)
        plt.close(fig)
        print(f"  {out_dir / f'resnet_{task}_{stem}.png'}")

    print(f"{task}:")
    one_figure("train_acc", "val_acc", ACC_LABEL.get(task, "Accuracy"),
               "accuracy")
    one_figure("train_macro_f1", "val_macro_f1", "Macro-F1", "f1")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--tasks", nargs="+",
                    default=["donation_type", "defects"],
                    choices=["donation_type", "defects", "freshness"])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    base = Path(cfg["paths"]["output_dir"])
    for task in args.tasks:
        plot_task(task, base / f"resnet_{task}")


if __name__ == "__main__":
    main()
