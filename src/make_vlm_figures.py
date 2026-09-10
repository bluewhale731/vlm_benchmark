#!/usr/bin/env python3
"""
make_vlm_figures.py
===================

Generate every vision-language-model figure for the ZeroWaste
donation-assessment manuscript, straight from the raw benchmark records.

    https://github.com/bluewhale731/vlm_benchmark/tree/main/results/raw

Output is written as figure0, figure1, figure2, ... in both PDF (for
Overleaf) and
PNG (for quick viewing), so they can be dropped into an Overleaf project
and referenced positionally:

    \\includegraphics[width=\\textwidth]{figures/figure3.pdf}

A companion figure_captions.tex is also written, containing a ready-made
figure environment for each one with a suggested caption and label.

Every figure here is about the five VLMs. The supervised YOLOv11 and
ResNet-18 references are deliberately not plotted: they are reported as
reference numbers in the text and tables, not as figures.

Usage
-----
    python make_vlm_figures.py                       # expects ./results/raw
    python make_vlm_figures.py --raw path/to/raw --out figures
    python make_vlm_figures.py --list                # show the figure list
    python make_vlm_figures.py --only 3 5            # regenerate a subset
    python make_vlm_figures.py --format pdf          # skip the PNGs

Requires: numpy, pandas, scipy, scikit-learn, matplotlib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

# =====================================================================
# Canonical names, ordering and colours
#
# Classes are always indexed by NAME through these lists, never by
# position in the source data. An earlier version of this analysis
# produced near-chance accuracy because a CSV was written in alphabetical
# class order while the model emitted training order.
# =====================================================================
FRESHNESS_CLASSES = ["fresh", "edible_soon", "spoiled"]
DONATION_CLASSES = ["packaged", "produce", "bakery"]
DEFECT_CLASSES = [
    "wrinkling",
    "visible_cut",
    "bruising",
    "discoloration",
    "leaking",
    "mold",
]

# Ordered by median inference latency, fastest first, so the cost and
# accuracy tension stays visible across every figure.
MODEL_ORDER = [
    "llava15_7b",
    "qwen2_vl_7b",
    "qwen25_vl_7b",
    "internvl3_8b",
    "llama32_11b_vision",
]

MODEL_LABELS = {
    "llava15_7b": "LLaVA-1.5 7B",
    "qwen2_vl_7b": "Qwen2-VL 7B",
    "qwen25_vl_7b": "Qwen2.5-VL 7B",
    "internvl3_8b": "InternVL3 8B",
    "llama32_11b_vision": "Llama-3.2 11B-V",
}

# One colour per model, held fixed everywhere a model appears.
MODEL_COLOURS = {
    "llava15_7b": "#1f77b4",
    "qwen2_vl_7b": "#ff7f0e",
    "qwen25_vl_7b": "#2ca02c",
    "internvl3_8b": "#d62728",
    "llama32_11b_vision": "#9467bd",
}

# One colour per querying strategy, likewise fixed. Free-form strategies
# are cool/warm pastels; likelihood scoring is always green, so the
# reader can find it at a glance in any grouped bar chart.
STRATEGY_COLOURS = {
    "freeform_neutral": "#9ecae1",
    "freeform_inspector": "#fdae6b",
    "freeform_cot": "#c7c7c7",
    "freeform_definitions": "#bcbddc",
    "logit_cascade": "#31a354",
    "logit_multichoice": "#31a354",
}

STRATEGY_LABELS = {
    "freeform_neutral": "Free-form, neutral",
    "freeform_inspector": "Free-form, inspector",
    "freeform_cot": "Free-form, chain of thought",
    "freeform_definitions": "Free-form, definitions",
    "logit_cascade": "Likelihood cascade",
    "logit_multichoice": "Likelihood, multiple choice",
    "freeform_list": "Free-form list",
    "logit_per_defect": "Likelihood, per defect",
}

STRATEGY_ORDER = {
    "donation_type": ["freeform_neutral", "freeform_definitions", "logit_multichoice"],
    "freshness": [
        "freeform_neutral",
        "freeform_inspector",
        "freeform_cot",
        "logit_cascade",
    ],
    "defects": ["freeform_list", "logit_per_defect"],
}

CLASS_LABELS = {
    "fresh": "Fresh",
    "edible_soon": "Edible soon",
    "spoiled": "Spoiled",
    "packaged": "Packaged",
    "produce": "Produce",
    "bakery": "Bakery",
    "wrinkling": "Wrinkling",
    "visible_cut": "Visible cut",
    "bruising": "Bruising",
    "discoloration": "Discoloration",
    "leaking": "Leaking",
    "mold": "Mould",
}

SEQ_CMAP = "viridis"  # continuous fields
MAT_CMAP = "Blues"  # confusion matrices
GREY = "0.35"
TAUS = np.round(np.arange(0.05, 1.0, 0.05), 2)


def style():
    """One rcParams block for the whole script, so every figure matches."""
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


# =====================================================================
# Data loading
# =====================================================================
def load_raw(raw_dir: Path) -> pd.DataFrame:
    """Read every JSONL record under raw_dir into one tidy DataFrame."""
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        sys.exit(
            f"No .jsonl files under {raw_dir}\n"
            f"Clone the benchmark and point --raw at its results/raw folder:\n"
            f"  git clone https://github.com/bluewhale731/vlm_benchmark.git"
        )
    rows = []
    for path in files:
        with path.open() as fh:
            for i, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"[warn] skipping malformed line {path.name}:{i}")
    df = pd.DataFrame(rows)

    # Benchmark runs are resumable, so a key can repeat if a job restarted.
    before = len(df)
    df = df.drop_duplicates(
        subset=["model", "task", "strategy", "file_name"], keep="last"
    )
    if len(df) < before:
        print(f"[info] dropped {before - len(df)} duplicate resume records")

    if "error" in df.columns:
        n_err = int(df["error"].notna().sum())
        if n_err:
            print(f"[warn] excluding {n_err} records with an error field")
            df = df[df["error"].isna()]

    # The defect task writes file_name with an images/ prefix; strip it so
    # keys join across tasks.
    df["file_name"] = (
        df["file_name"].astype(str).str.replace(r"^images/", "", regex=True)
    )
    return df.reset_index(drop=True)


def models_for(df: pd.DataFrame, task: str) -> list[str]:
    present = set(df[df.task == task].model)
    return [m for m in MODEL_ORDER if m in present] + sorted(present - set(MODEL_ORDER))


def strategies_for(df: pd.DataFrame, task: str) -> list[str]:
    present = set(df[df.task == task].strategy)
    known = STRATEGY_ORDER.get(task, [])
    return [s for s in known if s in present] + sorted(present - set(known))


def cell(df: pd.DataFrame, model: str, task: str, strategy: str) -> pd.DataFrame:
    sub = df[(df.model == model) & (df.task == task) & (df.strategy == strategy)]
    return sub.sort_values("file_name").reset_index(drop=True)


# =====================================================================
# Metrics
# =====================================================================
def per_class(y_true, y_pred, classes) -> dict:
    y_true, y_pred = list(y_true), list(y_pred)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(classes), average=None, zero_division=0
    )
    out = {
        "accuracy": float(np.mean([a == b for a, b in zip(y_true, y_pred)])),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=list(classes), average="macro",
                     zero_division=0)
        ),
        "cm": confusion_matrix(y_true, y_pred, labels=list(classes)),
    }
    for i, c in enumerate(classes):
        out[f"{c}_precision"] = float(p[i])
        out[f"{c}_recall"] = float(r[i])
        out[f"{c}_f1"] = float(f[i])
        out[f"{c}_support"] = int(s[i])
    return out


def majority_floor(y_true, classes) -> dict:
    maj = pd.Series(list(y_true)).value_counts().idxmax()
    return per_class(y_true, [maj] * len(y_true), classes)


def cascade_probs(s_spoil: float, s_degrade: float) -> dict:
    """Class probabilities induced by two independent binary questions."""
    return {
        "spoiled": s_spoil,
        "edible_soon": (1.0 - s_spoil) * s_degrade,
        "fresh": (1.0 - s_spoil) * (1.0 - s_degrade),
    }


def cascade_predict(s_spoil, s_degrade, tau_s=0.5, tau_d=0.5):
    return np.where(
        s_spoil >= tau_s,
        "spoiled",
        np.where(s_degrade >= tau_d, "edible_soon", "fresh"),
    )


def cascade_confidence(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Predicted-class probability and correctness for a cascade cell."""
    probs = [cascade_probs(r.s_spoil, r.s_degrad) for r in g.itertuples()]
    conf = np.array([p[q] for p, q in zip(probs, g["pred"])])
    corr = np.array([t == p for t, p in zip(g["gt"], g["pred"])])
    return conf, corr


def ece(conf: np.ndarray, corr: np.ndarray, n_bins: int = 10) -> float:
    e = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if m.sum():
            e += m.sum() / len(conf) * abs(corr[m].mean() - conf[m].mean())
    return float(e)


def youden_best(truths: np.ndarray, scores: np.ndarray, taus=TAUS) -> dict:
    """Youden-optimal threshold.

    Selected on the same items it is evaluated on, so the resulting
    sensitivity/specificity pair is optimistic. The manuscript says so
    wherever these numbers appear.
    """
    n_pos, n_neg = int(truths.sum()), int((~truths).sum())
    best = None
    for t in taus:
        pred = scores >= t
        sens = (pred & truths).sum() / n_pos if n_pos else np.nan
        spec = ((~pred) & (~truths)).sum() / n_neg if n_neg else np.nan
        j = sens + spec - 1
        if best is None or j > best["youden_j"]:
            best = {
                "tau": float(t),
                "sensitivity": float(sens),
                "specificity": float(spec),
                "youden_j": float(j),
                "n_pos": n_pos,
                "n_neg": n_neg,
            }
    return best


def defect_scores(df: pd.DataFrame, model: str, defect: str):
    g = df[
        (df.task == "defects")
        & (df.strategy == "logit_per_defect")
        & (df.model == model)
    ]
    scores, truths = [], []
    for r in g.itertuples():
        sc = (r.defect_scores or {}).get(defect)
        if sc is None:
            continue
        scores.append(sc)
        truths.append(defect in (r.gt or []))
    return np.asarray(scores), np.asarray(truths)


# =====================================================================
# Saving
# =====================================================================
def _escape_tex_percent(text: str) -> str:
    """Escape percent signs that are not already escaped."""
    out, prev = [], ""
    for ch in text:
        out.append("\\%" if ch == "%" and prev != "\\" else ch)
        prev = ch
    return "".join(out)


class Saver:
    """Writes figureN.pdf / figureN.png and records the caption."""

    def __init__(self, out_dir: Path, formats=("pdf", "png"), dpi=330):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.formats = formats
        self.dpi = dpi
        self.manifest: list[dict] = []

    def save(self, fig, number: int, slug: str, caption: str, label: str,
             width: str = r"\textwidth"):
        for ext in self.formats:
            p = self.out_dir / f"figure{number}.{ext}"
            fig.savefig(p, dpi=self.dpi)
        plt.close(fig)
        self.manifest.append(
            {
                "number": number,
                "file": f"figure{number}",
                "slug": slug,
                "caption": caption,
                "label": label,
                "width": width,
            }
        )
        exts = "/".join(self.formats)
        print(f"[figure {number}] figure{number}.{{{exts}}}  ({slug})")

    def write_captions(self):
        lines = [
            "% Auto-generated by make_vlm_figures.py. Do not edit by hand.",
            "% Copy the environments you need into the manuscript, or",
            "% \\input this file where the figures should appear.",
            "%",
            "% Figures are numbered in the order they are intended to appear.",
            "% LaTeX renumbers automatically, so cross-reference with the",
            "% \\label keys below rather than with the file numbers.",
            "",
        ]
        for m in sorted(self.manifest, key=lambda x: x["number"]):
            # A bare % starts a comment in LaTeX and would swallow the rest
            # of the caption, including its closing brace, producing a
            # "File ended while scanning" error far from the real cause.
            caption = _escape_tex_percent(m["caption"])
            lines += [
                f"% ---- figure{m['number']}: {m['slug']} ----",
                r"\begin{figure}[!htbp]",
                r"\centering",
                f"\\includegraphics[width={m['width']}]"
                f"{{figures/{m['file']}.pdf}}",
                f"\\caption{{{caption}}}",
                f"\\label{{{m['label']}}}",
                r"\end{figure}",
                "",
            ]
        p = self.out_dir / "figure_captions.tex"
        p.write_text("\n".join(lines))
        print(f"[captions] {p}")

        idx = self.out_dir / "figure_index.csv"
        pd.DataFrame(self.manifest).to_csv(idx, index=False)
        print(f"[index]    {idx}")


# =====================================================================
# FIGURE 1  Donation category identification
# =====================================================================
def figure1(df, sv: Saver, n: int):
    """Accuracy and macro-F1 for the coarse recognition task.

    Both panels are shown because they diverge: bakery has 22 examples, so
    a model can score high accuracy while failing that class entirely, and
    only macro-F1 exposes it.
    """
    style()
    task = "donation_type"
    models, strategies = models_for(df, task), strategies_for(df, task)

    acc = np.full((len(models), len(strategies)), np.nan)
    mf1 = np.full((len(models), len(strategies)), np.nan)
    for i, m in enumerate(models):
        for j, s in enumerate(strategies):
            g = cell(df, m, task, s)
            if g.empty:
                continue
            r = per_class(g["gt"], g["pred"], DONATION_CLASSES)
            acc[i, j] = r["accuracy"]
            mf1[i, j] = r["macro_f1"]

    ref = cell(df, models[0], task, strategies[0])
    floor = majority_floor(ref["gt"], DONATION_CLASSES)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    x = np.arange(len(models))
    w = 0.8 / len(strategies)
    off = (len(strategies) - 1) / 2

    for ax, dat, ylab, hline in (
        (axes[0], acc, "Accuracy", floor["accuracy"]),
        (axes[1], mf1, "Macro-$F_1$", floor["macro_f1"]),
    ):
        for j, s in enumerate(strategies):
            ax.bar(
                x + (j - off) * w, dat[:, j], w,
                label=STRATEGY_LABELS.get(s, s),
                color=STRATEGY_COLOURS.get(s, "#cccccc"),
                edgecolor="k", linewidth=0.4,
            )
        ax.axhline(hline, ls="--", c=GREY, lw=1)
        ax.text(len(models) - 0.55, hline + 0.015, "majority-class floor",
                fontsize=7, ha="right", color="0.3")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models],
                           rotation=25, ha="right")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1.05)
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()

    sv.save(
        fig, n, "donation_category",
        "Donation category identification across five vision--language "
        "models and two querying strategies ($n = 414$). Left: accuracy. "
        "Right: macro-$F_1$. Dashed lines mark the majority-class floor. "
        "The two panels diverge because the bakery class has only 22 "
        "examples, so a model can reach high accuracy while recovering "
        "that class poorly.",
        "fig:donation",
    )


# =====================================================================
# FIGURE 2  Produce condition: macro-F1 and spoiled recall
# =====================================================================
def figure2(df, sv: Saver, n: int):
    """The main condition-assessment result.

    The floor line matters here: on a 62/19/19 split a macro-F1 in the
    0.3s looks respectable until 0.255 is drawn beside it.
    """
    style()
    task = "freshness"
    models, strategies = models_for(df, task), strategies_for(df, task)

    mf1 = np.full((len(models), len(strategies)), np.nan)
    rec = np.full((len(models), len(strategies)), np.nan)
    for i, m in enumerate(models):
        for j, s in enumerate(strategies):
            g = cell(df, m, task, s)
            if g.empty:
                continue
            r = per_class(g["gt"], g["pred"], FRESHNESS_CLASSES)
            mf1[i, j] = r["macro_f1"]
            rec[i, j] = r["spoiled_recall"]

    ref = cell(df, models[0], task, strategies[0])
    floor = majority_floor(ref["gt"], FRESHNESS_CLASSES)["macro_f1"]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    x = np.arange(len(models))
    w = 0.8 / len(strategies)
    off = (len(strategies) - 1) / 2

    for ax, dat, ylab, hline in (
        (axes[0], mf1, "Macro-$F_1$", floor),
        (axes[1], rec, "Spoiled recall", None),
    ):
        for j, s in enumerate(strategies):
            ax.bar(
                x + (j - off) * w, dat[:, j], w,
                label=STRATEGY_LABELS.get(s, s),
                color=STRATEGY_COLOURS.get(s, "#cccccc"),
                edgecolor="k", linewidth=0.4,
            )
        if hline is not None:
            ax.axhline(hline, ls="--", c=GREY, lw=1)
            ax.text(len(models) - 0.55, hline + 0.012, "majority-class floor",
                    fontsize=7, ha="right", color="0.3")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models],
                           rotation=25, ha="right")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1.0)
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()

    sv.save(
        fig, n, "condition_grid",
        "Produce condition assessment across five vision--language models "
        "and three querying strategies ($n = 174$). Left: macro-$F_1$, with "
        "the dashed line marking the 0.255 majority-class floor. Right: "
        "recall on the spoiled class. The inspector persona raises spoiled "
        "recall in three of the five models, but does so by eliminating the "
        "intermediate edible-soon class.",
        "fig:grid",
    )


# =====================================================================
# FIGURE 3  Per-class F1 heatmap
# =====================================================================
def figure3(df, sv: Saver, n: int):
    """Where the condition task actually fails.

    A macro-F1 bar hides which class is responsible. This makes the
    edible-soon column visibly empty across almost every configuration.
    """
    style()
    task = "freshness"
    models, strategies = models_for(df, task), strategies_for(df, task)

    rows, ylabels = [], []
    for m in models:
        for s in strategies:
            g = cell(df, m, task, s)
            if g.empty:
                continue
            r = per_class(g["gt"], g["pred"], FRESHNESS_CLASSES)
            rows.append([r[f"{c}_f1"] for c in FRESHNESS_CLASSES])
            ylabels.append(f"{MODEL_LABELS.get(m, m)}  ·  "
                           f"{STRATEGY_LABELS.get(s, s)}")
    mat = np.array(rows)

    fig, ax = plt.subplots(figsize=(5.6, 0.30 * len(rows) + 1.5))
    im = ax.imshow(mat, cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="w" if mat[i, j] < 0.55 else "k")
    ax.set_xticks(range(len(FRESHNESS_CLASSES)))
    ax.set_xticklabels([CLASS_LABELS[c] for c in FRESHNESS_CLASSES])
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel("Annotated class")
    fig.colorbar(im, ax=ax, label="Per-class $F_1$", fraction=0.046, pad=0.03)
    fig.tight_layout()

    sv.save(
        fig, n, "per_class_f1",
        "Per-class $F_1$ for every model and querying strategy on the "
        "produce-condition task. The edible-soon column is the dominant "
        "failure: it never exceeds 0.33 and falls to exactly zero in four "
        "configurations, whereas fresh is recovered reliably throughout.",
        "fig:perclass", width="0.72\\textwidth",
    )


# =====================================================================
# FIGURE 4  Confusion matrices, persona against cascade
# =====================================================================
def figure4(df, sv: Saver, n: int, model: str | None = None):
    """Two strategies on one model, to show the mechanism.

    The persona does not sharpen the three-way decision, it collapses it
    into a binary one. Side-by-side matrices make that visible in a way
    that a pair of macro-F1 numbers does not.
    """
    style()
    task = "freshness"
    models = models_for(df, task)
    if model is None:
        model = max(
            models,
            key=lambda m: per_class(
                cell(df, m, task, "logit_cascade")["gt"],
                cell(df, m, task, "logit_cascade")["pred"],
                FRESHNESS_CLASSES,
            )["macro_f1"],
        )

    pair = ("freeform_inspector", "logit_cascade")
    labels = [CLASS_LABELS[c] for c in FRESHNESS_CLASSES]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))
    for ax, s in zip(axes, pair):
        g = cell(df, model, task, s)
        cm = per_class(g["gt"], g["pred"], FRESHNESS_CLASSES)["cm"].astype(float)
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap=MAT_CMAP, vmin=0, vmax=1)
        for i in range(len(FRESHNESS_CLASSES)):
            for j in range(len(FRESHNESS_CLASSES)):
                ax.text(j, i, f"{cmn[i, j]:.2f}\n({int(cm[i, j])})",
                        ha="center", va="center", fontsize=7.5,
                        color="w" if cmn[i, j] > 0.55 else "k")
        ax.set_xticks(range(3)); ax.set_xticklabels(labels)
        ax.set_yticks(range(3)); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Annotated")
        ax.set_title(f"{MODEL_LABELS.get(model, model)}, "
                     f"{STRATEGY_LABELS.get(s, s)}")
    fig.tight_layout()

    sv.save(
        fig, n, "confusion_pair",
        f"Row-normalised confusion matrices for {MODEL_LABELS.get(model, model)}, "
        "the strongest model on this task, with raw counts in parentheses. "
        "Left: the rejection-oriented persona reaches high spoiled recall "
        "but assigns no item to edible soon, converting the three-class "
        "decision into a binary one. Right: the likelihood cascade recovers "
        "the intermediate class at the cost of overall accuracy.",
        "fig:confusion",
    )


# =====================================================================
# FIGURE 5  Spoilage ROC
# =====================================================================
def figure5(df, sv: Saver, n: int, target: float = 0.90):
    """The paper's positive result.

    Fixed-threshold decisions are weak, but the underlying ranking is not,
    and a screening deployment uses the ranking.
    """
    style()
    fig, ax = plt.subplots(figsize=(4.3, 4.0))
    for m in models_for(df, "freshness"):
        g = cell(df, m, "freshness", "logit_cascade")
        if g.empty or "s_spoil" not in g.columns:
            continue
        yt = [v == "spoiled" for v in g["gt"]]
        sc = g["s_spoil"].to_numpy()
        fpr, tpr, _ = roc_curve(yt, sc)
        ax.plot(fpr, tpr, lw=1.6, color=MODEL_COLOURS.get(m),
                label=f"{MODEL_LABELS.get(m, m)} (AUC {roc_auc_score(yt, sc):.2f})")
    ax.axhline(target, ls="--", c="0.45", lw=1)
    ax.text(0.02, target + 0.015, f"{target:.0%} sensitivity target",
            fontsize=7, color="0.3")
    ax.plot([0, 1], [0, 1], ls=":", c="0.5", lw=1)
    ax.set_xlabel("False positive rate (sound produce flagged)")
    ax.set_ylabel("True positive rate (spoiled produce flagged)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right", fontsize=7.2)
    fig.tight_layout()

    sv.save(
        fig, n, "spoilage_roc",
        "Receiver operating characteristic curves for spoilage detection "
        "using the continuous cascade score $s_{\\mathrm{spoil}}$ "
        "($n = 174$, of which 33 are spoiled). All five models rank spoiled "
        "produce well above chance even though their fixed-threshold "
        "decisions are weak, which is what makes a ranked review queue "
        "viable where autonomous grading is not.",
        "fig:roc", width="0.58\\textwidth",
    )


# =====================================================================
# FIGURE 6  Screening trade-off at a sensitivity target
# =====================================================================
def figure6(df, sv: Saver, n: int, target: float = 0.90):
    """What the ROC means operationally.

    Specificity retained against the share of the batch a sorter must
    inspect, at a fixed sensitivity target. This is the figure a pantry
    would actually read.
    """
    style()
    rows = []
    for m in models_for(df, "freshness"):
        g = cell(df, m, "freshness", "logit_cascade")
        if g.empty or "s_spoil" not in g.columns:
            continue
        yt = np.array([v == "spoiled" for v in g["gt"]])
        sc = g["s_spoil"].to_numpy()
        fpr, tpr, thr = roc_curve(yt, sc)
        i = int(np.argmax(tpr >= target))
        tau = float(thr[i])
        rows.append(
            {
                "model": m,
                "specificity": float(1 - fpr[i]),
                "flagged": float((sc >= tau).mean()),
                "auroc": roc_auc_score(yt, sc),
            }
        )
    d = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for _, r in d.iterrows():
        ax.scatter(100 * r.flagged, r.specificity, s=95,
                   color=MODEL_COLOURS.get(r.model), edgecolor="k",
                   linewidth=0.6, zorder=3)
        ax.annotate(MODEL_LABELS.get(r.model, r.model),
                    xy=(100 * r.flagged, r.specificity),
                    xytext=(9, 0), textcoords="offset points",
                    fontsize=7.2, ha="left", va="center", color="0.25")
    ax.set_xlabel(f"Batch referred to human review at "
                  f"{target:.0%} sensitivity (%)")
    ax.set_ylabel("Specificity retained")
    ax.set_xlim(20, 105)
    ax.set_ylim(0.2, 0.9)
    ax.grid(True, axis="both", ls=":", lw=0.6, c="0.85")
    ax.set_axisbelow(True)
    fig.tight_layout()

    sv.save(
        fig, n, "screening_tradeoff",
        "Operating points with the cascade threshold set to achieve "
        f"{target * 100:.0f}\\% spoilage sensitivity. The horizontal axis is the "
        "share of the batch a human sorter must inspect and the vertical "
        "axis the specificity retained; upper-left is better. The model "
        "with the best macro-$F_1$ is not the best screen, which is why "
        "the deployment choice should be made on this plot rather than on "
        "a classification metric.",
        "fig:screening", width="0.6\\textwidth",
    )


# =====================================================================
# FIGURE 7  Calibration
# =====================================================================
def figure7(df, sv: Saver, n: int, n_bins: int = 5, min_bin: int = 8):
    """Reliability diagram for the cascade probabilities.

    Only the cascade emits a per-item probability, and calibration is what
    licenses thresholding it.

    Five equal-width bins are used for display, not the ten used for the
    reported ECE: with 174 items a ten-bin diagram is dominated by bins
    holding two or three items, whose observed accuracy swings between 0
    and 1 for reasons that have nothing to do with calibration. Marker
    area is proportional to bin population so the reader can see which
    points carry weight, and bins below min_bin items are dropped.
    """
    style()
    fig, ax = plt.subplots(figsize=(4.8, 4.3))
    edges = np.linspace(0, 1, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2

    for m in models_for(df, "freshness"):
        g = cell(df, m, "freshness", "logit_cascade")
        if g.empty or "s_spoil" not in g.columns:
            continue
        conf, corr = cascade_confidence(g)
        xs, ys, ns = [], [], []
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            msk = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
            if msk.sum() >= min_bin:
                xs.append(centres[i])
                ys.append(corr[msk].mean())
                ns.append(int(msk.sum()))
        if not xs:
            continue
        colour = MODEL_COLOURS.get(m)
        # ECE is quoted at the standard ten bins, matching the tables.
        ax.plot(xs, ys, "-", lw=1.4, color=colour,
                label=f"{MODEL_LABELS.get(m, m)} (ECE {ece(conf, corr):.3f})")
        ax.scatter(xs, ys, s=[6 + 1.4 * v for v in ns], color=colour,
                   edgecolor="k", linewidth=0.4, zorder=3)

    ax.plot([0, 1], [0, 1], ls=":", c="0.5", lw=1)
    ax.annotate("perfect calibration", xy=(0.82, 0.82), xytext=(0.86, 0.72),
                fontsize=7, color="0.4", ha="center",
                arrowprops=dict(arrowstyle="-", color="0.6", lw=0.7))
    ax.set_xlabel("Predicted class probability (5 bins)")
    ax.set_ylabel("Observed accuracy")
    ax.set_xlim(0.3, 1.0); ax.set_ylim(0.2, 1.0)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    ax.text(0.985, 0.03, "marker area $\\propto$ items in bin", fontsize=6.8,
            color="0.4", ha="right", transform=ax.transAxes)
    fig.tight_layout()

    sv.save(
        fig, n, "calibration",
        "Reliability diagram for the likelihood cascade, the only strategy "
        "that yields a per-item probability. Five display bins are used and "
        "marker area is proportional to the number of items in each bin; "
        "the quoted expected calibration error is computed at the standard "
        "ten bins. Points fall below the diagonal throughout, so the "
        "cascade is overconfident, but the ordering is monotone enough for "
        "the probability to serve as a deferral criterion.",
        "fig:calibration", width="0.58\\textwidth",
    )


# =====================================================================
# FIGURE 8  Selective prediction
# =====================================================================
def figure8(df, sv: Saver, n: int):
    """Accuracy against coverage when low-confidence items are deferred.

    The operating mode the paper actually recommends, plotted as a curve
    rather than the three-row table, so the shape of the trade is visible.
    """
    style()
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for m in models_for(df, "freshness"):
        g = cell(df, m, "freshness", "logit_cascade")
        if g.empty or "s_spoil" not in g.columns:
            continue
        conf, corr = cascade_confidence(g)
        order = np.argsort(-conf)  # most confident first
        c_sorted = corr[order]
        k = np.arange(1, len(c_sorted) + 1)
        acc = np.cumsum(c_sorted) / k
        cov = k / len(c_sorted)
        keep = cov >= 0.10  # below 10% coverage the curve is a few items
        ax.plot(100 * cov[keep], acc[keep], lw=1.6,
                color=MODEL_COLOURS.get(m), label=MODEL_LABELS.get(m, m))

    ref = cell(df, models_for(df, "freshness")[0], "freshness", "logit_cascade")
    floor = majority_floor(ref["gt"], FRESHNESS_CLASSES)["accuracy"]
    ax.axhline(floor, ls="--", c=GREY, lw=1)
    ax.text(0.99, floor + 0.010, "majority-class accuracy", fontsize=7,
            ha="right", color="0.3",
            transform=ax.get_yaxis_transform())

    ax.set_xlabel("Coverage: items the model decides (%)")
    ax.set_ylabel("Accuracy on decided items")
    ax.set_xlim(10, 100)
    ax.invert_xaxis()  # left to right = more deferral
    ax.legend(frameon=False, loc="upper left", fontsize=7.2)
    fig.tight_layout()

    sv.save(
        fig, n, "selective_prediction",
        "Selective prediction with the likelihood cascade. Items are ranked "
        "by predicted class probability and the least confident are "
        "deferred to human review, so coverage decreases from right to "
        "left. Accuracy on the retained items rises as coverage falls, "
        "which is the behaviour a deferral policy depends on.",
        "fig:selective", width="0.62\\textwidth",
    )


# =====================================================================
# FIGURE 9  Threshold sensitivity
# =====================================================================
def figure9(df, sv: Saver, n: int, taus=TAUS, k: int = 2):
    """Macro-F1 across the cascade threshold grid.

    Shared colour scale so the panels are comparable. The gap between the
    circle (default) and the star (in-sample optimum) is the point.
    """
    style()
    models = models_for(df, "freshness")
    scored = []
    for m in models:
        g = cell(df, m, "freshness", "logit_cascade")
        if g.empty:
            continue
        scored.append(
            (per_class(g["gt"], g["pred"], FRESHNESS_CLASSES)["macro_f1"], m)
        )
    chosen = [m for _, m in sorted(scored, reverse=True)[:k]]

    sweeps = {}
    for m in chosen:
        g = cell(df, m, "freshness", "logit_cascade")
        ss, sd, yt = (g["s_spoil"].to_numpy(), g["s_degrad"].to_numpy(),
                      g["gt"].to_numpy())
        rows = []
        for ts in taus:
            for td in taus:
                pred = cascade_predict(ss, sd, ts, td)
                rows.append(
                    {
                        "tau_spoil": ts, "tau_degrade": td,
                        "macro_f1": f1_score(yt, pred, labels=FRESHNESS_CLASSES,
                                             average="macro", zero_division=0),
                    }
                )
        sweeps[m] = pd.DataFrame(rows)

    vmin = min(s.macro_f1.min() for s in sweeps.values())
    vmax = max(s.macro_f1.max() for s in sweeps.values())
    step = float(taus[1] - taus[0])
    lo, hi = float(taus[0]) - step / 2, float(taus[-1]) + step / 2

    fig, axes = plt.subplots(1, len(chosen), figsize=(4.2 * len(chosen), 3.6))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, chosen):
        piv = sweeps[m].pivot(index="tau_degrade", columns="tau_spoil",
                              values="macro_f1")
        im = ax.imshow(piv.values, origin="lower", cmap=SEQ_CMAP,
                       aspect="auto", extent=[lo, hi, lo, hi],
                       vmin=vmin, vmax=vmax)
        ax.set_title(MODEL_LABELS.get(m, m))
        ax.set_xlabel(r"$\tau_{\mathrm{spoil}}$")
        ax.set_ylabel(r"$\tau_{\mathrm{degrade}}$")
        ax.scatter([0.5], [0.5], marker="o", s=42, facecolor="none",
                   edgecolor="w", lw=1.4)
        b = sweeps[m].loc[sweeps[m].macro_f1.idxmax()]
        ax.scatter([b.tau_spoil], [b.tau_degrade], marker="*", s=110,
                   c="w", edgecolor="k", lw=0.5)
        fig.colorbar(im, ax=ax, label="macro-$F_1$")
    fig.tight_layout()

    sv.save(
        fig, n, "threshold_sweep",
        "Macro-$F_1$ over the cascade threshold grid for the two strongest "
        "models. The open circle marks the default operating point "
        "$\\tau = (0.5, 0.5)$ and the star the in-sample optimum. "
        "Performance varies more across thresholds within a single model "
        "than it does across models at a fixed threshold, so the threshold "
        "is a primary deployment parameter rather than an implementation "
        "detail.",
        "fig:tau",
    )


# =====================================================================
# FIGURE 10  Per-defect discrimination
# =====================================================================
def figure10(df, sv: Saver, n: int, taus=TAUS):
    """Youden's J per defect and model.

    J rather than sensitivity, because the defect subset holds only
    degraded produce: a model answering Yes to everything scores
    sensitivity 1.00 and J near zero, and the figure should show that.
    """
    style()
    models = models_for(df, "defects")
    grid = {}
    for c in DEFECT_CLASSES:
        col = {}
        for m in models:
            sc, tr = defect_scores(df, m, c)
            if tr.size == 0 or tr.sum() == 0:
                continue
            col[m] = youden_best(tr, sc, taus)["youden_j"]
        if col:
            grid[c] = col
    defects = sorted(grid, key=lambda c: -float(np.mean(list(grid[c].values()))))

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    x = np.arange(len(defects))
    w = 0.8 / max(len(models), 1)
    off = (len(models) - 1) / 2
    for j, m in enumerate(models):
        vals = [grid[c].get(m, np.nan) for c in defects]
        ax.bar(x + (j - off) * w, vals, w, label=MODEL_LABELS.get(m, m),
               color=MODEL_COLOURS.get(m), edgecolor="k", linewidth=0.4)
    ax.axhline(0, c="0.3", lw=1)
    ax.text(len(defects) - 0.45, 0.02, "chance", fontsize=7, ha="right",
            color="0.3")
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_LABELS.get(c, c) for c in defects])
    ax.set_ylabel("Youden's $J$ at optimal $\\tau$")
    ax.set_ylim(-0.15, 1.0)
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=7.2)
    fig.tight_layout()

    sv.save(
        fig, n, "defect_youden",
        "Youden's $J$ at the per-defect optimal threshold, by defect and "
        "model, on the defect-annotated produce subset. Values at or below "
        "zero indicate performance no better than chance. Mould is "
        "identified reliably while bruising and wrinkling are not. "
        "Thresholds are selected in sample, so these values are "
        "optimistic. Leaking is omitted because it has no positive "
        "instances in this subset.",
        "fig:defects",
    )


# =====================================================================
# FIGURE 11  Accuracy against inference cost
# =====================================================================
def figure11(df, sv: Saver, n: int):
    """Condition macro-F1 against median per-query latency.

    Latency spans a twentyfold range across these models while macro-F1
    spans 0.12, so the deployment choice is not simply "take the best".
    """
    style()
    if "latency_s" not in df.columns:
        print("[skip] figure11: no latency_s column in the records")
        return

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    pts = []
    for m in models_for(df, "freshness"):
        lat = df[(df.model == m) & df.latency_s.notna()].latency_s.median()
        best_f1 = -1.0
        for s in strategies_for(df, "freshness"):
            g = cell(df, m, "freshness", s)
            if g.empty:
                continue
            f1 = per_class(g["gt"], g["pred"], FRESHNESS_CLASSES)["macro_f1"]
            best_f1 = max(best_f1, f1)
        pts.append((float(lat), best_f1, m))
    pts.sort()

    # Alternate the label above and below each point: on a log axis the
    # fast models bunch together and centred labels overlap.
    for i, (lat, f1, m) in enumerate(pts):
        ax.scatter(lat, f1, s=100, color=MODEL_COLOURS.get(m),
                   edgecolor="k", linewidth=0.6, zorder=3)
        dy = 12 if i % 2 == 0 else -16
        ax.annotate(MODEL_LABELS.get(m, m), xy=(lat, f1),
                    xytext=(0, dy), textcoords="offset points",
                    fontsize=7.2, ha="center", color="0.25")

    ref = cell(df, models_for(df, "freshness")[0], "freshness", "logit_cascade")
    floor = majority_floor(ref["gt"], FRESHNESS_CLASSES)["macro_f1"]
    ax.axhline(floor, ls="--", c=GREY, lw=1)
    ax.text(0.99, floor + 0.008, "majority-class floor", fontsize=7,
            ha="right", color="0.3", transform=ax.get_yaxis_transform())

    ax.set_xscale("log")
    ax.set_xlabel("Median inference latency per query (s, log scale)")
    ax.set_ylabel("Best macro-$F_1$ on produce condition")
    # Pad both axes so the outermost labels are not clipped.
    lo = min(p[0] for p in pts) / 1.9
    hi = max(p[0] for p in pts) * 1.9
    ax.set_xlim(lo, hi)
    ax.set_ylim(0.2, 0.62)
    ax.grid(True, ls=":", lw=0.6, c="0.85")
    ax.set_axisbelow(True)
    fig.tight_layout()

    sv.save(
        fig, n, "accuracy_vs_latency",
        "Best produce-condition macro-$F_1$ against median inference "
        "latency on one NVIDIA A100 80GB GPU. Latency spans a twentyfold "
        "range while macro-$F_1$ spans roughly 0.12, so throughput rather "
        "than accuracy is likely to govern model choice at food-bank "
        "scale.",
        "fig:latency", width="0.6\\textwidth",
    )


# =====================================================================
# FIGURE 0  Evaluation framework schematic
#
# Numbered 0 because it is not a result: it precedes the results figures
# in the manuscript and replaces the old BILP architecture diagram, which
# described an allocation stage no longer part of this study.
# =====================================================================
def figure0(df, sv: Saver, n: int):
    """Schematic of the evaluation framework.

    Drawn in code rather than a drawing tool so it lives beside the
    analysis and can be edited without a separate diagramming step. Takes
    df only to match the calling convention; the layout is fixed.
    """
    style()
    fig, ax = plt.subplots(figsize=(10.5, 3.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 34)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc, ec="#33333a"):
        ax.add_patch(
            plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                          linewidth=1.1, zorder=2)
        )
        ax.text(x + w / 2, y + h - 3.1, title, ha="center", va="top",
                fontsize=8.6, fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h - 7.4, body, ha="center", va="top",
                fontsize=7.2, zorder=3, linespacing=1.45)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#33333a",
                                    lw=1.1, shrinkA=0, shrinkB=0),
                    zorder=1)

    n_img = df[df.task == "donation_type"].file_name.nunique()

    box(1, 9, 16, 16, "Intake image",
        f"One donated item\nphotographed at\nthe pantry\n\n$n = {n_img}$",
        "#eef2f7")
    arrow(17, 17, 21, 17)

    box(21, 18.5, 24, 14.5, "Five vision-language models",
        "LLaVA-1.5 7B   Qwen2-VL 7B\nQwen2.5-VL 7B   InternVL3 8B\n"
        "Llama-3.2 11B-Vision\n(zero shot, no fine-tuning)", "#e3eef6")
    box(21, 1, 24, 14.5, "Supervised references",
        "YOLOv11-nano\n(donation category)\nResNet-18\n(produce condition)",
        "#f2f0e6")

    arrow(45, 25.5, 49, 20)
    arrow(45, 8, 49, 13)

    box(49, 9, 21, 16, "Querying strategies",
        "Free-form, neutral\nFree-form, inspector\nLikelihood cascade\n"
        "(Equation 1)", "#e8f3e8")
    arrow(70, 17, 74, 17)

    box(74, 9, 25, 16, "Evaluation",
        "Donation category\nProduce condition\nDefect identification\n\n"
        "Accuracy, macro-F1,\nAUROC, calibration", "#f6eaea")

    fig.tight_layout()

    sv.save(
        fig, n, "pipeline",
        "Evaluation framework. A single intake image of one donated item "
        "is presented to five open-weight vision--language models under "
        "three querying strategies (neutral free-form, rejection-oriented "
        "inspector, and the binary likelihood cascade of "
        "Eq.~\\ref{eq:cascade}). Supervised YOLOv11 and ResNet-18 models "
        "trained on the same images provide references. All configurations "
        "are evaluated on donation category assignment, produce condition "
        "assessment, and defect identification.",
        "fig:pipeline",
    )


# =====================================================================
# Registry and CLI
# =====================================================================
FIGURES = {
    0: ("pipeline", figure0),
    1: ("donation_category", figure1),
    2: ("condition_grid", figure2),
    3: ("per_class_f1", figure3),
    4: ("confusion_pair", figure4),
    5: ("spoilage_roc", figure5),
    6: ("screening_tradeoff", figure6),
    7: ("calibration", figure7),
    8: ("selective_prediction", figure8),
    9: ("threshold_sweep", figure9),
    10: ("defect_youden", figure10),
    11: ("accuracy_vs_latency", figure11),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="make_vlm_figures.py",
        description="Generate numbered vision-language-model figures from "
        "the raw benchmark records.",
    )
    ap.add_argument("--raw", default="results/raw",
                    help="Directory of per-cell JSONL records "
                         "(default: results/raw)")
    ap.add_argument("--out", default="figures",
                    help="Output directory (default: figures)")
    ap.add_argument("--only", nargs="+", type=int, metavar="N",
                    help="Generate only these figure numbers")
    ap.add_argument("--format", nargs="+", default=["pdf", "png"],
                    choices=["pdf", "png", "svg", "eps"],
                    help="Output formats (default: pdf png)")
    ap.add_argument("--dpi", type=int, default=330,
                    help="Raster resolution (default: 330)")
    ap.add_argument("--target-sensitivity", type=float, default=0.90,
                    help="Screening sensitivity target (default: 0.90)")
    ap.add_argument("--list", action="store_true",
                    help="List the figures and exit")
    args = ap.parse_args(argv)

    if args.list:
        print("Figure  Slug")
        for num, (slug, _) in sorted(FIGURES.items()):
            print(f"  {num:<5} {slug}")
        return 0

    df = load_raw(Path(args.raw))
    print(f"[load] {len(df)} records, {df.model.nunique()} models, "
          f"{df.task.nunique()} tasks")

    sv = Saver(Path(args.out), formats=tuple(args.format), dpi=args.dpi)
    wanted = set(args.only) if args.only else set(FIGURES)

    for num, (slug, fn) in sorted(FIGURES.items()):
        if num not in wanted:
            continue
        if num in (5, 6):
            fn(df, sv, num, args.target_sensitivity)
        else:
            fn(df, sv, num)

    sv.write_captions()
    print(f"[done] {len(sv.manifest)} figures in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
