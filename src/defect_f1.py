"""Compute per-defect F1 for each VLM from results/raw/*__defects__logit_per_defect.jsonl.

For each model x defect, the stored per-defect likelihood score is thresholded
over tau in np.arange(0.05, 1.0, 0.05) (the same sweep as
src/analyze_results.py). The tau that maximizes balanced accuracy
(mean of sensitivity and specificity) is selected, and F1 is reported at
that operating point together with the TP/FP/FN/TN counts.

Usage (from the repo root):
    python src/defect_f1.py
    python src/defect_f1.py --raw results/raw --out results/defect_f1.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFECT_CLASSES = ["mold", "visible_cut", "discoloration", "bruising",
                  "wrinkling", "leaking"]
MODEL_ORDER = ["llava15_7b", "qwen2_vl_7b", "qwen25_vl_7b", "internvl3_8b",
               "llama32_11b_vision"]
MODEL_LABEL = {"llava15_7b": "LLaVA-1.5 7B", "qwen2_vl_7b": "Qwen2-VL 7B",
               "qwen25_vl_7b": "Qwen2.5-VL 7B", "internvl3_8b": "InternVL3 8B",
               "llama32_11b_vision": "Llama-3.2 11B-V"}
DEFECT_LABEL = {"mold": "Mold", "visible_cut": "Visible cut",
                "discoloration": "Discoloration", "bruising": "Bruising",
                "wrinkling": "Wrinkling", "leaking": "Leaking"}


def load_jsonl(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def defect_f1(raw_dir: Path) -> pd.DataFrame:
    taus = np.round(np.arange(0.05, 1.0, 0.05), 2)
    out = []
    for path in sorted(raw_dir.glob("*__defects__logit_per_defect.jsonl")):
        model = path.name.split("__")[0]
        recs = [r for r in load_jsonl(path) if r.get("defect_scores")]
        for cls in DEFECT_CLASSES:
            scores = np.array([r["defect_scores"][cls] for r in recs
                               if cls in r["defect_scores"]], dtype=float)
            truths = np.array([cls in (r["gt"] or []) for r in recs
                               if cls in r["defect_scores"]], dtype=bool)
            n_pos, n_neg = int(truths.sum()), int((~truths).sum())
            if n_pos == 0:
                continue
            best = None
            for t in taus:
                pred = scores >= t
                tp = int((pred & truths).sum())
                fp = int((pred & ~truths).sum())
                fn = int((~pred & truths).sum())
                tn = int((~pred & ~truths).sum())
                sens = tp / n_pos
                spec = tn / n_neg if n_neg else float("nan")
                bal_acc = 0.5 * (sens + spec)
                denom = 2 * tp + fp + fn
                f1 = 2 * tp / denom if denom else 0.0
                # strict '>' keeps the first (lowest) tau on ties, matching
                # pandas idxmax behaviour in analyze_results.py
                if best is None or bal_acc > best["bal_acc"]:
                    best = dict(model=model, defect=cls, tau_star=float(t),
                                n_pos=n_pos, n_neg=n_neg,
                                tp=tp, fp=fp, fn=fn, tn=tn,
                                sensitivity=round(sens, 4),
                                specificity=round(spec, 4),
                                bal_acc=round(bal_acc, 4),
                                f1=round(f1, 6))
            out.append(best)
    df = pd.DataFrame(out)
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    df["defect"] = pd.Categorical(df["defect"], DEFECT_CLASSES, ordered=True)
    return df.sort_values(["model", "defect"]).reset_index(drop=True)


def latex_table(df: pd.DataFrame) -> str:
    models = [m for m in MODEL_ORDER if m in set(df.model)]
    piv = df.pivot(index="defect", columns="model", values="f1")
    counts = df.groupby("defect", observed=True)[["n_pos", "n_neg"]].first()
    lines = [
        r"\begin{tabular}{@{}lcc" + "c" * len(models) + r"@{}}",
        r"\toprule",
        r"\textbf{Defect} & $n^{+}$ & $n^{-}$ & "
        + " & ".join(rf"\textbf{{{MODEL_LABEL[m]}}}" for m in models) + r" \\",
        r"\midrule",
    ]
    for d in DEFECT_CLASSES:
        if d not in piv.index:
            continue
        row = piv.loc[d, models]
        best = row.max()
        cells = [rf"\textbf{{{v:.3f}}}" if v == best else f"{v:.3f}"
                 for v in row]
        lines.append(f"{DEFECT_LABEL[d]} & {counts.loc[d, 'n_pos']} & "
                     f"{counts.loc[d, 'n_neg']} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw", type=Path)
    ap.add_argument("--out", default="results/defect_f1.csv", type=Path)
    args = ap.parse_args()

    df = defect_f1(args.raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(f"\nWrote {args.out}\n")
    print(latex_table(df))


if __name__ == "__main__":
    main()
