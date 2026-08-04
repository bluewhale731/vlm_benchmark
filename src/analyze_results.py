"""Aggregate raw JSONL results into paper-ready tables.

Outputs (under results/):
  summary_single_label.csv     accuracy / macro-F1 / per-class P,R,F1 and
                               spoiled recall for donation_type + freshness
  summary_defects.csv          per-defect sensitivity + specificity per
                               model x strategy
  mcnemar_freshness.csv        exact McNemar tests between strategies
                               within each model (freshness task)
  tau_sweep.csv                cascade accuracy / spoiled recall across
                               (tau_spoil, tau_degrad) grid, per model
  tables.tex                   LaTeX versions of the main tables

Usage:
    python -m src.analyze_results config.yaml
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest
from sklearn.metrics import (confusion_matrix, f1_score,
                             precision_recall_fscore_support)

FRESHNESS_CLASSES = ["fresh", "edible_soon", "spoiled"]
DONATION_CLASSES = ["packaged", "produce", "bakery"]
DEFECT_CLASSES = ["wrinkling", "visible_cut", "bruising",
                  "discoloration", "leaking", "mold"]


def load_raw(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(raw_dir.glob("*.jsonl")):
        for line in p.open():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        sys.exit(f"No results found under {raw_dir}")
    df = pd.DataFrame(rows)
    # keep the last record per (model, task, strategy, file) — resume-safe
    df = df.drop_duplicates(subset=["model", "task", "strategy", "file_name"],
                            keep="last")
    return df


# ----------------------------------------------------------------
# Single-label tasks
# ----------------------------------------------------------------
def single_label_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (model, task, strat), g in df[df.task.isin(
            ["donation_type", "freshness"])].groupby(
            ["model", "task", "strategy"]):
        classes = (FRESHNESS_CLASSES if task == "freshness"
                   else DONATION_CLASSES)
        g = g[g.pred.notna()]
        y_true, y_pred = g["gt"].tolist(), g.pred.tolist()
        if not y_true:
            continue
        acc = float(np.mean([t == p for t, p in zip(y_true, y_pred)]))
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=classes, average=None, zero_division=0)
        row = dict(model=model, task=task, strategy=strat, n=len(g),
                   accuracy=round(acc, 4),
                   macro_f1=round(f1_score(y_true, y_pred, labels=classes,
                                           average="macro",
                                           zero_division=0), 4))
        for i, c in enumerate(classes):
            row[f"{c}_precision"] = round(p[i], 4)
            row[f"{c}_recall"] = round(r[i], 4)
            row[f"{c}_f1"] = round(f1[i], 4)
        if "parse_ok" in g:
            row["parse_fail_rate"] = round(
                1 - g.parse_ok.fillna(True).mean(), 4)
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        row["confusion_matrix"] = json.dumps(cm.tolist())
        out.append(row)
    return pd.DataFrame(out).sort_values(["task", "model", "strategy"])


# ----------------------------------------------------------------
# Defects: per-class sensitivity/specificity
# ----------------------------------------------------------------
def defects_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (model, strat), g in df[df.task == "defects"].groupby(
            ["model", "strategy"]):
        g = g[g.pred.notna()]
        for cls in DEFECT_CLASSES:
            tp = fn = fp = tn = 0
            for _, r in g.iterrows():
                truth = cls in (r["gt"] or [])
                pred = cls in (r.pred or [])
                tp += truth and pred
                fn += truth and not pred
                fp += pred and not truth
                tn += (not truth) and (not pred)
            n_pos = tp + fn
            out.append(dict(
                model=model, strategy=strat, defect=cls,
                annotated_count=n_pos,
                sensitivity=round(tp / n_pos, 4) if n_pos else np.nan,
                specificity=round(tn / (tn + fp), 4) if (tn + fp) else np.nan,
                precision=round(tp / (tp + fp), 4) if (tp + fp) else np.nan,
            ))
    return pd.DataFrame(out)


# ----------------------------------------------------------------
# Defect threshold sweep (re-derives per-defect predictions from the
# stored yes/no likelihood scores; no re-inference needed)
# ----------------------------------------------------------------
def defect_tau_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """For each model x defect, sweep tau over the stored logit scores and
    report sensitivity/specificity at each threshold plus the best
    operating point by Youden's J (sens + spec - 1)."""
    out = []
    g_all = df[(df.task == "defects") & (df.strategy == "logit_per_defect")]
    taus = np.round(np.arange(0.05, 1.0, 0.05), 2)
    for model, g in g_all.groupby("model"):
        g = g[g.defect_scores.notna()]
        if g.empty:
            continue
        for cls in DEFECT_CLASSES:
            scores, truths = [], []
            for _, r in g.iterrows():
                s = (r.defect_scores or {}).get(cls)
                if s is None:
                    continue
                scores.append(s)
                truths.append(cls in (r["gt"] or []))
            scores = np.asarray(scores)
            truths = np.asarray(truths)
            n_pos, n_neg = int(truths.sum()), int((~truths).sum())
            if n_pos == 0:
                continue
            for t in taus:
                pred = scores >= t
                sens = float((pred & truths).sum() / n_pos)
                spec = float((~pred & ~truths).sum() / n_neg) if n_neg else np.nan
                out.append(dict(model=model, defect=cls, tau=t,
                                n_pos=n_pos, n_neg=n_neg,
                                sensitivity=round(sens, 4),
                                specificity=round(spec, 4),
                                youden_j=round(sens + (spec or 0) - 1, 4)))
    return pd.DataFrame(out)


def defect_calibrated_summary(sweep: pd.DataFrame) -> pd.DataFrame:
    """Best operating point per model x defect by Youden's J."""
    if sweep.empty:
        return sweep
    idx = sweep.groupby(["model", "defect"]).youden_j.idxmax()
    best = sweep.loc[idx].rename(columns={"tau": "tau_star"})
    return best[["model", "defect", "tau_star", "n_pos", "n_neg",
                 "sensitivity", "specificity", "youden_j"]].sort_values(
                     ["model", "defect"])


# ----------------------------------------------------------------
# Exact McNemar between strategies (freshness), within each model
# ----------------------------------------------------------------
def mcnemar_freshness(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    fresh = df[(df.task == "freshness") & df.pred.notna()]
    for model, g in fresh.groupby("model"):
        strats = sorted(g.strategy.unique())
        piv = g.pivot_table(index="file_name", columns="strategy",
                            values="pred", aggfunc="first")
        gt = g.drop_duplicates("file_name").set_index("file_name")["gt"]
        for s1, s2 in combinations(strats, 2):
            sub = piv[[s1, s2]].dropna()
            if sub.empty:
                continue
            c1 = sub[s1].eq(gt.loc[sub.index])
            c2 = sub[s2].eq(gt.loc[sub.index])
            b = int((c1 & ~c2).sum())   # s1 right, s2 wrong
            c = int((~c1 & c2).sum())   # s2 right, s1 wrong
            p = (binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else 1.0)
            out.append(dict(model=model, strategy_a=s1, strategy_b=s2,
                            n=len(sub), a_only_correct=b, b_only_correct=c,
                            mcnemar_exact_p=round(p, 6)))
    return pd.DataFrame(out)


# ----------------------------------------------------------------
# Cascade threshold sweep (re-derives predictions from stored scores)
# ----------------------------------------------------------------
def tau_sweep(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    casc = df[(df.task == "freshness") & (df.strategy == "logit_cascade")
              & df.get("s_spoil", pd.Series(dtype=float)).notna()]
    taus = np.round(np.arange(0.05, 1.0, 0.05), 2)
    for model, g in casc.groupby("model"):
        s_spoil = g.s_spoil.to_numpy()
        s_degrad = g.s_degrad.to_numpy()
        gt = g["gt"].to_numpy()
        for ts in taus:
            for td in taus:
                pred = np.where(s_spoil >= ts, "spoiled",
                                np.where(s_degrad >= td, "edible_soon",
                                         "fresh"))
                acc = float((pred == gt).mean())
                spoiled_mask = gt == "spoiled"
                rec = (float((pred[spoiled_mask] == "spoiled").mean())
                       if spoiled_mask.any() else np.nan)
                out.append(dict(model=model, tau_spoil=ts, tau_degrad=td,
                                accuracy=round(acc, 4),
                                spoiled_recall=round(rec, 4),
                                macro_f1=round(f1_score(
                                    gt, pred, labels=FRESHNESS_CLASSES,
                                    average="macro", zero_division=0), 4)))
    return pd.DataFrame(out)


# ----------------------------------------------------------------
# LaTeX export
# ----------------------------------------------------------------
def latex_tables(single: pd.DataFrame, defects: pd.DataFrame,
                 out_path: Path):
    parts = []

    fresh = single[single.task == "freshness"]
    if not fresh.empty:
        lines = [r"\begin{table}[!htbp]", r"\centering",
                 r"\caption{Produce freshness classification across models "
                 r"and prompting strategies.}",
                 r"\label{tab:freshness_models}",
                 r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
                 r"\textbf{Model} & \textbf{Strategy} & \textbf{Acc.} & "
                 r"\textbf{Macro-F1} & \textbf{Spoiled Rec.} & "
                 r"\textbf{Spoiled Prec.} \\", r"\midrule"]
        for _, r in fresh.iterrows():
            lines.append(
                f"{r.model.replace('_', r'\_')} & "
                f"{r.strategy.replace('_', r'\_')} & "
                f"{r.accuracy * 100:.1f}\\% & {r.macro_f1:.3f} & "
                f"{r.spoiled_recall * 100:.1f}\\% & "
                f"{r.spoiled_precision * 100:.1f}\\% \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        parts.append("\n".join(lines))

    don = single[single.task == "donation_type"]
    if not don.empty:
        lines = [r"\begin{table}[!htbp]", r"\centering",
                 r"\caption{Donation category identification accuracy.}",
                 r"\label{tab:donation_models}",
                 r"\begin{tabular}{@{}llcc@{}}", r"\toprule",
                 r"\textbf{Model} & \textbf{Strategy} & \textbf{Acc.} & "
                 r"\textbf{Macro-F1} \\", r"\midrule"]
        for _, r in don.iterrows():
            lines.append(f"{r.model.replace('_', r'\_')} & "
                         f"{r.strategy.replace('_', r'\_')} & "
                         f"{r.accuracy * 100:.1f}\\% & {r.macro_f1:.3f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        parts.append("\n".join(lines))

    if not defects.empty:
        lines = [r"\begin{table}[!htbp]", r"\centering",
                 r"\caption{Zero-shot defect detection sensitivity by model "
                 r"(logit per-defect strategy).}",
                 r"\label{tab:defects_models}"]
        piv = defects[defects.strategy == "logit_per_defect"].pivot_table(
            index="defect", columns="model", values="sensitivity")
        piv = piv.reindex(DEFECT_CLASSES)
        cols = "l" + "c" * len(piv.columns)
        lines += [rf"\begin{{tabular}}{{@{{}}{cols}@{{}}}}", r"\toprule"]
        header = r"\textbf{Defect} & " + " & ".join(
            rf"\textbf{{{c.replace('_', r'\_')}}}" for c in piv.columns)
        lines += [header + r" \\", r"\midrule"]
        for defect, row in piv.iterrows():
            cells = " & ".join("--" if pd.isna(v) else f"{v * 100:.1f}\\%"
                               for v in row)
            lines.append(f"{defect.replace('_', r'\_')} & {cells} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        parts.append("\n".join(lines))

    out_path.write_text("\n\n".join(parts))


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    res_dir = Path(cfg["paths"]["output_dir"])
    df = load_raw(res_dir / "raw")

    n_err = df.get("error", pd.Series(dtype=object)).notna().sum()
    if n_err:
        print(f"[warn] {n_err} records contain errors (excluded from "
              f"metrics); inspect results/raw/*.jsonl")

    single = single_label_summary(df)
    single.to_csv(res_dir / "summary_single_label.csv", index=False)
    print(f"[out] summary_single_label.csv ({len(single)} rows)")

    defs = defects_summary(df)
    defs.to_csv(res_dir / "summary_defects.csv", index=False)
    print(f"[out] summary_defects.csv ({len(defs)} rows)")

    dsweep = defect_tau_sweep(df)
    if not dsweep.empty:
        dsweep.to_csv(res_dir / "defect_tau_sweep.csv", index=False)
        dbest = defect_calibrated_summary(dsweep)
        dbest.to_csv(res_dir / "summary_defects_calibrated.csv", index=False)
        print(f"[out] defect_tau_sweep.csv ({len(dsweep)} rows), "
              f"summary_defects_calibrated.csv ({len(dbest)} rows)")
        print("\nCalibrated defect operating points (Youden's J):")
        print(dbest.to_string(index=False))

    mn = mcnemar_freshness(df)
    mn.to_csv(res_dir / "mcnemar_freshness.csv", index=False)
    print(f"[out] mcnemar_freshness.csv ({len(mn)} rows)")

    sweep = tau_sweep(df)
    sweep.to_csv(res_dir / "tau_sweep.csv", index=False)
    print(f"[out] tau_sweep.csv ({len(sweep)} rows)")
    if not sweep.empty:
        best = sweep.loc[sweep.groupby("model").macro_f1.idxmax()]
        print("\nBest (tau_spoil, tau_degrad) per model by macro-F1:")
        print(best[["model", "tau_spoil", "tau_degrad", "accuracy",
                    "spoiled_recall", "macro_f1"]].to_string(index=False))

    latex_tables(single, defs, res_dir / "tables.tex")
    print("[out] tables.tex")

    # console overview
    if not single.empty:
        cols = ["model", "task", "strategy", "n", "accuracy", "macro_f1"]
        if "spoiled_recall" in single:
            cols.append("spoiled_recall")
        print("\n" + single[cols].to_string(index=False))


if __name__ == "__main__":
    main()