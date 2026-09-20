"""Aggregate raw JSONL results into paper-ready tables.

Outputs (under results/):
  summary_single_label.csv     accuracy / macro-F1 / per-class P,R,F1 and
                               spoiled recall for donation_type + freshness
  summary_defects.csv          per-defect precision / recall / F1 per
                               model x strategy, plus a macro-F1 row
  mcnemar_freshness.csv        exact McNemar tests between strategies
                               within each model (freshness task)
  tables.tex                   LaTeX versions of the main tables
  f1_donation_type.csv         F1 only, one row per model x strategy:
  f1_freshness.csv             macro-F1 plus the F1 of every class
  f1_defects.csv               (or defect). One file per task.

All metrics are computed from predicted labels against ground truth.
Nothing here uses likelihood scores, threshold sweeps, AUROC or
calibration.

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
DEFECT_TAU = 0.50   # fixed decision threshold for likelihood scores; never tuned

MODEL_ORDER = ["llava15_7b", "qwen2_vl_7b", "qwen25_vl_7b", "internvl3_8b",
               "llama32_11b_vision"]
STRATEGY_ORDER = ["freeform_neutral", "freeform_definitions",
                  "freeform_inspector", "freeform_cot", "freeform_list",
                  "freeform_brutal_critic", "freeform_indifferent_critic",
                  "freeform_nice_critic",
                  "logit_multichoice", "logit_cascade", "logit_per_defect"]


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
# Defects: per-defect precision / recall / F1 from predicted labels
# ----------------------------------------------------------------
def defects_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per model x strategy x defect, plus a 'macro' row that
    averages F1 over defects with at least one annotated positive
    (leaking has none, so it never enters the average)."""
    out = []
    for (model, strat), g in df[df.task == "defects"].groupby(
            ["model", "strategy"]):
        g = g[g.pred.notna()]
        # Score only images with at least one annotated defect, whether or
        # not drop_empty_gt has been run on the raw files.
        g = g[g["gt"].map(lambda v: bool(v) if isinstance(v, (list, tuple, set))
                          else False)]
        if g.empty:
            continue
        has_scores = "defect_scores" in g.columns
        f1s = []
        for cls in DEFECT_CLASSES:
            tp = fn = fp = 0
            for _, r in g.iterrows():
                truth = cls in (r["gt"] or [])
                scores = r.defect_scores if has_scores else None
                if isinstance(scores, dict) and cls in scores:
                    # likelihood strategy: label = score >= fixed 0.50
                    pred = scores[cls] >= DEFECT_TAU
                else:
                    pred = cls in (r.pred or [])
                tp += truth and pred
                fn += truth and not pred
                fp += pred and not truth
            n_pos = tp + fn
            if n_pos:
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec = tp / n_pos
                f1 = 2 * tp / (2 * tp + fp + fn)
                f1s.append(f1)
            else:                       # no positives: F1 undefined
                prec = rec = f1 = np.nan
            out.append(dict(
                model=model, strategy=strat, defect=cls, n=len(g),
                annotated_count=n_pos, tp=tp, fp=fp, fn=fn,
                precision=round(prec, 4), recall=round(rec, 4),
                f1=round(f1, 4)))
        out.append(dict(
            model=model, strategy=strat, defect="macro", n=len(g),
            annotated_count=np.nan, tp=np.nan, fp=np.nan, fn=np.nan,
            precision=np.nan, recall=np.nan,
            f1=round(float(np.mean(f1s)), 4) if f1s else np.nan))
    res = pd.DataFrame(out)
    if not res.empty:
        for c in ("annotated_count", "tp", "fp", "fn"):
            res[c] = res[c].astype("Int64")
    return res


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
# F1-only CSVs, one per task
# ----------------------------------------------------------------
def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    def rank(values, order):
        return values.map(lambda v: order.index(v) if v in order
                          else len(order))
    return (df.assign(_m=rank(df.model, MODEL_ORDER),
                      _s=rank(df.strategy, STRATEGY_ORDER))
              .sort_values(["_m", "model", "_s", "strategy"])
              .drop(columns=["_m", "_s"]).reset_index(drop=True))


def f1_csvs(single: pd.DataFrame, defects: pd.DataFrame, res_dir: Path):
    """Write f1_<task>.csv: model, strategy, n, macro_f1, then one F1
    column per class (or per defect). Defects with no annotated positives
    (leaking) get no column."""
    for task, classes in (("donation_type", DONATION_CLASSES),
                          ("freshness", FRESHNESS_CLASSES)):
        sub = single[single.task == task] if not single.empty else single
        if sub.empty:
            print(f"[skip] f1_{task}.csv: no records")
            continue
        out = sub[["model", "strategy", "n", "macro_f1"]
                  + [f"{c}_f1" for c in classes]]
        out = out.rename(columns={f"{c}_f1": f"f1_{c}" for c in classes})
        _ordered(out).to_csv(res_dir / f"f1_{task}.csv", index=False)
        print(f"[out] f1_{task}.csv ({len(out)} rows)")

    if defects.empty:
        print("[skip] f1_defects.csv: no records")
        return
    wide = defects.pivot_table(index=["model", "strategy"], columns="defect",
                               values="f1", dropna=False)
    n = defects.groupby(["model", "strategy"]).n.first()
    keep = [d for d in DEFECT_CLASSES
            if d in wide.columns and wide[d].notna().any()]
    out = pd.DataFrame({"n": n, "macro_f1": wide["macro"]})
    for d in keep:
        out[f"f1_{d}"] = wide[d]
    out = _ordered(out.reset_index())
    out.to_csv(res_dir / "f1_defects.csv", index=False)
    print(f"[out] f1_defects.csv ({len(out)} rows)")


# ----------------------------------------------------------------
# LaTeX export
# ----------------------------------------------------------------
def tex(s) -> str:
    """Escape underscores for LaTeX. Kept outside the f-strings because
    Python < 3.12 does not allow a backslash inside an f-string expression."""
    return str(s).replace("_", r"\_")


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
                f"{tex(r.model)} & "
                f"{tex(r.strategy)} & "
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
            lines.append(f"{tex(r.model)} & "
                         f"{tex(r.strategy)} & "
                         f"{r.accuracy * 100:.1f}\\% & {r.macro_f1:.3f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        parts.append("\n".join(lines))

    strat_names = {"freeform_list": "free-form list strategy",
                   "logit_per_defect": "per-defect likelihood strategy, "
                                       "fixed threshold $p \\geq 0.50$"}
    for strat in ([] if defects.empty else sorted(defects.strategy.unique())):
        piv = defects[defects.strategy == strat].pivot_table(
            index="defect", columns="model", values="f1")
        piv = piv.reindex(DEFECT_CLASSES + ["macro"]).dropna(how="all")
        if piv.empty:
            continue
        cols = "l" + "c" * len(piv.columns)
        lines = [r"\begin{table}[!htbp]", r"\centering",
                 r"\caption{Zero-shot per-defect $F_1$ by model ("
                 + strat_names.get(strat, strat.replace("_", r"\_"))
                 + r"). Defects with no annotated positives are omitted.}",
                 rf"\label{{tab:defects_f1_{strat}}}",
                 rf"\begin{{tabular}}{{@{{}}{cols}@{{}}}}", r"\toprule"]
        header = r"\textbf{Defect} & " + " & ".join(
            rf"\textbf{{{tex(c)}}}" for c in piv.columns)
        lines += [header + r" \\", r"\midrule"]
        for defect, row in piv.iterrows():
            if defect == "macro":
                lines.append(r"\midrule")
            name = "Macro-$F_1$" if defect == "macro" else defect.replace(
                "_", r"\_")
            cells = " & ".join("--" if pd.isna(v) else f"{v:.3f}"
                               for v in row)
            lines.append(f"{name} & {cells} \\\\")
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

    mn = mcnemar_freshness(df)
    mn.to_csv(res_dir / "mcnemar_freshness.csv", index=False)
    print(f"[out] mcnemar_freshness.csv ({len(mn)} rows)")

    f1_csvs(single, defs, res_dir)

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