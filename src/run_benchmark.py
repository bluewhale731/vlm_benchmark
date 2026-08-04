"""Run the full VLM benchmark grid: models x tasks x prompt strategies.

Every (model, task, strategy) cell writes one JSONL file under
results/raw/, one record per image, containing raw generations and/or
logit scores. Runs are resumable: already-processed images are skipped.

Usage (single model, useful for SLURM arrays):
    python -m src.run_benchmark config.yaml --model llava15_7b
Usage (everything, sequentially):
    python -m src.run_benchmark config.yaml
Options:
    --tasks freshness defects     restrict tasks
    --limit 10                    smoke-test on first N images
"""

from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

from .data_loading import load_task
from .parsers import PARSERS
from .prompts import STRATEGIES
from .vlm_wrappers import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    for line in path.open():
        try:
            ids.add(json.loads(line)["file_name"])
        except (json.JSONDecodeError, KeyError):
            continue
    return ids


def run_cell(model, model_name, task, strategy_name, strat, examples,
             cfg, out_dir: Path, limit=None):
    kind = strat["kind"]
    if kind.startswith("logit") and not model.supports_logits:
        print(f"  [skip] {strategy_name}: backend has no logit access")
        return

    out_path = out_dir / f"{model_name}__{task}__{strategy_name}.jsonl"
    done = _done_ids(out_path)
    todo = [e for e in examples if e.file_name not in done]
    if limit:
        todo = todo[:max(0, limit - len(done))]
    if not todo:
        print(f"  [done] {strategy_name} ({len(done)} records)")
        return
    print(f"  [run ] {strategy_name}: {len(todo)} images "
          f"({len(done)} already done)")

    rt = cfg["runtime"]

    with out_path.open("a") as f:
        for i, ex in enumerate(todo):
            img = ex.path
            rec = {"model": model_name, "task": task,
                   "strategy": strategy_name, "file_name": ex.file_name,
                   "gt": sorted(ex.labels) if task == "defects" else ex.label}
            t0 = time.time()
            try:
                if kind == "freeform":
                    text = model.generate(img, strat["prompt"])
                    parsed, ok = PARSERS[strat["parse"]](text)
                    rec["raw_text"] = text
                    rec["parse_ok"] = ok
                    rec["pred"] = (sorted(parsed) if isinstance(parsed, set)
                                   else parsed)

                elif kind == "logit_binary_cascade":
                    s_spoil = model.yes_no_score(img, strat["q_spoil"])
                    s_degrad = model.yes_no_score(img, strat["q_degrad"])
                    rec["s_spoil"] = s_spoil
                    rec["s_degrad"] = s_degrad
                    if s_spoil >= rt["tau_spoil"]:
                        rec["pred"] = "spoiled"
                    elif s_degrad >= rt["tau_degrad"]:
                        rec["pred"] = "edible_soon"
                    else:
                        rec["pred"] = "fresh"

                elif kind == "logit_multichoice":
                    letters = list(strat["options"])
                    scores = model.option_scores(img, strat["prompt"],
                                                 letters)
                    rec["option_scores"] = scores
                    best = max(scores, key=scores.get)
                    rec["pred"] = strat["options"][best]

                elif kind == "logit_multilabel":
                    scores = {c: model.yes_no_score(img, q)
                              for c, q in strat["questions"].items()}
                    rec["defect_scores"] = scores
                    rec["pred"] = sorted(c for c, s in scores.items()
                                         if s >= 0.5)
                else:
                    raise ValueError(f"Unknown strategy kind: {kind}")
            except Exception:  # noqa: BLE001
                rec["error"] = traceback.format_exc(limit=3)
                rec["pred"] = None
            rec["latency_s"] = round(time.time() - t0, 3)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"      {i + 1}/{len(todo)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--model", default=None,
                    help="run a single model (config key)")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: only first N images per cell")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg["runtime"].get("seed", 42))
    out_dir = Path(cfg["paths"]["output_dir"]) / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_names = [args.model] if args.model else list(cfg["models"])
    task_names = args.tasks or list(cfg["tasks"])

    # Load all task data once up front (fails fast on annotation problems)
    data = {t: load_task(t, cfg) for t in task_names}
    for t, ex in data.items():
        print(f"[data] {t}: {len(ex)} images")

    for mname in model_names:
        mcfg = cfg["models"][mname]
        print(f"\n=== Loading model: {mname} ({mcfg.get('hf_id') or mcfg.get('api_model')}) ===")
        model = build_model(mname, mcfg, cfg["runtime"])
        try:
            for task in task_names:
                print(f"[task] {task}")
                for sname in cfg["tasks"][task]["strategies"]:
                    strat = STRATEGIES[task][sname]
                    run_cell(model, mname, task, sname, strat, data[task],
                             cfg, out_dir, limit=args.limit)
        finally:
            model.unload()


if __name__ == "__main__":
    main()
