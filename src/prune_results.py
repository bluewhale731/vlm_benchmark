"""Prune existing defect result files to the current defect dataset.

After changing the defect dataset definition (e.g. exclude_fresh: true),
already-completed result JSONLs still contain records for images that are
no longer in the evaluation set. This removes those records in place so
analyze_results reflects the current dataset — no re-inference needed.
Backups are written alongside as *.jsonl.bak.

Usage:
    python -m src.prune_results config.yaml
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import yaml

from .data_loading import load_task


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    keep = {e.file_name for e in load_task("defects", cfg)}
    # file_name in records is relative to the dataset root; also accept
    # basename matches in case roots shifted between runs
    keep_base = {Path(f).name for f in keep}

    raw = Path(cfg["paths"]["output_dir"]) / "raw"
    for p in sorted(raw.glob("*__defects__*.jsonl")):
        lines = p.read_text().splitlines()
        kept, dropped = [], 0
        for line in lines:
            try:
                fn = json.loads(line)["file_name"]
            except (json.JSONDecodeError, KeyError):
                continue
            if fn in keep or Path(fn).name in keep_base:
                kept.append(line)
            else:
                dropped += 1
        if dropped:
            shutil.copy(p, p.with_suffix(".jsonl.bak"))
            p.write_text("\n".join(kept) + ("\n" if kept else ""))
        print(f"{p.name}: kept {len(kept)}, dropped {dropped}"
              + (" (backup written)" if dropped else ""))


if __name__ == "__main__":
    main()
