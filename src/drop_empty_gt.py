"""Remove defect result records whose ground truth is empty.

Rewrites every results/raw/*__defects__*.jsonl in place, keeping only
records where the image has at least one annotated defect (gt non-empty).
Backups are written alongside as *.jsonl.gtbak.

Usage:
    python -m src.drop_empty_gt config.yaml
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import yaml


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    raw = Path(cfg["paths"]["output_dir"]) / "raw"
    files = sorted(raw.glob("*__defects__*.jsonl"))
    if not files:
        sys.exit(f"No defect result files found under {raw}")

    for p in files:
        kept, dropped = [], 0
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("gt"):          # non-empty defect list -> keep
                kept.append(line)
            else:
                dropped += 1
        if dropped:
            shutil.copy(p, p.with_suffix(".jsonl.gtbak"))
            p.write_text("\n".join(kept) + ("\n" if kept else ""))
        print(f"{p.name}: kept {len(kept)}, dropped {dropped}"
              + (" (backup written)" if dropped else ""))


if __name__ == "__main__":
    main()
