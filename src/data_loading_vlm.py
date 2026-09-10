"""Build unified evaluation manifests from per-task dataset directories.

Supported formats (set per task in config.yaml):

  yolo         Ultralytics detection layout. Class names come from the
               dataset's data.yaml (`names:` list or dict) — never from
               hard-coded indices. Image-level labels are derived from the
               class IDs in each image's labels/*.txt file:
                 - single-label tasks (donation_type): majority class over
                   boxes, ties broken toward the later class in the task's
                   canonical order (safety-first for freshness-like tasks)
                 - multi-label tasks (defects): the set of box classes;
                   images with a missing/empty label file are negatives
                   for every class
               Handles both `root/images/{split}/` + `root/labels/{split}/`
               and `root/{split}/images/` + `root/{split}/labels/` layouts,
               as well as flat `root/images/` + `root/labels/`.

  imagefolder  ResNet/ImageFolder layout: class-named directories contain
               the images, optionally nested under split directories, e.g.
               `root/fresh/*.jpg` or `root/train/edible_soon/*.jpg`.
               The label is the nearest ancestor directory whose name
               normalizes to a known class.

Optional per-task `splits: [test]` restricts loading to paths containing
that directory component (e.g. only the held-out split). The global
`subset_manifest` (one image filename per line) further restricts by
basename and applies to all tasks.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DONATION_CLASSES = ["packaged", "produce", "bakery"]
FRESHNESS_CLASSES = ["fresh", "edible_soon", "spoiled"]  # order = severity
DEFECT_CLASSES = ["wrinkling", "visible_cut", "bruising",
                  "discoloration", "leaking", "mold"]

TASK_CLASSES = {
    "donation_type": DONATION_CLASSES,
    "freshness": FRESHNESS_CLASSES,
    "defects": DEFECT_CLASSES,
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_CANON = {
    # donation type
    "packaged": "packaged", "package": "packaged", "packaged_item": "packaged",
    "packaged_items": "packaged",
    "produce": "produce", "bakery": "bakery", "baked": "bakery",
    "baked_goods": "bakery",
    # freshness
    "fresh": "fresh",
    "edible_soon": "edible_soon", "edible-soon": "edible_soon",
    "ediblesoon": "edible_soon", "edible": "edible_soon",
    "spoiled": "spoiled", "rotten": "spoiled", "spoilt": "spoiled",
    # defects
    "wrinkling": "wrinkling", "wrinkled": "wrinkling", "wrinkle": "wrinkling",
    "wrinkles": "wrinkling",
    "visible_cut": "visible_cut", "cut": "visible_cut", "cuts": "visible_cut",
    "visible-cut": "visible_cut", "visible_cuts": "visible_cut",
    "bruising": "bruising", "bruise": "bruising", "bruised": "bruising",
    "bruises": "bruising",
    "discoloration": "discoloration", "discolored": "discoloration",
    "discolouration": "discoloration", "discoloured": "discoloration",
    "leaking": "leaking", "leak": "leaking", "leakage": "leaking",
    "mold": "mold", "mould": "mold", "moldy": "mold", "mouldy": "mold",
}


def _norm(name: str) -> str:
    name = unicodedata.normalize("NFKC", str(name)).strip().lower()
    name = re.sub(r"[\s\-]+", "_", name)
    return _CANON.get(name, name)


@dataclass
class Example:
    path: Path                       # absolute image path
    file_name: str                   # path relative to dataset root (unique)
    label: str | None = None         # single-label tasks
    labels: set[str] = field(default_factory=set)  # multi-label (defects)


# ----------------------------------------------------------------
# YOLO format
# ----------------------------------------------------------------
def _load_yolo_names(root: Path, data_yaml: Path | None) -> dict[int, str]:
    """Read class names from an explicit data_yaml, any *.yaml with
    `names:` in root or its parent, or a CVAT-style obj.names file."""
    candidates = []
    if data_yaml:
        candidates.append(Path(data_yaml).expanduser().resolve())
    for d in (root, root.parent):
        candidates += sorted(d.glob("*.yaml")) + sorted(d.glob("*.yml"))
    for p in candidates:
        if not p.exists():
            continue
        try:
            y = yaml.safe_load(p.read_text())
        except yaml.YAMLError:
            continue
        names = (y or {}).get("names") if isinstance(y, dict) else None
        if names is None:
            continue
        if isinstance(names, dict):
            return {int(k): _norm(v) for k, v in names.items()}
        if isinstance(names, list):
            return {i: _norm(v) for i, v in enumerate(names)}
    # CVAT YOLO 1.1 export: obj.names, one class name per line
    for d in (root, root.parent):
        p = d / "obj.names"
        if p.exists():
            lines = [l.strip() for l in p.read_text().splitlines()
                     if l.strip()]
            return {i: _norm(v) for i, v in enumerate(lines)}
    raise FileNotFoundError(
        f"No class-name source found for {root}: looked for a YAML with a "
        f"`names:` block (or obj.names) in the dataset root and its parent. "
        f"Set `data_yaml:` for this dataset in config.yaml to point at it "
        f"explicitly.")


def _yolo_label_path(img: Path) -> Path | None:
    """Map an image path to its label .txt across common layouts:
    Ultralytics (images/ -> labels/), sibling labels dir, or CVAT-style
    label .txt next to the image in the same directory."""
    parts = list(img.parts)
    if "images" in parts:
        i = len(parts) - 1 - parts[::-1].index("images")  # last occurrence
        parts[i] = "labels"
        cand = Path(*parts).with_suffix(".txt")
        if cand.exists():
            return cand
    cand = img.parent.parent / "labels" / (img.stem + ".txt")
    if cand.exists():
        return cand
    cand = img.with_suffix(".txt")  # CVAT: label lives beside the image
    if cand.exists():
        return cand
    return None


_SKIP_DIRS = {"labels", "runs", ".ipynb_checkpoints", "__pycache__",
              "weights", "plots"}


def _find_images(root: Path, splits: list[str] | None) -> list[Path]:
    imgs = [p for p in root.rglob("*")
            if p.suffix.lower() in IMG_EXTS
            and not (_SKIP_DIRS & set(p.parts))
            and not any(part.startswith(".") for part in p.parts)]
    if splits:
        split_set = set(splits)
        imgs = [p for p in imgs if split_set & set(p.parts)]
    return sorted(imgs)


def _read_yolo_classes(label_path: Path | None,
                       names: dict[int, str]) -> list[str]:
    if not label_path or not label_path.exists():
        return []
    out = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            cid = int(float(line.split()[0]))
        except (ValueError, IndexError):
            continue
        cname = names.get(cid)
        if cname:
            out.append(cname)
    return out


def load_yolo_task(root: Path, task: str, splits: list[str] | None,
                   data_yaml=None, exclude_fresh=False) -> list[Example]:
    names = _load_yolo_names(root, data_yaml)
    valid = TASK_CLASSES[task]
    all_known = {c for cls in TASK_CLASSES.values() for c in cls}
    unknown = sorted(set(names.values()) - all_known)
    if unknown:
        print(f"[warn] {root.name}: data.yaml classes unknown to every "
              f"task and ignored: {unknown}")

    multi = task == "defects"
    severity = {c: i for i, c in enumerate(valid)}
    out = []
    n_fresh_excluded = 0
    for img in _find_images(root, splits):
        all_classes = _read_yolo_classes(_yolo_label_path(img), names)
        classes = [c for c in all_classes if c in valid]
        rel = str(img.relative_to(root))
        if multi:
            if exclude_fresh:
                fr = [c for c in all_classes if c in FRESHNESS_CLASSES]
                # drop images whose freshness annotation is fresh-only
                # (no annotation at all -> keep; can't determine condition)
                if fr and all(c == "fresh" for c in fr):
                    n_fresh_excluded += 1
                    continue
            out.append(Example(img, rel, labels=set(classes)))
        else:
            if not classes:
                continue  # unlabeled image in a single-label task: skip
            counts = Counter(classes).most_common()
            top_n = counts[0][1]
            tied = [c for c, n in counts if n == top_n]
            label = max(tied, key=lambda c: severity[c])
            out.append(Example(img, rel, label=label))
    if n_fresh_excluded:
        print(f"[info] {task}: excluded {n_fresh_excluded} fresh-only "
              f"images (exclude_fresh=true)")
    return out


# ----------------------------------------------------------------
# COCO format (CVAT export): labels from JSON, images from a directory
# ----------------------------------------------------------------
def load_coco_task(root: Path, task: str, coco_json,
                   splits: list[str] | None) -> list[Example]:
    import json as _json
    coco_path = Path(coco_json).expanduser().resolve()
    if not coco_path.exists():
        raise FileNotFoundError(f"COCO json not found: {coco_path}")
    coco = _json.loads(coco_path.read_text())
    cats = {c["id"]: _norm(c["name"]) for c in coco.get("categories", [])}
    img_names = {im["id"]: Path(im["file_name"]).name
                 for im in coco.get("images", [])}
    valid = TASK_CLASSES[task]
    multi = task == "defects"
    severity = {c: i for i, c in enumerate(valid)}

    votes: dict[int, Counter] = {}
    for a in coco.get("annotations", []):
        cname = cats.get(a["category_id"])
        if cname in valid:
            votes.setdefault(a["image_id"], Counter())[cname] += 1

    # index actual image files under root by basename
    on_disk = {p.name: p for p in _find_images(root, splits)}
    out, missing = [], 0
    ids = img_names if multi else votes  # defects: unannotated = negatives
    for img_id in ids:
        base = img_names.get(img_id)
        path = on_disk.get(base) if base else None
        if path is None:
            missing += 1
            continue
        rel = str(path.relative_to(root))
        if multi:
            out.append(Example(path, rel,
                               labels=set(votes.get(img_id, Counter()))))
        else:
            counts = votes[img_id].most_common()
            tied = [c for c, n in counts if n == counts[0][1]]
            out.append(Example(path, rel,
                               label=max(tied, key=lambda c: severity[c])))
    if missing:
        print(f"[warn] {task}: {missing} images referenced in "
              f"{coco_path.name} not found under {root}")
    return out


# ----------------------------------------------------------------
# ImageFolder (ResNet) format
# ----------------------------------------------------------------
def load_imagefolder_task(root: Path, task: str,
                          splits: list[str] | None) -> list[Example]:
    valid = set(TASK_CLASSES[task])
    out = []
    for img in _find_images(root, splits):
        label = None
        for ancestor in img.parents:
            if ancestor == root.parent:
                break
            cand = _norm(ancestor.name)
            if cand in valid:
                label = cand
                break
        if label is None:
            continue
        out.append(Example(img, str(img.relative_to(root)), label=label))
    if not out:
        raise RuntimeError(
            f"No class-labeled images found under {root}. Expected "
            f"directories named after classes ({sorted(valid)}), optionally "
            f"nested under split directories.")
    return out


# ----------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------
def apply_subset(examples: list[Example], manifest_path) -> list[Example]:
    if not manifest_path:
        return examples
    keep = {Path(l.strip()).name for l in open(manifest_path) if l.strip()}
    return [e for e in examples if e.path.name in keep]


def load_task(task: str, cfg: dict) -> list[Example]:
    dcfg = cfg["datasets"][task]
    root = Path(dcfg["root"]).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root for '{task}' not found: "
                                f"{root}")
    fmt = dcfg["format"]
    splits = dcfg.get("splits")
    if fmt == "yolo":
        ex = load_yolo_task(root, task, splits, dcfg.get("data_yaml"),
                            exclude_fresh=dcfg.get("exclude_fresh", False))
    elif fmt == "coco":
        ex = load_coco_task(root, task, dcfg.get("coco_json"), splits)
    elif fmt == "imagefolder":
        ex = load_imagefolder_task(root, task, splits)
    else:
        raise ValueError(f"Unknown dataset format '{fmt}' for task '{task}'")
    ex = apply_subset(ex, cfg["paths"].get("subset_manifest"))
    if not ex:
        raise RuntimeError(f"No examples loaded for task '{task}' — check "
                           f"root, format, splits, and class names.")
    ex.sort(key=lambda e: e.file_name)
    return ex


if __name__ == "__main__":
    # sanity check: python -m src.data_loading config.yaml
    import sys
    cfg = yaml.safe_load(open(sys.argv[1]))
    for t in cfg.get("tasks", {}):
        try:
            ex = load_task(t, cfg)
            if t == "defects":
                pos = Counter(c for e in ex for c in e.labels)
                n_neg = sum(1 for e in ex if not e.labels)
                print(f"{t}: {len(ex)} images | defect counts: "
                      f"{dict(pos)} | defect-free images: {n_neg}")
            else:
                print(f"{t}: {len(ex)} images | "
                      f"{Counter(e.label for e in ex)}")
        except Exception as e:  # noqa: BLE001
            print(f"{t}: FAILED -> {e}")