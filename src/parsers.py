"""Parsers mapping free-form VLM generations to task labels.

Every parser returns (label_or_labels, parse_ok). On failure we return a
deterministic fallback and parse_ok=False so parse-failure rates can be
reported per model x strategy (a result in itself for the paper).
"""

from __future__ import annotations

import json
import re

FRESHNESS_CLASSES = ["fresh", "edible_soon", "spoiled"]
DONATION_CLASSES = ["packaged", "produce", "bakery"]
DEFECT_CLASSES = ["wrinkling", "visible_cut", "bruising",
                  "discoloration", "leaking", "mold"]

_DEFECT_SYNONYMS = {
    "wrinkling": ["wrinkl", "shrivel"],
    "visible_cut": ["cut", "slice", "gouge", "rupture", "gash"],
    "bruising": ["bruis"],
    "discoloration": ["discolor", "discolour", "brown spot", "brown patch",
                      "black spot", "dark spot", "dark patch"],
    "leaking": ["leak", "liquid pool", "oozing"],
    "mold": ["mold", "mould", "fuzz", "fungal", "fungus"],
}


def _extract_json(text: str) -> dict | None:
    """Pull the last JSON object out of a generation (handles ```json fences
    and CoT preambles)."""
    text = re.sub(r"```(?:json)?", "", text)
    candidates = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            try:
                return json.loads(cand.replace("'", '"'))
            except json.JSONDecodeError:
                continue
    return None


def _keyword_vote(text: str, classes: list[str],
                  aliases: dict[str, list[str]]) -> str | None:
    """Last occurrence wins (final verdict usually comes last in CoT)."""
    text_l = text.lower()
    best, best_pos = None, -1
    for cls in classes:
        for kw in aliases.get(cls, [cls]):
            pos = text_l.rfind(kw)
            if pos > best_pos:
                best, best_pos = cls, pos
    return best


def parse_freshness(text: str) -> tuple[str, bool]:
    obj = _extract_json(text)
    if obj:
        for key in ("condition", "freshness", "status", "label", "category"):
            v = str(obj.get(key, "")).strip().lower().replace(" ", "_")
            v = v.replace("-", "_")
            if v in FRESHNESS_CLASSES:
                return v, True
            if "edible" in v:
                return "edible_soon", True
            if v in ("rotten", "moldy", "bad"):
                return "spoiled", True
    kw = _keyword_vote(text, FRESHNESS_CLASSES, {
        "fresh": ["fresh"],
        "edible_soon": ["edible_soon", "edible soon", "edible-soon"],
        "spoiled": ["spoiled", "spoilt", "rotten"],
    })
    if kw:
        return kw, True
    return "fresh", False  # fallback = majority class; flagged as parse fail


def parse_donation(text: str) -> tuple[str, bool]:
    obj = _extract_json(text)
    if obj:
        for key in ("category", "type", "donation_type", "label"):
            v = str(obj.get(key, "")).strip().lower()
            if v in DONATION_CLASSES:
                return v, True
    kw = _keyword_vote(text, DONATION_CLASSES, {
        "packaged": ["packaged", "package", "canned", "boxed"],
        "produce": ["produce", "fruit", "vegetable"],
        "bakery": ["bakery", "bread", "baked"],
    })
    if kw:
        return kw, True
    return "packaged", False


def parse_defect_list(text: str) -> tuple[set[str], bool]:
    obj = _extract_json(text)
    found: set[str] = set()
    ok = False
    if obj and isinstance(obj.get("defects"), list):
        ok = True
        for item in obj["defects"]:
            item_l = str(item).lower()
            for cls, kws in _DEFECT_SYNONYMS.items():
                if cls in item_l or any(k in item_l for k in kws):
                    found.add(cls)
    if not ok:
        # keyword scan over the raw text as fallback
        text_l = text.lower()
        for cls, kws in _DEFECT_SYNONYMS.items():
            if any(k in text_l for k in kws):
                found.add(cls)
        ok = bool(found) or ("no defect" in text_l or '"defects": []' in text)
    return found, ok


PARSERS = {
    "parse_freshness": parse_freshness,
    "parse_donation": parse_donation,
    "parse_defect_list": parse_defect_list,
}
