"""Prompt registry for the three tasks.

Two strategy families:
  * "freeform"  : model generates text; we parse a label out of it.
                  Works on every backend (HF + API).
  * "logit"     : we score P("Yes") vs P("No") (or option letters) from
                  first-token logits. Open-weight HF backends only.

Each strategy dict:
  kind      : "freeform" | "logit_binary_cascade" | "logit_multichoice"
              | "logit_multilabel"
  prompt(s) : text used
  parse     : (freeform only) name of parser in parsers.py
"""

FRESHNESS_CLASSES = ["fresh", "edible_soon", "spoiled"]
DONATION_CLASSES = ["packaged", "produce", "bakery"]
DEFECT_CLASSES = ["wrinkling", "visible_cut", "bruising",
                  "discoloration", "leaking", "mold"]

# ---------------------------------------------------------------
# Task 1: donation type
# ---------------------------------------------------------------
_DONATION_NEUTRAL = (
    "Look at this food donation item. Classify it into exactly one category: "
    "'packaged', 'produce', or 'bakery'. "
    'Respond ONLY with raw JSON: {"category": "packaged|produce|bakery"}'
)

_DONATION_DEFS = (
    "Classify this food pantry donation item into exactly one category.\n"
    "- 'packaged': shelf-stable manufactured goods in cans, boxes, jars, "
    "bags, or bottles (e.g., canned beans, cereal, pasta, condiments).\n"
    "- 'produce': fresh fruits or vegetables, loose or in produce bags.\n"
    "- 'bakery': bread, rolls, pastries, cakes, or other baked goods, "
    "including bakery items in clamshells or bakery bags.\n"
    'Respond ONLY with raw JSON: {"category": "packaged|produce|bakery"}'
)

_DONATION_MC = (
    "What type of food pantry donation is shown in this image?\n"
    "A. Packaged shelf-stable item (canned, boxed, bagged, or bottled)\n"
    "B. Fresh produce (fruit or vegetable)\n"
    "C. Bakery item (bread or baked goods)\n"
    "Answer with a single letter (A, B, or C). Answer:"
)

# ---------------------------------------------------------------
# Task 2: produce freshness
# ---------------------------------------------------------------
_FRESH_NEUTRAL = (
    "Look at this produce item and assess its condition. Classify it as "
    "'fresh' (good condition, suitable for distribution), 'edible_soon' "
    "(showing aging, should be consumed quickly), or 'spoiled' (unsafe, "
    "must be discarded). "
    'Respond ONLY with raw JSON: {"condition": "fresh|edible_soon|spoiled"}'
)

# Aggressive rejection-oriented persona — verbatim structure from the paper.
_FRESH_INSPECTOR = (
    "ACT AS FOOD SAFETY INSPECTOR. Your goal is to find reasons to REJECT "
    "this item.\n"
    "STEP 1: ANALYZE IMAGES FOR DEFECTS\n"
    "- Do you see any white/green fuzz or grey furry patches? (Mold)\n"
    "- Is there any liquid pooling or wet stickiness? (Leaking)\n"
    "- Are there deep black pits, skin ruptures, or collapsed soft areas? "
    "(Rot)\n"
    "STEP 2: CLASSIFY BASED ON HIERARCHY\n"
    "1. If ANY Step 1 defects are found -> ALWAYS 'spoiled'.\n"
    "2. If skin is dry but has wrinkles, brown bruises, discoloration, or "
    "shriveling -> 'edible_soon'.\n"
    "3. If skin is perfectly taut, shiny, and vibrant -> 'fresh'.\n"
    'Output ONLY raw JSON: {"condition": "fresh|edible_soon|spoiled"}'
)

_FRESH_COT = (
    "You are inspecting donated produce for a food pantry. Think step by "
    "step:\n"
    "1. Identify the item.\n"
    "2. Describe the skin/surface texture, color uniformity, and any "
    "visible damage.\n"
    "3. Decide: 'fresh' (no visible degradation), 'edible_soon' (visible "
    "aging such as wrinkles, bruises, or minor discoloration), or "
    "'spoiled' (mold, rot, leakage, or collapse).\n"
    "After your reasoning, end with one line of raw JSON exactly like: "
    '{"condition": "fresh|edible_soon|spoiled"}'
)

# Cascade questions — verbatim from Sec. "Decomposed Binary Logit-Likelihood
# Scoring Cascade" of the manuscript.
Q_SPOIL = ("Does this produce item exhibit rot, mold, liquid leakage, or "
           "severe surface collapse? Answer Yes or No. Answer:")
Q_DEGRAD = ("Does this produce item show signs of aging such as skin "
            "wrinkling, surface bruising, cuts, or minor discoloration? "
            "Answer Yes or No. Answer:")

# ---------------------------------------------------------------
# Task 3: fine-grained defects
# ---------------------------------------------------------------
_DEFECT_LIST = (
    "Inspect this produce item closely for physical defects. Consider each "
    "of the following defect types: wrinkling, visible_cut, bruising, "
    "discoloration, leaking, mold.\n"
    "Respond ONLY with raw JSON listing every defect you can see (empty "
    'list if none): {"defects": ["..."]}'
)

_DEFECT_QUESTIONS = {
    "wrinkling": ("Does the skin or surface of this produce item show "
                  "wrinkling, shriveling, or loss of firmness? "
                  "Answer Yes or No. Answer:"),
    "visible_cut": ("Does this produce item have a visible cut, slice, "
                    "gouge, or skin rupture? Answer Yes or No. Answer:"),
    "bruising": ("Does this produce item show bruising, such as soft "
                 "darkened areas beneath the skin? Answer Yes or No. "
                 "Answer:"),
    "discoloration": ("Does this produce item show abnormal discoloration, "
                      "such as brown, black, or gray patches that are not "
                      "part of its normal coloring? Answer Yes or No. "
                      "Answer:"),
    "leaking": ("Is this produce item leaking liquid, or is there liquid "
                "pooling around it? Answer Yes or No. Answer:"),
    "mold": ("Does this produce item have any mold, such as white, green, "
             "gray, or black fuzzy growth? Answer Yes or No. Answer:"),
}

# ---------------------------------------------------------------
# Registry
# ---------------------------------------------------------------
STRATEGIES = {
    "donation_type": {
        "freeform_neutral": dict(kind="freeform", prompt=_DONATION_NEUTRAL,
                                 parse="parse_donation"),
        "freeform_definitions": dict(kind="freeform", prompt=_DONATION_DEFS,
                                     parse="parse_donation"),
        "logit_multichoice": dict(kind="logit_multichoice",
                                  prompt=_DONATION_MC,
                                  options={"A": "packaged", "B": "produce",
                                           "C": "bakery"}),
    },
    "freshness": {
        "freeform_neutral": dict(kind="freeform", prompt=_FRESH_NEUTRAL,
                                 parse="parse_freshness"),
        "freeform_inspector": dict(kind="freeform", prompt=_FRESH_INSPECTOR,
                                   parse="parse_freshness"),
        "freeform_cot": dict(kind="freeform", prompt=_FRESH_COT,
                             parse="parse_freshness"),
        "logit_cascade": dict(kind="logit_binary_cascade",
                              q_spoil=Q_SPOIL, q_degrad=Q_DEGRAD),
    },
    "defects": {
        "freeform_list": dict(kind="freeform", prompt=_DEFECT_LIST,
                              parse="parse_defect_list"),
        "logit_per_defect": dict(kind="logit_multilabel",
                                 questions=_DEFECT_QUESTIONS),
    },
}
