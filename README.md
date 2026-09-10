# VLM Food-Rescue Benchmark (ZeroWaste)

Code and results for *Evaluating Vision–Language Models for Automated
Quality Assessment of Food Donations* (Singh & Ayanlade). Zero-shot benchmark of 5 open-weight vision-language models on three
image-level tasks from the ZeroWaste food-pantry dataset. Every VLM
prediction is scored against human ground-truth annotations; the repo
produces the evaluation tables in the manuscript.

## Tasks and prompt strategies

| Task | Classes | Strategies |
|---|---|---|
| `donation_type` | packaged / produce / bakery | freeform_neutral, logit_multichoice (A/B/C letter likelihood) |
| `freshness` | fresh / edible_soon / spoiled | freeform_neutral, freeform_inspector, logit_cascade (Eq. 1 of the paper) |
| `defects` (multi-label) | bruising, discoloration, mold, visible_cut, wrinkling (leaking is also scored but has no positive images and is omitted from the paper) | freeform_list, logit_per_defect (one Yes/No likelihood per defect) |

Models (`config.yaml`): LLaVA-1.5 7B, Qwen2-VL 7B, Qwen2.5-VL 7B,
InternVL3 8B, Llama-3.2 11B-Vision (gated — `huggingface-cli login` first).

## Ground-truth annotations

The images were annotated in CVAT and exported in two formats. The
benchmark reads these files only to obtain the image-level label for each
image; no model is trained on them.

| Folder | Task | Format | How the label is derived |
|---|---|---|---|
| `donation_type/` | donation_type | YOLO (`data.yaml`, `images/`, `labels/*.txt`) | class name of the annotated box(es); class names are matched by *name* from `data.yaml`, never by index |
| `defect/` | defects | YOLO | set of defect classes present in the label file; an image with a missing or empty label file is a negative for every defect |
| `produce/` | freshness | COCO JSON (`instances_default_produce.json`) | category of the annotated instance |

The `datasets:` block of `config.yaml` points at these three roots. The
`format:` key selects the loader in `src/data_loading.py` and is required.

```yaml
datasets:
  donation_type: {format: yolo, root: donation_type/obj_train_data}
  defects:       {format: yolo, root: defect/obj_train_data}
  freshness:     {format: coco, root: produce/custom_produce,
                  coco_json: produce/instances_default_produce.json}
```

Sanity-check loading before using GPU time:

```bash
python -m src.data_loading config.yaml
```

Expected counts: donation_type 412 (two of the 414 captured images lack
usable annotations), freshness 174, defects 52 images with at least one
annotated defect (54 annotations; two images carry two labels). These
match Table 1 of the paper. Fix any `[warn] ... classes not in the canonical set` messages by
adding the exact spelling from your annotation file to `_CANON` in
`src/data_loading.py`.

## Setup

```bash
conda create -n vlm_bench python=3.11 -y && conda activate vlm_bench
pip install "transformers>=4.52" accelerate bitsandbytes torch torchvision \
            pillow pyyaml pandas numpy scipy scikit-learn qwen-vl-utils
```

## Run

Smoke test (5 images per cell, one model):

```bash
python -m src.run_benchmark config.yaml --model llava15_7b --limit 5
```

Full grid on Nova (SLURM):

```bash
sbatch slurm/run_all.sbatch
```

Runs are resumable — re-submitting skips completed images. Raw per-image
records (generations, Yes/No likelihoods, latencies) land in
`results/raw/{model}__{task}__{strategy}.jsonl`.

## Analyze

```bash
python -m src.analyze_results config.yaml
```

## Results

Individual VLM performance and evaluation
Defect F1 at a fixed decision threshold (Table 5):

Each defect is predicted present when the normalized Yes/No probability
from Eq. 1 is ≥ 0.50. The threshold is the same for every model and
defect and is not tuned on the evaluation data. The CSV also reports the
in-sample balanced-accuracy-optimised F1 for comparison (not used in the
paper).

## Notes

- Every prediction is deterministic: greedy decoding, seed 42, exact
  first-token likelihoods.
- The cascade questions in `src/prompts.py` are verbatim the ones in the
  manuscript; if you edit one, edit both.
- `parse_fail_rate` per model×strategy is itself reportable: free-form
  JSON compliance varies sharply across VLMs.
