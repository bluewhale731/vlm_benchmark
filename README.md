# VLM Food-Rescue Benchmark

Benchmarks 5 VLMs × 3 tasks × multiple prompt strategies on the ZeroWaste
pantry dataset, producing the evaluation tables for the journal manuscript.

## Grid

| Task | Strategies |
|---|---|
| `donation_type` (packaged/produce/bakery) | freeform_neutral, freeform_definitions, logit_multichoice (A/B/C letter likelihood) |
| `freshness` (fresh/edible_soon/spoiled) | freeform_neutral, freeform_inspector (rejection persona), freeform_cot, **logit_cascade** (Eq. 1 of the paper) |
| `defects` (6 classes, multi-label) | freeform_list, logit_per_defect (one Yes/No likelihood per defect) |

Models (config.yaml): LLaVA-1.5 7B, Qwen2-VL 7B, Qwen2.5-VL 7B,
InternVL3-8B, Llama-3.2-11B-Vision (gated — `huggingface-cli login` first,
or delete it). Optional GPT-4o entry (generation strategies only) — uncomment
in config and set `OPENAI_API_KEY`.

## Setup

```bash
conda create -n vlm_bench python=3.11 -y && conda activate vlm_bench
pip install "transformers>=4.52" accelerate bitsandbytes torch torchvision \
            pillow pyyaml pandas numpy scipy scikit-learn qwen-vl-utils openai
```

Point the `datasets:` block of `config.yaml` at your three dataset roots:

```yaml
datasets:
  donation_type: {format: yolo,        root: /work/.../donation_type}
  defects:       {format: yolo,        root: /work/.../defect}
  freshness:     {format: imagefolder, root: /work/.../produce}
```

- **yolo** roots need the Ultralytics `data.yaml` (class names are read
  from its `names:` block by *name*, never by index position) plus the
  usual `images/` + `labels/` txt files; both `images/{split}/` and
  `{split}/images/` layouts are handled. For defects, images with a
  missing or empty label file count as negatives for every defect class.
- **imagefolder** roots use class-named directories (`fresh/`,
  `edible_soon/`, `spoiled/`), optionally nested under `train/val/test`.
- Per task, `splits: [test]` restricts to one split (e.g. the ResNet
  held-out set for the head-to-head table); `null` uses everything.

Sanity-check data loading before burning GPU hours:

```bash
python -m src.data_loading config.yaml
```

Expected: donation_type 412, freshness 174, defects ≈66 labeled
(edible_soon + spoiled) images with the class counts from Tables 1–3.
Fix any `[warn] ... classes not in the canonical set` messages by adding
the exact spelling from your data.yaml to `_CANON` in
`src/data_loading.py`.

## Run

Smoke test (5 images per cell, one model):

```bash
python -m src.run_benchmark config.yaml --model llava15_7b --limit 5
```

Full grid on Nova:

```bash
sbatch slurm/run_all.sbatch
```

Runs are resumable — re-submitting skips completed images. Raw per-image
records (generations, logit scores, latencies) land in `results/raw/*.jsonl`.

To evaluate on the ResNet held-out split for the head-to-head table, either
set `splits: [test]` on the freshness dataset (if your ImageFolder tree has
split directories) or set `paths.subset_manifest` to a file of test-split
filenames — then re-run into a different `output_dir`.

## ResNet-18 classification baselines

The VLMs answer image-level classification questions, so the fair CNN
comparison is an image-level classifier, not the YOLO detectors.
`src/train_resnet.py` trains ResNet-18 on any task using the same
image-level ground truth as the VLM benchmark (labels come through
`src.data_loading.load_task`, so the head-to-head tables share identical
labels): donation_type and freshness are single-label cross-entropy;
defects is multi-label BCE with per-class pos_weight and greedy iterative
stratification so every scarce defect appears in each split.

```bash
# check the stratified 70/15/15 split before training (no torch needed)
python -m src.train_resnet config.yaml --task donation_type --dry-run

# train (Nova)
sbatch slurm/train_resnet.sbatch donation_type
sbatch slurm/train_resnet.sbatch defects
```

Protocol matches the locked freshness run: ResNet-18 ImageNet weights,
224x224, Adam 1e-4, seed 42, best checkpoint by val macro-F1, metrics on
the held-out test split. Outputs land in `results/resnet_{task}/`:
`best.pt`, `training_history.json`, `test_predictions.json`,
`test_metrics.json`, an analyzer-style summary CSV, and
`test_manifest.txt`.

For the head-to-head table, uncomment the task's `subset_manifest:` line
in `config.yaml` (pointing at that `test_manifest.txt`) and re-run the
VLM benchmark into a fresh `output_dir` — the VLMs are then scored on
exactly the ResNet held-out images. `subset_manifest` under a dataset
entry overrides the global `paths.subset_manifest` for that task only.

## Analyze

```bash
python -m src.analyze_results config.yaml
```

Produces in `results/`:
- `summary_single_label.csv` — accuracy, macro-F1, per-class P/R/F1,
  spoiled recall/precision, parse-failure rates, confusion matrices
- `summary_defects.csv` — per-defect sensitivity/specificity per model
- `mcnemar_freshness.csv` — exact McNemar tests between every strategy pair
  within each model
- `tau_sweep.csv` — cascade performance over the full (τ_spoil, τ_degrad)
  grid, re-derived from stored scores (no re-inference needed); prints the
  best operating point per model
- `tables.tex` — drop-in LaTeX tables for the manuscript

## Notes for the paper

- `parse_fail_rate` per model×strategy is itself reportable: free-form JSON
  compliance varies sharply across VLMs.
- The τ-sweep gives you the "adjustable safety threshold" figure (spoiled
  recall vs. accuracy trade-off curve) claimed in the Discussion.
- Every prediction is deterministic: greedy decoding, seed 42, exact
  first-token likelihoods — matches the protocol paragraph in Methods.
- The cascade questions in `src/prompts.py` are verbatim the ones in the
  manuscript; if you edit one, edit both.
