# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

This is a fork of facebookresearch's MAE (Masked Autoencoder) adapted to **self-supervised pre-training on 3D cardiac MRI (CMR) LGE volumes**. The upstream ImageNet pipeline is still present (`main_pretrain.py`, `main_finetune.py`, `main_linprobe.py`, `README.md`, `PRETRAIN.md`, `FINETUNE.md`, `submitit_*.py`), but the active work is the CMR adaptation. When in doubt, the CMR-specific files are the ones to read; the original ImageNet docs (`README.md`, `PRETRAIN.md`, `FINETUNE.md`) describe the upstream behavior and are not kept in sync with the CMR changes.

## Environment

All Python runs use the conda env named **`mae`** (`source activate mae` before running anything). It pins `timm==0.3.2` (asserted at import in the main scripts) and provides torch, nibabel, scipy, matplotlib. There is no test suite, linter config, or build step — this is a research codebase driven by argparse scripts.

## Two pipelines, end to end

### 1. CMR self-supervised pre-training (the main pipeline)

- **Entry point:** `main_pretrain_cmr.py` (NOT `main_pretrain.py`, which is the ImageNet version).
- **Data flow:** raw NIfTI volumes → `preprocessing/extract_heart_roi_cmr.py` (crop to heart ROI) → `NiftiSliceDataset` (`util/datasets.py`) → `models_mae.py` → `engine_pretrain.py`.
- **Run:** `source activate mae && python main_pretrain_cmr.py --data_path <heart_roi_dir> --model mae_vit_base_patch16 --input_size 128 --in_chans 1`
- Key CMR defaults that differ from upstream MAE: `--input_size 128`, `--in_chans 1` (single-channel grayscale, not RGB), `--dataset nifti`. The model factory (`models_mae.__dict__[args.model]`) is called with `img_size` and `in_chans` passed through, so the same `mae_vit_*` definitions work for 1-channel 128px input.
- **Output layout:** `--output_dir` is automatically suffixed with `split_<id>` (or `split_random_seed_<seed>`). Each run writes `checkpoint-best.pth` (best val loss), periodic `checkpoint-<epoch>.pth` (every 20 epochs), `split.json` (exact train/val volume paths), `log.txt`, and tensorboard logs.
- **Cross-validation splits:** `--split_id 0..4` selects a fold from a fixed 5-fold partition of the sorted volume list; `--split_id -1` (with `--split_seed`) does a random `train_ratio` split instead. The split logic lives in `NiftiSliceDataset.__init__`.
- Training has **early stopping** (`--early_stop_patience`, default 20) and evaluates a val loss each epoch via `evaluate_one_epoch`.

### 2. nnUNet-based heart segmentation (upstream of analysis)

Segmentation masks are produced **externally** by an nnUNet model (`Dataset141_ACDC+MNMs`) — this repo does not run nnUNet. The label scheme from that model is used throughout: **1 = RV, 2 = myocardium, 3 = LV blood pool**. `preprocessing/extract_heart_roi_acdc_mnms.py` and `preprocessing/preprocess_MnMs.py` prepare the ACDC+M&Ms training data for that nnUNet model.

## Critical domain conventions (these recur in every script)

- **Filename pairing:** image volumes are named like `LATE_ENH_001_0000.nii.gz` (the `_0000` is nnUNet's channel suffix; `001` is the patient ID). Segmentation masks and lesion annotations drop the channel suffix: `LATE_ENH_001.nii.gz`. The standard way to recover the base name is `re.sub(r"_\d{4}$", "", stem)`; the numeric patient ID is the trailing `\d+`.
- **Two cohorts:** `control/` (healthy, ~89 patients) and `myocarditis/` (~63 patients), stored in separate directories. They are **different cohorts/scanners**, so raw MRI intensities are NOT comparable across cohorts without per-patient normalization (see below).
- **MRI intensities are arbitrary units**, not Hounsfield/CT. Volumes can contain negative values (already rescaled upstream). Any cross-patient comparison of raw intensities is confounded by scanner/acquisition; valid comparisons normalize per-patient against an internal reference (heart-ROI z-score, or LV blood-pool mean — label 3) or compare within a single patient.
- **Normalization placement:** preprocessing applies no intensity normalization (only `data - data.min()` to shift non-negative). Per-slice z-score is applied at load time in `NiftiSliceDataset.__getitem__` (toggle with `--zscore`/`--no_zscore`, default on).
- **Volume orientation:** slice axis is the 3rd dim; volumes are `(H, W, n_slices)` (sometimes 4D, in which case `[..., 0]` is taken). A "2D filter on the whole volume" means in-plane only, e.g. `median_filter(vol, size=(3, 3, 1))`.

## Standard data locations (argparse defaults)

These absolute paths are baked in as defaults across scripts; override with flags as needed:
- Raw volumes: `/gpu-data3/nikos/datasets/cmr/original/{control,myocarditis}/`
- Cropped heart ROI (pre-training input): `/gpu-data3/nikos/datasets/cmr/heart_roi/{control,myocarditis}/`
- nnUNet segmentations: `/gpu-data3/nikos/nnUnet_experiments/nnUNet_results/Dataset141_ACDC+MNMs/predict-cmr/postprocessed/`
- Expert lesion annotations (only ~49 myocarditis patients): `/gpu-data3/nikos/datasets/cmr/original/labels/`
- Outputs: `./output_dir_cmr/` (pre-training, comparisons) and `./output_dir/` (ImageNet-era)

## Exploratory / analysis scripts

These are standalone argparse scripts (run with the `mae` env) that load NIfTI volumes + masks and emit plots/derived masks under `output_dir_cmr/`. They share the filename-pairing and label conventions above:
- `compare_myocardium_intensities.py` — control vs myocarditis myocardium intensity histograms under several per-patient normalizations.
- `find_enhanced_regions.py` — flags hyper-enhanced myocardium voxels and writes new segmentation masks with label **5** added; also reports per-patient myocardium and lesion mean intensities.
- `create_healthy_lesion_histogram.py` — per-patient non-lesion vs lesion myocardium intensity histograms (uses expert lesion annotations).
- `compare_reconstruction.py` / `visualize_nikos.py` — load a pre-trained checkpoint and visualize MAE reconstructions on CMR slices (`prepare_model` builds the model with `in_chans=1, img_size=128`).

When writing a new analysis script, mirror the existing ones: argparse with the standard path defaults, `re.sub(r"_\d{4}$", "", stem)` pairing, the `1/2/3` (+`5`) label constants, and save to `output_dir_cmr/`.
# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
