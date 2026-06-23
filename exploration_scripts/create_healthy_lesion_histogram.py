"""Per-patient histograms of myocardium raw intensity, lesion vs non-lesion.

For each patient that has an expert lesion annotation, the myocardium (nnUNet
label 2) is split into lesion and non-lesion voxels and their intensity
distributions are plotted together with the lesion's 1st-percentile threshold.
Also reports, per patient, that threshold relative to the blood-pool mean and the
percentile of non-lesion myocardium that falls below it, plus the median of that
percentile across all patients (a lesion/healthy separability measure).
"""

import sys
import os
import re
import numpy as np
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import median_filter

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(REPO_ROOT)

# nnUNet (Dataset141_ACDC+MNMs) label scheme
MYOCARDIUM_LABEL = 2
BLOODPOOL_LABEL = 3  # LV cavity / blood pool


def plot_patient_histogram(lesion_vals, nonlesion_vals, patient_id, out_dir, threshold):
    """Save a non-lesion vs lesion raw-intensity histogram for one patient."""
    all_vals = np.concatenate([nonlesion_vals, lesion_vals])
    lo, hi = np.percentile(all_vals, [0.5, 99.5])
    bins = np.linspace(lo, hi, 61)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(nonlesion_vals, bins=bins, density=True, alpha=0.6, color="steelblue",
            label=f"Non-lesion myocardium ({nonlesion_vals.size:,} voxels)")
    ax.hist(lesion_vals, bins=bins, density=True, alpha=0.6, color="tomato",
            label=f"Lesion myocardium ({lesion_vals.size:,} voxels)")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"1st-pct lesion threshold = {threshold:.1f}")
    ax.set_xlabel("Raw intensity (a.u.)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(f"Patient {patient_id}: myocardium raw intensity, non-lesion vs lesion",
                 fontsize=13)
    ax.legend(fontsize=11)
    fig.tight_layout()

    out_path = out_dir / f"{patient_id}_healthy_lesion_histogram.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def process_group(name, img_dir, seg_dir, lesion_dir, out_dir):
    """Plot a per-patient histogram for every patient with a lesion annotation."""
    img_dir = Path(img_dir)
    seg_dir = Path(seg_dir)
    lesion_dir = Path(lesion_dir)
    print(f"Processing {name} group...")

    n_patients = 0
    nonlesion_pcts = []
    for img_path in sorted(img_dir.glob("*.nii.gz")):
        # Strip _0000 (nnUNet channel suffix) to get the base patient name
        stem = img_path.name.replace(".nii.gz", "")
        base_name = re.sub(r"_\d{4}$", "", stem)  # e.g. LATE_ENH_001
        m = re.search(r"(\d+)$", base_name)
        patient_id = m.group(1) if m else base_name  # e.g. 001

        lesion_path = lesion_dir / f"{base_name}.nii.gz"
        seg_path = seg_dir / f"{base_name}.nii.gz"
        if not lesion_path.exists() or not seg_path.exists():
            continue  # only patients with both an annotation and a segmentation

        img_data = nib.load(img_path).get_fdata(dtype=np.float32)
        if img_data.ndim > 3:
            img_data = img_data[..., 0]

        # 2D 3x3 median filter applied per slice (in-plane only, no mixing across slices)
        img_data = median_filter(img_data, size=(3, 3, 1))

        # img_data = (img_data - img_data.min()) / img_data.std()  # z-normalize the whole volume

        seg_data = nib.load(seg_path).get_fdata()
        if seg_data.ndim > 3:
            seg_data = seg_data[..., 0]

        lesion_data = nib.load(lesion_path).get_fdata()
        if lesion_data.ndim > 3:
            lesion_data = lesion_data[..., 0]

        if lesion_data.shape != img_data.shape or seg_data.shape != img_data.shape:
            print(f"  Warning: shape mismatch for {base_name}, skipping.")
            continue

        myo_mask = seg_data == MYOCARDIUM_LABEL
        if not np.any(myo_mask):
            print(f"  Warning: no myocardium for {base_name}, skipping.")
            continue

        lesion_mask = myo_mask & (lesion_data > 0)
        nonlesion_mask = myo_mask & ~(lesion_data > 0)
        if not np.any(lesion_mask):
            print(f"  Warning: lesion does not overlap myocardium for {base_name}, skipping.")
            continue

        lesion_vals = img_data[lesion_mask]
        nonlesion_vals = img_data[nonlesion_mask]

        # Robust lesion threshold (5th percentile), expressed relative to the
        # blood-pool mean: ~95% of lesion voxels lie above this threshold.
        blood_voxels = img_data[seg_data == BLOODPOOL_LABEL]
        blood_mean = float(blood_voxels.mean()) if blood_voxels.size > 0 else float("nan")
        threshold = float(np.percentile(lesion_vals, 1))
        ratio = threshold / blood_mean

        # Percentile rank of the threshold within non-lesion myocardium: fraction
        # of healthy voxels darker than the lesion's 1st percentile (high = good separation).
        nonlesion_pct = 100.0 * float(np.mean(nonlesion_vals < threshold))
        nonlesion_pcts.append(nonlesion_pct)

        out_path = plot_patient_histogram(lesion_vals, nonlesion_vals, patient_id, out_dir, threshold)
        n_patients += 1

        print(f"  [{name}] {base_name}: non-lesion={nonlesion_vals.size:,} "
              f"(mean={nonlesion_vals.mean():.2f}), lesion={lesion_vals.size:,} "
              f"(mean={lesion_vals.mean():.2f}) -> {out_path.name}")
        print(f"      threshold={threshold:.2f} = {ratio:.3f} * blood_mean "
              f"(blood_mean={blood_mean:.2f}); non-lesion {nonlesion_pct:.1f}th pct below it")

    return n_patients, nonlesion_pcts


def main(args):
    total = 0
    all_pcts = []
    for name, img_dir in [("control", args.input_control_dir),
                          ("myocarditis", args.input_myocarditis_dir)]:
        n, pcts = process_group(name, img_dir, args.input_segmentation_dir,
                                args.input_lesion_dir, args.output_dir)
        total += n
        all_pcts.extend(pcts)
    print(f"\nSaved {total} per-patient histograms to {args.output_dir}")
    if all_pcts:
        print(f"Median non-lesion percentile across {len(all_pcts)} patients: "
              f"{np.median(all_pcts):.1f}th pct")


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Healthy vs lesion myocardium histogram',
        description="Per-patient lesion vs non-lesion myocardium intensity histograms.",
        add_help=False,
    )

    parser.add_argument(
        "--input_control_dir",
        default="/gpu-data3/nikos/datasets/cmr/original/control/",
        help="Directory with control input image volumes (.nii.gz).",
    )
    parser.add_argument(
        "--input_myocarditis_dir",
        default="/gpu-data3/nikos/datasets/cmr/original/myocarditis/",
        help="Directory with myocarditis input image volumes (.nii.gz).",
    )
    parser.add_argument(
        "--input_segmentation_dir",
        default="/gpu-data3/nikos/nnUnet_experiments/nnUNet_results/Dataset141_ACDC+MNMs/predict-cmr/postprocessed/",
        help="Directory with heart segmentation masks (.nii.gz).",
    )
    parser.add_argument(
        "--input_lesion_dir",
        default="/gpu-data3/nikos/datasets/cmr/original/labels",
        help="Directory with expert lesion annotations (.nii.gz).",
    )
    parser.add_argument(
        "--output_dir",
        default="./output_dir_cmr/exploratory_results/",
        help="Directory to save the output plots.",
    )

    return parser


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
