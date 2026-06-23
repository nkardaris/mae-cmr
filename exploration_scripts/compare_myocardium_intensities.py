import sys
import os
import re
import numpy as np
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(REPO_ROOT)

# nnUNet (Dataset141_ACDC+MNMs) label scheme
RV_LABEL = 1
MYOCARDIUM_LABEL = 2
BLOODPOOL_LABEL = 3  # LV cavity / blood pool


def collect_patient_stats(img_dir, seg_dir):
    """Per-patient z-normalized myocardium voxels + blood-pool reference.

    Each volume is z-normalized using the whole heart-ROI (seg > 0) mean/std
    before any metric is computed. Returns a list of dicts, one per patient:
        myo_z        : z-normalized myocardium voxel intensities (1D array)
        blood_mean_z : z-normalized LV blood-pool mean (np.nan if label absent)
    """
    img_dir = Path(img_dir)
    seg_dir = Path(seg_dir)
    patients = []

    for img_path in sorted(img_dir.glob("*.nii.gz")):
        # Strip _0000 (nnUNet channel suffix) to get the base patient name
        stem = img_path.name.replace(".nii.gz", "")
        base_name = re.sub(r"_\d{4}$", "", stem)  # e.g. LATE_ENH_001

        seg_path = seg_dir / f"{base_name}.nii.gz"
        if not seg_path.exists():
            print(f"  Warning: no segmentation for {img_path.name}, skipping.")
            continue

        img_data = nib.load(img_path).get_fdata(dtype=np.float32)
        if img_data.ndim > 3:
            img_data = img_data[..., 0]

        seg_data = nib.load(seg_path).get_fdata(dtype=np.float32)
        if seg_data.ndim > 3:
            seg_data = seg_data[..., 0]

        if not np.any(seg_data == MYOCARDIUM_LABEL):
            print(f"  Warning: no myocardium voxels found in {img_path.name}.")
            continue

        # z-normalize the whole volume using heart-ROI (seg > 0) stats
        roi_voxels = img_data[seg_data > 0]
        roi_mean = float(roi_voxels.mean())
        roi_std = float(roi_voxels.std())
        if roi_std < 1e-6:
            roi_std = 1.0
        img_z = (img_data - roi_mean) / roi_std

        myo_z = img_z[seg_data == MYOCARDIUM_LABEL]
        blood_z = img_z[seg_data == BLOODPOOL_LABEL]
        blood_mean_z = float(blood_z.mean()) if blood_z.size > 0 else np.nan

        patients.append({
            "myo_z": myo_z,
            "blood_mean_z": blood_mean_z,
        })

    return patients


def bloodpool_ratio(patients):
    """Pooled per-voxel z-norm myocardium/blood-pool ratios (no blood pool -> skipped)."""
    vals = [p["myo_z"] / p["blood_mean_z"] for p in patients if np.isfinite(p["blood_mean_z"])]
    n_skipped = sum(1 for p in patients if not np.isfinite(p["blood_mean_z"]))
    return (np.concatenate(vals) if vals else np.array([])), n_skipped


def per_patient_zscore(patients):
    """Pooled per-voxel z-scores (volumes already z-normalized by heart-ROI stats)."""
    return np.concatenate([p["myo_z"] for p in patients]) if patients else np.array([])


def focal_enhancement_fraction(patients, n_sd=5.0):
    """One scalar per patient: fraction of myocardium > mean + n_sd*std (own myo)."""
    out = []
    for p in patients:
        m = p["myo_z"]
        out.append(float(np.mean(m > m.mean() + n_sd * m.std())))
    return np.array(out)


def _overlay_hist(ax, ctrl_vals, myo_vals, n_ctrl, n_myo, xlabel, title, bins=60):
    all_vals = np.concatenate([ctrl_vals, myo_vals])
    if all_vals.size == 0:
        ax.set_title(title + " (no data)")
        return
    lo, hi = np.percentile(all_vals, [0.5, 99.5])
    edges = np.linspace(lo, hi, bins + 1)
    ax.hist(ctrl_vals, bins=edges, density=True, alpha=0.6,
            color="steelblue", label=f"Control (n={n_ctrl})")
    ax.hist(myo_vals, bins=edges, density=True, alpha=0.6,
            color="tomato", label=f"Myocarditis (n={n_myo})")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)


def main(args):
    print("Processing control group...")
    control = collect_patient_stats(args.input_control_dir, args.input_segmentation_dir)
    print(f"  {len(control)} patients collected.")

    print("Processing myocarditis group...")
    myocarditis = collect_patient_stats(args.input_myocarditis_dir, args.input_segmentation_dir)
    print(f"  {len(myocarditis)} patients collected.")

    n_ctrl, n_myo = len(control), len(myocarditis)

    # 1) Myocardium-to-blood-pool ratio (pooled per voxel)
    ctrl_ratio, ctrl_skip = bloodpool_ratio(control)
    myo_ratio, myo_skip = bloodpool_ratio(myocarditis)
    if ctrl_skip or myo_skip:
        print(f"  Blood-pool ratio: skipped {ctrl_skip} control / {myo_skip} "
              f"myocarditis patients lacking an LV blood-pool (label {BLOODPOOL_LABEL}).")

    # 2) Per-patient z-score (pooled per voxel)
    ctrl_z = per_patient_zscore(control)
    myo_z = per_patient_zscore(myocarditis)

    # 3) Focal enhancement fraction (one value per patient)
    ctrl_focal = focal_enhancement_fraction(control)
    myo_focal = focal_enhancement_fraction(myocarditis)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    _overlay_hist(
        axes[0], ctrl_ratio, myo_ratio,
        n_ctrl - ctrl_skip, n_myo - myo_skip,
        "Myocardium / blood-pool", "Myo-to-blood-pool ratio (per voxel)",
    )
    _overlay_hist(
        axes[1], ctrl_z, myo_z, n_ctrl, n_myo,
        "Per-patient z-score (a.u.)", "Per-patient z-score (per voxel)",
    )
    _overlay_hist(
        axes[2], ctrl_focal, myo_focal, n_ctrl, n_myo,
        "Focal enhancement fraction", "Focal enhancement (>mean+5SD, per patient)",
        bins=30,
    )

    fig.suptitle("Myocardium Intensity: Control vs Myocarditis (LGE CMR), normalized", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = args.output_dir / "myocardium_intensity_comparison_normalized.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_path}")


def get_args_parser():
    parser = argparse.ArgumentParser('MAE CMR reconstruction experiments', add_help=False)

    # Dataset parameters
    parser.add_argument(
        "--input_control_dir",
        default="/gpu-data3/nikos/datasets/cmr/original/control/",
        help="Directory with input image volumes (.nii.gz).",
    )
    parser.add_argument(
        "--input_myocarditis_dir",
        default="/gpu-data3/nikos/datasets/cmr/original/myocarditis/",
        help="Directory with input image volumes (.nii.gz).",
    )    
    parser.add_argument(
        "--input_segmentation_dir",
        default="/gpu-data3/nikos/nnUnet_experiments/nnUNet_results/Dataset141_ACDC+MNMs/predict-cmr/postprocessed/",
        help="Directory with heart segmentation masks (.nii.gz).",
    )
    parser.add_argument(
        "--output_dir",
        default="./output_dir_cmr/intensity_comparison/",
        help="Directory to save output files.",
    )    
    
    return parser



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        args.output_dir = Path(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
