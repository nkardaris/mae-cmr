"""Flag enhanced myocardium voxels and write relabeled segmentation masks.

For each patient, the volume is normalized by the LV blood-pool mean and
myocardium (nnUNet label 2) voxels above a threshold (myocardium mean + n_std *
std) are marked as ENHANCED_LABEL (5) in a copy of the segmentation mask, saved
flat to the output dir. When an expert lesion annotation exists, the mean raw
intensity of the lesion voxels is also reported.
"""

import sys
import os
import re
import numpy as np
import argparse
from pathlib import Path

import nibabel as nib

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(REPO_ROOT)

# nnUNet (Dataset141_ACDC+MNMs) label scheme
RV_LABEL = 1
MYOCARDIUM_LABEL = 2
BLOODPOOL_LABEL = 3  # LV cavity / blood pool
ENHANCED_LABEL = 5   # label written for enhanced myocardium voxels


def process_group(name, img_dir, seg_dir, out_dir, n_std, lesion_dir):
    """Detect enhanced myocardium per patient and write a relabeled mask.

    For each volume: normalize by the LV blood-pool mean, then flag myocardium
    voxels whose normalized intensity exceeds (myo_mean + n_std * myo_std). Those
    voxels are written as ENHANCED_LABEL into a copy of the segmentation mask.
    If an expert lesion annotation exists for the patient, the mean RAW intensity
    of the lesion voxels is also reported.
    """
    img_dir = Path(img_dir)
    seg_dir = Path(seg_dir)
    lesion_dir = Path(lesion_dir)
    print(f"Processing {name} group...")

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

        seg_img = nib.load(seg_path)
        seg_data = seg_img.get_fdata()
        if seg_data.ndim > 3:
            seg_data = seg_data[..., 0]

        myo_mask = seg_data == MYOCARDIUM_LABEL
        if not np.any(myo_mask):
            print(f"  Warning: no myocardium in {seg_path.name}, skipping.")
            continue

        blood_voxels = img_data[seg_data == BLOODPOOL_LABEL]
        if blood_voxels.size == 0:
            print(f"  Warning: no blood pool in {seg_path.name}, skipping.")
            continue

        # Step 2: normalize the whole volume by the blood-pool mean
        blood_mean = 1 #float(blood_voxels.mean())
        img_norm = img_data / blood_mean

        # Step 3: threshold within the myocardium
        myo_vals = img_norm[myo_mask]
        myo_mean = float(myo_vals.mean())
        thresh = myo_mean #+ n_std * myo_vals.std()
        enhanced = myo_mask & (img_norm > thresh)

        # Step 4: write ENHANCED_LABEL into a copy of the segmentation mask
        out_data = np.rint(seg_data).astype(np.uint8)
        out_data[enhanced] = ENHANCED_LABEL

        out_img = nib.Nifti1Image(out_data, seg_img.affine, seg_img.header)
        out_img.header.set_data_dtype(np.uint8)
        out_path = out_dir / seg_path.name
        nib.save(out_img, out_path)

        # Optional expert lesion annotation: mean of RAW intensities in lesion
        lesion_path = lesion_dir / f"{base_name}.nii.gz"
        lesion_str = "  lesion_mean=n/a"
        if lesion_path.exists():
            lesion_data = nib.load(lesion_path).get_fdata()
            if lesion_data.ndim > 3:
                lesion_data = lesion_data[..., 0]
            if lesion_data.shape != img_data.shape:
                lesion_str = (f"  lesion shape {lesion_data.shape} != "
                              f"img {img_data.shape}, skipped")
            else:
                lesion_mask = lesion_data > 0
                if np.any(lesion_mask):
                    lesion_mean = float(img_data[lesion_mask].mean())
                    lesion_str = (f"  lesion_mean={lesion_mean:.4f} "
                                  f"({int(lesion_mask.sum())} vox)")
                else:
                    lesion_str = "  lesion_mean=n/a (empty annotation)"

        n_enh = int(enhanced.sum())
        n_myo = int(myo_mask.sum())
        print(f"  [{name}] {seg_path.name}: myo_mean={myo_mean:.4f}{lesion_str}  "
              f"{n_enh}/{n_myo} myocardium voxels enhanced "
              f"({100 * n_enh / n_myo:.2f}%) -> {out_path.name}")


def main(args):
    process_group("control", args.input_control_dir,
                  args.input_segmentation_dir, args.output_dir, args.n_std,
                  args.input_lesion_dir)
    process_group("myocarditis", args.input_myocarditis_dir,
                  args.input_segmentation_dir, args.output_dir, args.n_std,
                  args.input_lesion_dir)
    print(f"\nEnhanced-region masks saved to {args.output_dir}")


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Find enhanced myocardium regions',
        description="Mark enhanced myocardium voxels (label 5) in copied segmentation masks.",
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
        help="Directory with expert lesion annotations (.nii.gz); optional per patient.",
    )
    parser.add_argument(
        "--output_dir",
        default="./output_dir_cmr/enhanced_regions/",
        help="Flat directory to save relabeled masks (both groups together).",
    )
    parser.add_argument(
        "--n_std",
        type=float,
        default=4.0,
        help="Number of standard deviations above the myocardium mean for the threshold.",
    )

    return parser


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
