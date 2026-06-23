"""Blob detection in myocarditis myocardium vs ground-truth lesion masks.

For each myocarditis patient that has a segmentation and an expert lesion
annotation, the myocardium (nnUNet label 2) is isolated and blobs are detected
slice-by-slice with three skimage methods (blob_log, blob_dog, blob_doh), keeping
only blobs whose centre falls inside the myocardium. Four separate 3D binary
masks are saved per patient (<id>_lesion_gt / _blob_log / _blob_dog / _blob_doh),
sharing the source geometry, so each can be loaded as its own Segmentation in
3D Slicer and compared against the ground-truth lesion.
"""

import sys
import os
import re
import numpy as np
import argparse
from pathlib import Path

import nibabel as nib
from skimage.feature import blob_log, blob_dog, blob_doh
from skimage.draw import disk

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(REPO_ROOT)

# nnUNet (Dataset141_ACDC+MNMs) label scheme
MYOCARDIUM_LABEL = 2

# One 3D mask file is written per patient per channel: <id>_<channel>.nii.gz
CHANNELS = ["lesion_gt", "blob_log", "blob_dog", "blob_doh"]


def _normalize_slice(slice_2d):
    """Robust min-max to [0, 1] using 1st/99th percentiles; None if ~constant."""
    lo, hi = np.percentile(slice_2d, [1, 99])
    if hi - lo < 1e-6:
        return None
    return np.clip((slice_2d - lo) / (hi - lo), 0.0, 1.0)


def _blobs_to_mask(blobs, myo_slice, radius_factor):
    """Draw filled disks for blobs whose centre is inside the myocardium.

    skimage returns (row, col, sigma) per blob; blob radius is radius_factor*sigma
    (sqrt(2)*sigma for LoG/DoG, sigma for DoH). Returns (mask, n_kept).
    """
    mask = np.zeros(myo_slice.shape, dtype=bool)
    h, w = myo_slice.shape
    n_kept = 0
    for row, col, sigma in blobs:
        ri, ci = int(round(row)), int(round(col))
        if not (0 <= ri < h and 0 <= ci < w) or not myo_slice[ri, ci]:
            continue  # keep only blobs centred in myocardium
        n_kept += 1
        radius = max(radius_factor * sigma, 1.0)
        rr, cc = disk((row, col), radius, shape=myo_slice.shape)
        mask[rr, cc] = True
    return mask & myo_slice, n_kept


def process_patient(img_path, seg_path, lesion_path, base_name, out_dir, args):
    img_obj = nib.load(img_path)
    img_data = img_obj.get_fdata(dtype=np.float32)
    if img_data.ndim > 3:
        img_data = img_data[..., 0]

    seg_data = nib.load(seg_path).get_fdata()
    if seg_data.ndim > 3:
        seg_data = seg_data[..., 0]

    lesion_data = nib.load(lesion_path).get_fdata()
    if lesion_data.ndim > 3:
        lesion_data = lesion_data[..., 0]

    if seg_data.shape != img_data.shape or lesion_data.shape != img_data.shape:
        print(f"  Warning: shape mismatch for {base_name}, skipping.")
        return None

    myo = seg_data == MYOCARDIUM_LABEL
    if not myo.any():
        print(f"  Warning: no myocardium for {base_name}, skipping.")
        return None

    # method name -> (function, radius_factor, kwargs)
    methods = [
        ("blob_log", blob_log, np.sqrt(2), dict(
            min_sigma=args.min_sigma, max_sigma=args.max_sigma,
            num_sigma=args.num_sigma, threshold=args.threshold_log)),
        ("blob_dog", blob_dog, np.sqrt(2), dict(
            min_sigma=args.min_sigma, max_sigma=args.max_sigma,
            threshold=args.threshold_dog)),
        ("blob_doh", blob_doh, 1.0, dict(
            min_sigma=args.min_sigma, max_sigma=args.max_sigma,
            num_sigma=args.num_sigma, threshold=args.threshold_doh)),
    ]

    masks = {name: np.zeros(img_data.shape, dtype=np.uint8) for name in CHANNELS}
    masks["lesion_gt"] = (lesion_data > 0).astype(np.uint8)
    counts = {name: 0 for name, *_ in methods}

    for z in range(img_data.shape[2]):
        myo_slice = myo[:, :, z]
        if not myo_slice.any():
            continue
        norm = _normalize_slice(img_data[:, :, z])
        if norm is None:
            continue
        for name, fn, rfactor, kwargs in methods:
            blobs = fn(norm, **kwargs)
            mask, n_kept = _blobs_to_mask(blobs, myo_slice, rfactor)
            masks[name][:, :, z] = mask.astype(np.uint8)
            counts[name] += n_kept

    for name, arr in masks.items():
        out_img = nib.Nifti1Image(arr, img_obj.affine, img_obj.header)
        out_img.header.set_data_dtype(np.uint8)
        nib.save(out_img, out_dir / f"{base_name}_{name}.nii.gz")
    return counts, int(masks["lesion_gt"].sum())


def main(args):
    img_dir = Path(args.input_myocarditis_dir)
    seg_dir = Path(args.input_segmentation_dir)
    lesion_dir = Path(args.input_lesion_dir)
    out_dir = args.output_dir

    print(f"Per-patient files: <id>_{{{','.join(CHANNELS)}}}.nii.gz")
    n_done = 0
    for img_path in sorted(img_dir.glob("*.nii.gz")):
        # Strip _0000 (nnUNet channel suffix) to get the base patient name
        stem = img_path.name.replace(".nii.gz", "")
        base_name = re.sub(r"_\d{4}$", "", stem)  # e.g. LATE_ENH_152

        seg_path = seg_dir / f"{base_name}.nii.gz"
        lesion_path = lesion_dir / f"{base_name}.nii.gz"
        if not seg_path.exists() or not lesion_path.exists():
            continue  # only patients with both a segmentation and a lesion annotation

        result = process_patient(img_path, seg_path, lesion_path, base_name, out_dir, args)
        if result is None:
            continue
        counts, n_gt = result
        n_done += 1
        print(f"  {base_name}: lesion_gt={n_gt} vox; "
              f"LoG={counts['blob_log']}, DoG={counts['blob_dog']}, "
              f"DoH={counts['blob_doh']} blobs -> {base_name}_*.nii.gz")

    print(f"\nSaved {n_done} blob/lesion NIfTI files to {out_dir}")


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Blob detection in myocardium',
        description="Detect blobs in myocarditis myocardium and save them with lesion GT.",
        add_help=False,
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
        default="./output_dir_cmr/blobs/",
        help="Directory to save the per-patient blob/lesion NIfTI files.",
    )

    # Blob-detection parameters (operate on per-slice [0, 1]-normalized intensities)
    parser.add_argument("--min_sigma", type=float, default=1.0)
    parser.add_argument("--max_sigma", type=float, default=10.0)
    parser.add_argument("--num_sigma", type=int, default=10,
                        help="Number of sigmas for blob_log / blob_doh.")
    parser.add_argument("--threshold_log", type=float, default=0.1)
    parser.add_argument("--threshold_dog", type=float, default=0.1)
    parser.add_argument("--threshold_doh", type=float, default=0.005)

    return parser


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
