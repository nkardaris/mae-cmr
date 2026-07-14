
import argparse
import re
from typing import cast

import nibabel as nib
import numpy as np
from pathlib import Path
from scipy.ndimage import binary_dilation, zoom


def _resize_volume_slices(volume: np.ndarray, target_width: int, target_height: int, order: int) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")

    resized_slices = []
    for z in range(volume.shape[2]):
        slice_2d = volume[:, :, z]
        if slice_2d.shape == (target_height, target_width):
            resized_slice = slice_2d
        else:
            zoom_factors = (
                float(target_height) / float(slice_2d.shape[0]),
                float(target_width) / float(slice_2d.shape[1]),
            )
            resized_slice = zoom(slice_2d, zoom=zoom_factors, order=order, mode="nearest")

            # Guard against occasional off-by-one output shapes from interpolation rounding.
            if resized_slice.shape != (target_height, target_width):
                aligned_slice = np.zeros((target_height, target_width), dtype=resized_slice.dtype)
                copy_h = min(target_height, resized_slice.shape[0])
                copy_w = min(target_width, resized_slice.shape[1])
                aligned_slice[:copy_h, :copy_w] = resized_slice[:copy_h, :copy_w]
                resized_slice = aligned_slice

        resized_slices.append(resized_slice)

    return cast(np.ndarray, np.stack(resized_slices, axis=2))


def _znorm_volume_roi(volume: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """Z-score the whole volume using the mean/std of the heart-ROI voxels only.

    The heart ROI is anatomically consistent across patients, unlike the bounding-box
    crop (whose lung/chest-wall content varies), so its statistics make intensities
    comparable between patients and cohorts.
    """
    roi_voxels = volume[roi_mask]
    std = float(roi_voxels.std())
    if std < 1e-6:
        std = 1.0
    return (volume.astype(np.float32) - float(roi_voxels.mean())) / std


def extract_heart_roi_from_myocarditis_dataset(
    raw_data_dir,
    segmentation_dir,
    lesion_dir,
    images_output_dir,
    segmentation_output_dir,
    lesion_output_dir,
    width=None,
    height=None,
    znorm=False,
    zero_background=False,
):
    raw_data_dir = Path(raw_data_dir)
    segmentation_dir = Path(segmentation_dir)
    lesion_dir = Path(lesion_dir)
    images_output_dir = Path(images_output_dir)
    segmentation_output_dir = Path(segmentation_output_dir)
    lesion_output_dir = Path(lesion_output_dir)
    images_output_dir.mkdir(parents=True, exist_ok=True)
    segmentation_output_dir.mkdir(parents=True, exist_ok=True)
    lesion_output_dir.mkdir(parents=True, exist_ok=True)

    should_resize_slices = False
    target_width = None
    target_height = None
    if width is None and height is None:
        pass
    elif width is not None and height is not None:
        should_resize_slices = True
        target_width = int(width)
        target_height = int(height)
        if target_width <= 0 or target_height <= 0:
            raise ValueError(
                f"width and height must be positive integers, got width={target_width}, height={target_height}"
            )
    else:
        raise ValueError("Both width and height must be provided together for resizing.")

    # Get list of all raw data files
    raw_data_files = sorted(raw_data_dir.glob("*.nii.gz"))
    total_files = len(raw_data_files)
    processed_count = 0
    missing_seg_count = 0
    failed_count = 0
    segmentation_saved_count = 0
    lesion_saved_count = 0
    lesion_missing_count = 0
    lesion_failed_count = 0
    discarded_slice_count = 0
    empty_volume_count = 0

    
    for raw_file in raw_data_files:
        # Extract patient ID from nnUNet image naming, e.g. LATE_ENH_001_0000.nii.gz -> LATE_ENH_001
        raw_name = raw_file.name
        if raw_name.endswith(".nii.gz"):
            raw_stem = raw_name[:-7]
        else:
            print(f"Unsupported file extension for {raw_name}. Skipping.")
            continue

        # Remove modality/channel suffix like _0000 used by nnUNet images.
        patient_id = re.sub(r"_\d{4}$", "", raw_stem)

        # Find corresponding segmentation file
        seg_file = segmentation_dir / f"{patient_id}.nii.gz"
        if not seg_file.is_file():
            print(f"Segmentation file not found for patient {patient_id}. Skipping.")
            missing_seg_count += 1
            continue

        try:
            # Load raw image and segmentation
            raw_img = nib.load(raw_file)
            seg_img = nib.load(seg_file)

            raw_data = raw_img.get_fdata()
            seg_data = seg_img.get_fdata().round().astype("int16")

            # Create a mask for the heart 
            roi_mask = seg_data > 0  # Assuming non-zero labels correspond to heart regions; adjust if needed for specific label values.

            if roi_mask.ndim != 3 or raw_data.ndim != 3:
                failed_count += 1
                print(f"Image/segmentation is not 3D for patient {patient_id}. Skipping.")
                continue
            if raw_data.shape != seg_data.shape:
                failed_count += 1
                print(f"Image/segmentation shape mismatch for patient {patient_id}. Skipping.")
                continue

            # Dilate the heart mask with a 3x3 structuring element.
            structure = np.ones((3, 3), dtype=bool)
            dilated_slices = [
                binary_dilation(roi_mask[:, :, z], structure=structure)
                for z in range(roi_mask.shape[2])
            ]
            dilated_heart_mask = np.stack(dilated_slices, axis=2)

            # Compute a 3D bounding box around the dilated heart mask and crop the image.
            foreground_coords = np.argwhere(dilated_heart_mask)
            if foreground_coords.size == 0:
                failed_count += 1
                print(f"Dilated heart mask is empty for patient {patient_id}. Skipping.")
                continue

            bbox_min = foreground_coords.min(axis=0)
            bbox_max = foreground_coords.max(axis=0) + 1  # exclusive upper bound
            crop_slices = tuple(slice(int(start), int(stop)) for start, stop in zip(bbox_min, bbox_max))

            if any(raw_data.shape[dim] < int(bbox_max[dim]) for dim in range(3)):
                failed_count += 1
                print(f"Bounding box exceeds image shape for patient {patient_id}. Skipping.")
                continue

            cropped_seg = seg_data[crop_slices]
            keep_indices = []
            discarded_indices = []
            for z in range(cropped_seg.shape[2]):
                label_slice = cropped_seg[:, :, z]
                present_structures = 0
                for label_value in (1, 2, 3):
                    if np.count_nonzero(label_slice == label_value) > 50:
                        present_structures += 1
                if present_structures >= 2:
                    keep_indices.append(z)
                else:
                    discarded_indices.append(z)

            if discarded_indices:
                discarded_slice_count += len(discarded_indices)
                print(
                    f"Discarded {len(discarded_indices)} slices for patient {patient_id} "
                    f"(need >=2 structures with >50 pixels): {discarded_indices}"
                )
            if not keep_indices:
                empty_volume_count += 1
                print(
                    f"No slices meet keep criteria (>=2 structures with >50 pixels) for patient {patient_id}. "
                    "Skipping."
                )
                continue

            heart_roi = raw_data[crop_slices][:, :, keep_indices]

            if znorm:
                heart_roi = _znorm_volume_roi(heart_roi, cropped_seg[:, :, keep_indices] > 0)

            if zero_background:
                # Fill with the minimum rather than a literal 0: after z-scoring, 0 is the
                # tissue mean, so a 0 background would sit inside the myocardium distribution.
                cropped_mask = dilated_heart_mask[crop_slices][:, :, keep_indices]
                heart_roi = np.where(cropped_mask, heart_roi, heart_roi.min())

            if should_resize_slices:
                assert target_width is not None and target_height is not None
                heart_roi = _resize_volume_slices(
                    heart_roi, target_width=target_width, target_height=target_height, order=1
                )

            # Save the extracted heart ROI as a new NIfTI file
            roi_img = nib.Nifti1Image(heart_roi, affine=raw_img.affine, header=raw_img.header)
            output_path = images_output_dir / raw_file.name
            nib.save(roi_img, output_path)
            processed_count += 1
            print(
                f"Saved heart ROI for patient ({processed_count}/{total_files}) {patient_id} to {output_path} "
                f"with bbox min={tuple(bbox_min.tolist())}, max={tuple((bbox_max - 1).tolist())}"
            )

            # Crop and save the original segmentation labels with the same bounding box.
            seg_data = cropped_seg[:, :, keep_indices]
            if should_resize_slices:
                assert target_width is not None and target_height is not None
                seg_data = _resize_volume_slices(
                    seg_data, target_width=target_width, target_height=target_height, order=0
                ).astype("int16")

            seg_output_path = segmentation_output_dir / seg_file.name
            trimmed_seg_img = nib.Nifti1Image(seg_data, affine=seg_img.affine, header=seg_img.header)
            nib.save(trimmed_seg_img, seg_output_path)
            segmentation_saved_count += 1
            print(
                f"Saved trimmed segmentation for patient {patient_id} to {seg_output_path} "
                f"(original shape: {seg_img.shape}, trimmed shape: {seg_data.shape})"
            )

            # Find and process the associated lesion mask with the same bounding-box crop.
            lesion_file = lesion_dir / f"{patient_id}.nii.gz"
            if not lesion_file.is_file():
                lesion_missing_count += 1
                print(f"Associated lesion file not found for patient {patient_id}.")
                continue

            try:
                lesion_img = nib.load(lesion_file)
                lesion_data = lesion_img.get_fdata().round().astype("int16")

                if lesion_data.ndim != 3:
                    lesion_failed_count += 1
                    print(f"Lesion mask for patient {patient_id} is not 3D, cannot apply bbox cropping.")
                    continue
                if lesion_data.shape != raw_data.shape:
                    lesion_failed_count += 1
                    print(f"Lesion/image shape mismatch for patient {patient_id}.")
                    continue

                lesion_data = lesion_data[crop_slices][:, :, keep_indices]
                if should_resize_slices:
                    assert target_width is not None and target_height is not None
                    lesion_data = _resize_volume_slices(
                        lesion_data, target_width=target_width, target_height=target_height, order=0
                    ).astype("int16")

                lesion_output_path = lesion_output_dir / lesion_file.name
                trimmed_lesion_img = nib.Nifti1Image(lesion_data, affine=lesion_img.affine, header=lesion_img.header)
                nib.save(trimmed_lesion_img, lesion_output_path)
                lesion_saved_count += 1
                print(
                    f"Saved trimmed lesion for patient {patient_id} to {lesion_output_path} "
                    f"(original shape: {lesion_img.shape}, trimmed shape: {lesion_data.shape})"
                )
            except Exception as lesion_exc:
                lesion_failed_count += 1
                print(f"Failed processing lesion for patient {patient_id}: {lesion_exc}")
        except Exception as exc:
            failed_count += 1
            print(f"Failed processing patient {patient_id}: {exc}")

    print(
        "Summary: "
        f"total={total_files}, processed={processed_count}, "
        f"missing_seg={missing_seg_count}, failed={failed_count}, "
        f"segmentations_saved={segmentation_saved_count}, lesions_saved={lesion_saved_count}, "
        f"lesions_missing={lesion_missing_count}, lesions_failed={lesion_failed_count}, "
        f"discarded_slices={discarded_slice_count}, empty_volumes={empty_volume_count}"
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crop CMR volumes around heart ROI and resize slices (optional)."
    )
    parser.add_argument(
        "--input_images_dir",
        default="/gpu-data3/nikos/datasets/cmr/original/control/",
        help="Directory with input image volumes (.nii.gz).",
    )
    parser.add_argument(
        "--input_segmentation_dir",
        default="/gpu-data3/nikos/nnUnet_experiments/nnUNet_results/Dataset141_ACDC+MNMs/predict-cmr/postprocessed/",
        help="Directory with heart segmentation masks (.nii.gz).",
    )
    parser.add_argument(
        "--input_lesion_dir",
        default="/gpu-data3/nikos/datasets/cmr/original/labels/",
        help="Directory with lesion masks (.nii.gz).",
    )
    parser.add_argument(
        "--output_images_dir",
        default="/gpu-data3/nikos/datasets/cmr/heart_roi_raw/control/",
        help="Directory to write cropped image volumes.",
    )
    parser.add_argument(
        "--output_segmentation_dir",
        default="/gpu-data3/nikos/datasets/cmr/heart_roi_raw/segmentation/",
        help="Directory to write cropped heart segmentation masks.",
    )
    parser.add_argument(
        "--output_lesion_dir",
        default="/gpu-data3/nikos/datasets/cmr/heart_roi_raw/labels/",
        help="Directory to write cropped lesion masks.",
    )
    parser.add_argument("--width", type=int, default=None, help="Target slice width (optional).")
    parser.add_argument("--height", type=int, default=None, help="Target slice height (optional).")
    parser.add_argument(
        "--znorm",
        action="store_true",
        help="Z-score the cropped image volumes using heart-ROI (seg > 0) mean/std.",
    )
    parser.add_argument(
        "--zero_background",
        action="store_true",
        help="Zero out the voxels outside the dilated heart mask.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    extract_heart_roi_from_myocarditis_dataset(
        raw_data_dir=args.input_images_dir,
        segmentation_dir=args.input_segmentation_dir,
        lesion_dir=args.input_lesion_dir,
        images_output_dir=args.output_images_dir,
        segmentation_output_dir=args.output_segmentation_dir,
        lesion_output_dir=args.output_lesion_dir,
        width=args.width,
        height=args.height,
        znorm=args.znorm,
        zero_background=args.zero_background,
    )


if __name__ == "__main__":
    main()