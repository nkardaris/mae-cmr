
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


def extract_heart_roi_from_myocarditis_dataset(raw_data_dir, labels_dir, segmentation_dir, images_output_dir, labels_output_dir, width=None, height=None):
    raw_data_dir = Path(raw_data_dir)
    labels_dir = Path(labels_dir)
    segmentation_dir = Path(segmentation_dir)
    images_output_dir = Path(images_output_dir)
    labels_output_dir = Path(labels_output_dir)
    images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)

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
    raw_data_files = list(raw_data_dir.glob("*.nii*"))
    total_files = len(raw_data_files)
    processed_count = 0
    missing_seg_count = 0
    failed_count = 0
    label_saved_count = 0
    label_missing_count = 0
    label_failed_count = 0

    
    for raw_file in raw_data_files:
        # Extract patient ID from nnUNet image naming, e.g. LATE_ENH_001_0000.nii.gz -> LATE_ENH_001
        raw_name = raw_file.name
        if raw_name.endswith(".nii.gz"):
            raw_stem = raw_name[:-7]
        elif raw_name.endswith(".nii"):
            raw_stem = raw_name[:-4]
        else:
            print(f"Unsupported file extension for {raw_name}. Skipping.")
            continue

        # Remove modality/channel suffix like _0000 used by nnUNet images.
        patient_id = re.sub(r"_\d{4}$", "", raw_stem)

        # Find corresponding segmentation file
        seg_candidates = [
            segmentation_dir / f"{patient_id}.nii.gz",
            segmentation_dir / f"{patient_id}.nii",
        ]
        seg_file = next((p for p in seg_candidates if p.is_file()), None)
        if seg_file is None:
            print(f"Segmentation file not found for patient {patient_id}. Skipping.")
            missing_seg_count += 1
            continue

        try:
            # Load raw image and segmentation
            raw_img = nib.load(raw_file)
            seg_img = nib.load(seg_file)

            raw_data = raw_img.get_fdata()
            seg_data = seg_img.get_fdata()

            # Create a mask for the myocardium region (label 2 in segmentation)
            roi_mask = seg_data == 2

            if roi_mask.ndim != 3 or raw_data.ndim < 3:
                failed_count += 1
                print(f"Image/segmentation is not 3D for patient {patient_id}. Skipping.")
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

            heart_roi = raw_data[crop_slices]
            cropped_mask = dilated_heart_mask[crop_slices]
            heart_roi = np.where(cropped_mask, heart_roi, 0)

            if should_resize_slices:
                assert target_width is not None and target_height is not None
                heart_roi = _resize_volume_slices(heart_roi, target_width=target_width, target_height=target_height, order=1)

            # Save the extracted heart ROI as a new NIfTI file
            roi_img = nib.Nifti1Image(heart_roi, affine=raw_img.affine, header=raw_img.header)
            output_path = images_output_dir / f"{patient_id}_0000.nii.gz"
            nib.save(roi_img, output_path)
            processed_count += 1
            print(
                f"Saved heart ROI for patient ({processed_count}/{total_files}) {patient_id} to {output_path} "
                f"with bbox min={tuple(bbox_min.tolist())}, max={tuple((bbox_max - 1).tolist())}"
            )

            # Find and process the associated label file with the same bounding-box crop.
            label_file = labels_dir.joinpath(f"{patient_id}.nii.gz")

            if not label_file.is_file():
                label_missing_count += 1
                print(f"Associated label file not found for patient {patient_id}.")
                continue

            try:
                label_img = nib.load(label_file)
                # Cast labels to integer segmentation dtype to avoid float truncation warnings on load.
                label_data = label_img.get_fdata().round().astype("int16")

                if label_data.ndim < 3:
                    label_failed_count += 1
                    print(f"Label for patient {patient_id} is not 3D, cannot apply bbox cropping.")
                    continue
                if any(label_data.shape[dim] < int(bbox_max[dim]) for dim in range(3)):
                    label_failed_count += 1
                    print(f"Label/image bounding-box mismatch for patient {patient_id}.")
                    continue

                label_data = label_data[crop_slices]
                if should_resize_slices:
                    assert target_width is not None and target_height is not None
                    label_data = _resize_volume_slices(
                        label_data, target_width=target_width, target_height=target_height, order=0
                    )
                    label_data = label_data.astype("int16")

                label_output_path = labels_output_dir / f"{patient_id}.nii.gz"
                trimmed_label_img = nib.Nifti1Image(label_data, affine=label_img.affine, header=label_img.header)
                nib.save(trimmed_label_img, label_output_path)
                label_saved_count += 1
                print(f"Saved trimmed label for patient {patient_id} to {label_output_path} (original shape: {label_img.shape}, trimmed shape: {label_data.shape})")
            except Exception as label_exc:
                label_failed_count += 1
                print(f"Failed processing label for patient {patient_id}: {label_exc}")
        except Exception as exc:
            failed_count += 1
            print(f"Failed processing patient {patient_id}: {exc}")

    print(
        "Summary: "
        f"total={total_files}, processed={processed_count}, "
        f"missing_seg={missing_seg_count}, failed={failed_count}, "
        f"labels_saved={label_saved_count}, labels_missing={label_missing_count}, labels_failed={label_failed_count}"
    )
    

if __name__ == "__main__":
    raw_data_dir="/gpu-data3/nikos/datasets/cardiac_mri_ae/original/all/"
    labels_dir="/gpu-data3/nikos/datasets/cardiac_mri_ae/original/gt/"
    segmentation_dir="/gpu-data3/nikos/nnUnet_experiments/nnUNet_results/Dataset141_ACDC+MNMs/predict/postprocessed/"
    images_output_dir="/gpu-data3/nikos/datasets/cardiac_mri_ae/myocardium/all/"
    labels_output_dir="/gpu-data3/nikos/datasets/cardiac_mri_ae/myocardium/gt/"
    
    extract_heart_roi_from_myocarditis_dataset(raw_data_dir=raw_data_dir,
                                               labels_dir=labels_dir,
                                               segmentation_dir=segmentation_dir,
                                               images_output_dir=images_output_dir,
                                               labels_output_dir=labels_output_dir,
                                               width=128,
                                               height=128)