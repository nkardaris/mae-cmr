import argparse
from pathlib import Path
from typing import cast, Optional, Tuple

import nibabel as nib
import numpy as np
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


def _split_nii_name(filename: str) -> Optional[Tuple[str, str]]:
    if filename.endswith(".nii.gz"):
        return filename[:-7], ".nii.gz"
    if filename.endswith(".nii"):
        return filename[:-4], ".nii"
    return None


def _find_label_path(labels_dir: Path, stem: str) -> Optional[Path]:
    candidates = [
        labels_dir / f"{stem}_gt.nii.gz",
        labels_dir / f"{stem}_gt.nii",
    ]
    return next((path for path in candidates if path.is_file()), None)


def extract_heart_roi_acdc_mnms(
    input_images_dir: str,
    input_labels_dir: str,
    output_images_dir: str,
    output_labels_dir: str,
    width: int,
    height: int,
) -> None:
    images_dir = Path(input_images_dir)
    labels_dir = Path(input_labels_dir)
    out_images_dir = Path(output_images_dir)
    out_labels_dir = Path(output_labels_dir)
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive integers, got width={width}, height={height}")

    image_files = sorted(images_dir.glob("*.nii*"))
    total_files = len(image_files)
    processed_count = 0
    missing_label_count = 0
    empty_slice_count = 0
    failed_count = 0

    for image_path in image_files:
        name_parts = _split_nii_name(image_path.name)
        if name_parts is None:
            print(f"Unsupported file extension for {image_path.name}. Skipping.")
            continue

        stem, _ = name_parts
        label_path = _find_label_path(labels_dir, stem)
        if label_path is None:
            missing_label_count += 1
            print(f"Label file not found for {image_path.name}. Skipping.")
            continue

        try:
            image_obj = nib.load(image_path)
            label_obj = nib.load(label_path)

            image_data = image_obj.get_fdata()
            label_data = label_obj.get_fdata().round().astype("int16")

            if image_data.ndim != 3 or label_data.ndim != 3:
                failed_count += 1
                print(f"Image/label is not 3D for {image_path.name}. Skipping.")
                continue
            if image_data.shape != label_data.shape:
                failed_count += 1
                print(f"Image/label shape mismatch for {image_path.name}. Skipping.")
                continue

            heart_mask = label_data > 0
            structure = np.ones((3, 3), dtype=bool)
            dilated_slices = [
                binary_dilation(heart_mask[:, :, z], structure=structure)
                for z in range(heart_mask.shape[2])
            ]
            dilated_mask = np.stack(dilated_slices, axis=2)

            foreground_coords = np.argwhere(dilated_mask)
            if foreground_coords.size == 0:
                failed_count += 1
                print(f"Dilated mask is empty for {image_path.name}. Skipping.")
                continue

            bbox_min = foreground_coords.min(axis=0)
            bbox_max = foreground_coords.max(axis=0) + 1
            crop_slices = tuple(slice(int(start), int(stop)) for start, stop in zip(bbox_min, bbox_max))

            if any(image_data.shape[dim] < int(bbox_max[dim]) for dim in range(3)):
                failed_count += 1
                print(f"Bounding box exceeds image shape for {image_path.name}. Skipping.")
                continue

            cropped_image = image_data[crop_slices]
            cropped_label = label_data[crop_slices]
            cropped_dilated_mask = dilated_mask[crop_slices]
            cropped_image = np.where(cropped_dilated_mask, cropped_image, 0)

            keep_indices = []
            for z in range(cropped_label.shape[2]):
                label_slice = cropped_label[:, :, z]
                non_zero_labels = np.unique(label_slice)
                non_zero_labels = non_zero_labels[non_zero_labels != 0]
                if non_zero_labels.size >= 2:
                    keep_indices.append(z)

            if not keep_indices:
                empty_slice_count += 1
                print(f"No slices with >=2 heart regions for {image_path.name}. Skipping.")
                continue

            cropped_image = cropped_image[:, :, keep_indices]
            cropped_label = cropped_label[:, :, keep_indices]

            cropped_image = _resize_volume_slices(
                cropped_image, target_width=width, target_height=height, order=1
            )
            cropped_label = _resize_volume_slices(
                cropped_label, target_width=width, target_height=height, order=0
            ).astype("int16")

            output_image_path = out_images_dir / image_path.name
            output_label_path = out_labels_dir / label_path.name

            nib.save(
                nib.Nifti1Image(cropped_image, affine=image_obj.affine, header=image_obj.header),
                output_image_path,
            )
            nib.save(
                nib.Nifti1Image(cropped_label, affine=label_obj.affine, header=label_obj.header),
                output_label_path,
            )
            processed_count += 1
            print(
                f"Saved ROI for {image_path.name} -> {output_image_path.name} "
                f"(kept slices={len(keep_indices)}/{dilated_mask.shape[2]})"
            )
        except Exception as exc:
            failed_count += 1
            print(f"Failed processing {image_path.name}: {exc}")

    print(
        "Summary: "
        f"total={total_files}, processed={processed_count}, missing_labels={missing_label_count}, "
        f"empty_slices={empty_slice_count}, failed={failed_count}"
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crop ACDC+MnMs volumes around heart ROI and resize slices.")
    parser.add_argument(
        "--input_images_dir",
        default="/gpu-data3/nikos/datasets/ACDC_MnMs_all/original/images/",
        help="Directory with input image volumes.",
    )
    parser.add_argument(
        "--input_labels_dir",
        default="/gpu-data3/nikos/datasets/ACDC_MnMs_all/original/labels/",
        help="Directory with input label volumes.",
    )
    parser.add_argument(
        "--output_images_dir",
        default="/gpu-data3/nikos/datasets/ACDC_MnMs_all/heart_roi/images/",
        help="Directory to write cropped image volumes.",
    )
    parser.add_argument(
        "--output_labels_dir",
        default="/gpu-data3/nikos/datasets/ACDC_MnMs_all/heart_roi/labels/",
        help="Directory to write cropped label volumes.",
    )
    parser.add_argument("--width", type=int, default=128, help="Target slice width.")
    parser.add_argument("--height", type=int, default=128, help="Target slice height.")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    extract_heart_roi_acdc_mnms(
        input_images_dir=args.input_images_dir,
        input_labels_dir=args.input_labels_dir,
        output_images_dir=args.output_images_dir,
        output_labels_dir=args.output_labels_dir,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
