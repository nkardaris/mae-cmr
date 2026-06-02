import csv
import os
from pathlib import Path
import argparse

import nibabel as nib

def read_csv(csv_file: str):
    patient_info = {}

    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        patient_index = headers.index("External code")
        ed_index = headers.index("ED")
        es_index = headers.index("ES")
        vendor_index = headers.index("Vendor")

        for row in reader:
            patient_info[row[patient_index]] = {
                "ed": int(row[ed_index]),
                "es": int(row[es_index]),
                "vendor": row[vendor_index],
            }

    return patient_info


def convert_mnms(src_data_folder: Path, csv_file_name: str, images_out_dir: Path, labels_out_dir: Path):
    images_out_dir.mkdir(parents=True, exist_ok=True)
    labels_out_dir.mkdir(parents=True, exist_ok=True)
    patients_train = [f for f in (src_data_folder / "Training" / "Labeled").iterdir() if f.is_dir()]
    patients_val = [f for f in (src_data_folder / "Validation").iterdir() if f.is_dir()]
    patients_test = [f for f in (src_data_folder / "Testing").iterdir() if f.is_dir()]

    patient_info = read_csv(str(src_data_folder / csv_file_name))

    save_cardiac_phases(patients_train, patient_info, images_out_dir, labels_out_dir)
    save_cardiac_phases(patients_test, patient_info, images_out_dir, labels_out_dir)
    save_cardiac_phases(patients_val, patient_info, images_out_dir, labels_out_dir)

    # There are non-orthonormal direction cosines in the test and validation data.
    # Not sure if the data should be fixed, or we should skip the problematic data.


def save_cardiac_phases(patients: list[Path], patient_info: dict[str, dict[str, int]], images_out_dir: Path, labels_out_dir: Path):
    for patient in patients:
        print(f"Processing patient: {patient.name}")

        image = nib.load(patient / f"{patient.name}_sa.nii.gz")
        ed_frame = patient_info[patient.name]["ed"]
        es_frame = patient_info[patient.name]["es"]

        save_extracted_nifti_slice(image, ed_frame=ed_frame, es_frame=es_frame, out_dir=images_out_dir, patient=patient)

        if labels_out_dir:
            label = nib.load(patient / f"{patient.name}_sa_gt.nii.gz")
            save_extracted_nifti_slice(label, ed_frame=ed_frame, es_frame=es_frame, out_dir=labels_out_dir, patient=patient)


def save_extracted_nifti_slice(image, ed_frame: int, es_frame: int, out_dir: Path, patient: Path):
    # Save only extracted diastole and systole slices from the 4D H x W x D x time volume.
    image_ed = nib.Nifti1Image(image.dataobj[..., ed_frame], image.affine)
    image_es = nib.Nifti1Image(image.dataobj[..., es_frame], image.affine)

    # Labels do not have modality identifiers. Labels always end with 'gt'.
    suffix = "_sa_gt.nii.gz" if image.get_filename().endswith("_gt.nii.gz") else "_sa.nii.gz"

    nib.save(image_ed, str(out_dir / f"{patient.name}_frame{ed_frame:02d}{suffix}"))
    nib.save(image_es, str(out_dir / f"{patient.name}_frame{es_frame:02d}{suffix}"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MNMs conversion utility helper.")
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        default="/gpu-data3/nikos/datasets/MnMs/",
        help="The downloaded MNMs dataset dir. Should contain a csv file, as well as Training, Validation and Testing "
        "folders.",
    )
    parser.add_argument(
        "-c",
        "--csv_file_name",
        type=str,
        default="211230_M&Ms_Dataset_information_diagnosis_opendataset.csv",
        help="The csv file containing the dataset information.",
    )
    parser.add_argument(
        "-o",
        "--images_output",
        type=str,
        default="/gpu-data3/nikos/datasets/ACDC_MnMs_all/original/images/",
        help="Output directory for images",
    )
    parser.add_argument(
        "-l",
        "--labels_output",
        type=str,
        default="/gpu-data3/nikos/datasets/ACDC_MnMs_all/original/labels/",
        help="Output directory for labels",
    )
    args = parser.parse_args()
    args.input_folder = Path(args.input_folder)
    args.images_output_folder = Path(args.images_output)
    args.labels_output_folder = Path(args.labels_output)


    print("Converting...")
    convert_mnms(args.input_folder, args.csv_file_name, args.images_output_folder, args.labels_output_folder)

    print("Done!")
    
    