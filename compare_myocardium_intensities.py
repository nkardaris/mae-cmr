import sys
import os
import torch
import numpy as np
import argparse
from pathlib import Path
import json

import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib

# from visualize_nikos import prepare_model, show_image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(REPO_ROOT)




def main(args):
    # Process control set
     
    
    # Process myocarditis set


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


def load_volume(path):

    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim > 3:
        data = data[..., 0]

    
    mean = float(data.mean())
    std = float(data.std())
    if std < 1e-6:
        std = 1.0
    data = (data - mean) / std

    return data, mean, std


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        args.output_dir = Path(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
