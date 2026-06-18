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

from visualize_nikos import prepare_model, load_volume, show_image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(REPO_ROOT)
import models_mae

def get_args_parser():
    parser = argparse.ArgumentParser('MAE CMR reconstruction experiments', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=128, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Dataset parameters
    parser.add_argument('--pretrain_root', default='/gpu-data3/nikos/code/mae-cmr/output_dir_cmr', type=str,
                        help='dataset path (healthy controls)')
    parser.add_argument('--myocarditis_path', default='/gpu-data3/nikos/datasets/cmr/heart_roi/myocarditis/', type=str,
                        help='dataset path (myocarditis patients)')
    parser.add_argument('--in_chans', default=1, type=int,
                        help='number of input channels')
    parser.add_argument('--zscore', action='store_true',
                        help='enable per-slice z-score normalization for NIfTI data')
    parser.add_argument('--no_zscore', action='store_false', dest='zscore',
                        help='disable per-slice z-score normalization for NIfTI data')
    parser.set_defaults(zscore=True)


    parser.add_argument('--output_dir', default='./output_dir_cmr',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    return parser



def run_one_image(img, model, mean=0.0, std=1.0, output_path=None):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = x.unsqueeze(dim=-1)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x, mask_ratio=0.75)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *model.in_chans)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0], mean, std, "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], mean, std, "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], mean, std, "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], mean, std, "reconstruction + visible")

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    
    plt.close()
    return loss
    

def main(args):
    # Process validation set
    # validation_mean_losses = []
    # for split_id in range(5):
    #     print(f'Processing split {split_id}...')
    #     chkpt_dir = os.path.join(args.pretrain_root, f'split_{split_id}', 'checkpoint-best.pth')
    #     model_mae = prepare_model(chkpt_dir, 'mae_vit_base_patch16')
    #     print('Model loaded.')

    #     # Read validation volume paths
    #     split_file = os.path.join(args.pretrain_root, f'split_{split_id}', 'split.json')
    #     with open(split_file, 'r') as f:
    #         split_data = json.load(f)
    #     val_volume_paths = split_data['val']
        
    #     for volume_path in val_volume_paths:
    #         # print(f'Processing volume: {volume_path}')
    #         vol, mean, std = load_volume(volume_path)

    #         plt.rcParams['figure.figsize'] = [5, 5]
    #         per_slice_losses = []
    #         for slice in range(vol.shape[2]):
    #             img = vol[:, :, slice]
    #             img = np.array(
    #                 Image.fromarray(img).resize((args.input_size, args.input_size), resample=Image.BICUBIC))
                
    #             output_path = os.path.join(args.output_dir, f'{Path(volume_path).stem}_slice_{slice}.jpg')
    #             loss = run_one_image(img, model_mae, mean, std, output_path=output_path)
    #             # print(f'MAE with pixel reconstruction: {loss.item():.4f}')
    #             per_slice_losses.append(loss.item())

    #         print(f'Average MAE with pixel reconstruction for volume {volume_path}: {np.mean(per_slice_losses):.4f}')
    #         validation_mean_losses.append(np.mean(per_slice_losses))

    # Process myocarditis patients
    myocarditis_mean_losses = []
    print(f'Processing myocarditis patients...')
    chkpt_dir = os.path.join(args.pretrain_root, 'split_0', 'checkpoint-best.pth')
    model_mae = prepare_model(chkpt_dir, 'mae_vit_base_patch16')
    print('Model loaded.')
    myocarditis_volume_paths = [os.path.join(args.myocarditis_path, f) for f in os.listdir(args.myocarditis_path) if f.endswith('.nii.gz')]
    for volume_path in myocarditis_volume_paths:
        print(f'Processing volume: {volume_path}')
        vol, mean, std = load_volume(volume_path)

        plt.rcParams['figure.figsize'] = [5, 5]
        per_slice_losses = []
        for slice in range(vol.shape[2]):
            img = vol[:, :, slice]
            img = np.array(
                Image.fromarray(img).resize((args.input_size, args.input_size), resample=Image.BICUBIC))
            
            output_path = os.path.join(args.output_dir, f'{Path(volume_path).stem}_slice_{slice}.jpg')
            loss = run_one_image(img, model_mae, mean, std, output_path=output_path)
            # print(f'MAE with pixel reconstruction: {loss.item():.4f}')
            per_slice_losses.append(loss.item())

        print(f'Average MAE with pixel reconstruction for volume {volume_path}: {np.mean(per_slice_losses):.4f}')
        myocarditis_mean_losses.append(np.mean(per_slice_losses))
        
    # print(f'Overall average MAE with pixel reconstruction across validation set: {np.mean(validation_mean_losses):.4f}')
    print(f'Overall average MAE with pixel reconstruction across myocarditis patients: {np.mean(myocarditis_mean_losses):.4f}')

        

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        args.output_dir = Path(args.output_dir).joinpath("reconstruction_comparison")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
