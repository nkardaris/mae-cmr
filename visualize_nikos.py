import sys
import os
import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(REPO_ROOT)
import models_mae

# define the utils

def show_image(image, mean, std, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3 or image.shape[2] == 1, "Image should have 3 channels or 1 channel."
    # im = image * std + mean
    
    img_min = image.min()
    img_max = image.max()
    if img_max > img_min:
        img = (image - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(image)
    
    if image.shape[2] == 3:
        plt.imshow(torch.clip(img * 255, 0, 255).int())
    else:
        plt.imshow(torch.clip(img * 255, 0, 255).int(), cmap='gray')
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)(in_chans=1, img_size=128)
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


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

def run_one_image(img, model, mean=0.0, std=1.0, slice_num=0):
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

    plt.savefig(f"slice_{slice_num}.jpg", dpi=150, bbox_inches="tight")
    plt.close()
    return loss
    

if __name__ == '__main__':

    chkpt_dir = '/gpu-data3/nikos/code/mae-cmr/output_dir_cmr/split_0/checkpoint-best.pth'
    model_mae = prepare_model(chkpt_dir, 'mae_vit_base_patch16')
    print('Model loaded.')
    
    
    torch.manual_seed(2)
    
    # load an image
    volume_path = '/gpu-data3/nikos/datasets/cmr/heart_roi/control/LATE_ENH_001_0000.nii.gz'
    vol, mean, std = load_volume(volume_path)


    plt.rcParams['figure.figsize'] = [5, 5]
    losses = []
    for slice in range(vol.shape[2]):
        img = vol[:, :, slice]
        img = np.array(
            Image.fromarray(img).resize((128, 128), resample=Image.BILINEAR))
        
        # img = img.astype(np.float32)
        # img_min = img.min()
        # img_max = img.max()
        # if img_max > img_min:
        #     img = (img - img_min) / (img_max - img_min)
        # else:
        #     img = np.zeros_like(img)

        # img = np.stack([img, img, img], axis=-1)
        # show_image(torch.tensor(img), mean, std, title=f'slice {slice}')
        # plt.savefig(f"slice_{slice}.jpg", dpi=150, bbox_inches="tight")
        # plt.close()
        
        loss = run_one_image(img, model_mae, mean, std, slice_num=slice)
        print(f'MAE with pixel reconstruction: {loss.item():.4f}')
        losses.append(loss.item())

    print(f'Average MAE with pixel reconstruction: {np.mean(losses):.4f}')