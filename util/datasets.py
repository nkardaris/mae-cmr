# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import re
import random
from collections import OrderedDict
from pathlib import Path

import PIL
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# nnUNet (Dataset141_ACDC+MNMs) label scheme; myocardium is the structure we mask.
MYOCARDIUM_LABEL = 2


class NiftiSliceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        is_train,
        input_size,
        train_ratio=0.8,
        split_seed=0,
        zscore=True,
        rotation_deg=0.0,
        hflip=False,
        volume_cache_size=0,
        translate_frac=0.0,
        scale_range=(1.0, 1.0),
        elastic_alpha=0.0,
        elastic_sigma=0.0,
        elastic_prob=0.0,
        split_id = -1,
        seg_dir=None,
        patch_size=16,
    ):
        self.data_dir = Path(data_dir)
        self.seg_dir = Path(seg_dir) if seg_dir is not None else None
        self.patch_size = int(patch_size)
        if self.seg_dir is not None and int(input_size) % self.patch_size != 0:
            raise ValueError("input_size must be divisible by patch_size for structural masking")
        self.is_train = is_train
        self.input_size = int(input_size)
        self.train_ratio = float(train_ratio)
        self.split_seed = int(split_seed) # Active only when split_id is -1 for random split; ignored otherwise
        self.split_id = int(split_id)
        if self.split_id < -1 or self.split_id > 4:
            raise ValueError("split_id must be between 0 and 4 (inclusive) or -1 for random split")
        self.zscore = bool(zscore)
        self.rotation_deg = float(rotation_deg)
        self.hflip = bool(hflip)
        self.volume_cache_size = int(volume_cache_size)
        self.translate_frac = float(translate_frac)
        if isinstance(scale_range, (list, tuple)) and len(scale_range) == 2:
            self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        else:
            raise ValueError("scale_range must be a 2-element list/tuple")
        if self.scale_range[0] <= 0 or self.scale_range[1] <= 0:
            raise ValueError("scale_range values must be > 0")
        if self.scale_range[0] > self.scale_range[1]:
            raise ValueError("scale_range min must be <= max")
        self.elastic_alpha = float(elastic_alpha)
        self.elastic_sigma = float(elastic_sigma)
        self.elastic_prob = float(elastic_prob)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"NIfTI directory not found: {self.data_dir}")

        volume_paths = list(self.data_dir.rglob("*.nii")) + list(self.data_dir.rglob("*.nii.gz"))
        volume_paths = sorted(set(volume_paths))
        if not volume_paths:
            raise FileNotFoundError(f"No NIfTI files found under: {self.data_dir}")

        if self.split_id == -1: # Random split based on seed
            rng = random.Random(self.split_seed)
            rng.shuffle(volume_paths)
            split_index = int(len(volume_paths) * self.train_ratio)
            if split_index == 0 or split_index == len(volume_paths):
                raise ValueError("Train/val split would be empty; adjust train_ratio or dataset size.")

            if self.is_train:
                self.volume_paths = volume_paths[:split_index]
            else:
                self.volume_paths = volume_paths[split_index:]
        else: # Predefined split into 5 folds. split_id determines which fold is used for validation (0-4), the rest for training
            split_size = len(volume_paths) // 5
            
            if self.is_train:
                self.volume_paths = volume_paths[:self.split_id*split_size] + volume_paths[(self.split_id+1)*split_size:]
            else:
                self.volume_paths = volume_paths[self.split_id*split_size:(self.split_id+1)*split_size]
                

        # Resolve the paired segmentation mask for each volume (structural masking).
        self._seg_paths = {}
        if self.seg_dir is not None:
            if not self.seg_dir.exists():
                raise FileNotFoundError(f"Segmentation directory not found: {self.seg_dir}")
            missing = []
            for volume_path in self.volume_paths:
                seg_path = self._find_seg_path(volume_path)
                if seg_path is None:
                    missing.append(volume_path.name)
                else:
                    self._seg_paths[volume_path] = seg_path
            if missing:
                raise FileNotFoundError(
                    f"No segmentation found in {self.seg_dir} for {len(missing)} volume(s), "
                    f"e.g. {missing[:5]}"
                )

        self.slice_mappings = []
        for volume_path in self.volume_paths:
            volume_shape = self._get_volume_shape(volume_path)
            depth = volume_shape[2]
            for slice_idx in range(depth):
                self.slice_mappings.append((volume_path, slice_idx))

        self._volume_cache = OrderedDict()
        self._elastic_kernel_cache = {}

    def _find_seg_path(self, volume_path):
        """Locate the segmentation mask for an image volume.

        Handles the two naming conventions in this project: the CMR/LGE images
        carry an nnUNet channel suffix (``LATE_ENH_001_0000`` -> ``LATE_ENH_001``)
        while ACDC+M&Ms labels add a ``_gt`` suffix (``patient001`` -> ``patient001_gt``).
        """
        name = volume_path.name
        if name.endswith(".nii.gz"):
            stem, ext = name[:-7], ".nii.gz"
        elif name.endswith(".nii"):
            stem, ext = name[:-4], ".nii"
        else:
            return None
        base = re.sub(r"_\d{4}$", "", stem)
        candidates = [f"{base}{ext}", f"{stem}_gt{ext}", f"{stem}{ext}"]
        for cand in candidates:
            cand_path = self.seg_dir / cand
            if cand_path.is_file():
                return cand_path
        return None

    @staticmethod
    def _extract_slice(dataobj, slice_idx):
        if dataobj.ndim == 4:
            return np.asarray(dataobj[:, :, slice_idx, 0])
        return np.asarray(dataobj[:, :, slice_idx])

    def _patch_myocardium(self, mask_tensor):
        """Reduce a (1, H, W) label slice to an (L,) per-patch myocardium indicator.

        A patch is flagged (1.0) if it contains any myocardium voxel; ordering is
        row-major (h outer, w inner) to match ``MaskedAutoencoderViT.patchify``.
        """
        p = self.patch_size
        myo = (mask_tensor[0] == MYOCARDIUM_LABEL).to(torch.float32)
        myo = myo.unfold(0, p, p).unfold(1, p, p)  # (h, w, p, p)
        patch = (myo.sum(dim=(-1, -2)) > 0).to(torch.float32)  # (h, w)
        return patch.flatten()

    def _get_volume_shape(self, volume_path):
        img = nib.load(str(volume_path))
        shape = img.shape
        if len(shape) == 4:
            shape = shape[:3]
        if len(shape) != 3:
            raise ValueError(f"Expected 3D or 4D NIfTI, got shape {img.shape} for {volume_path}")
        return shape

    def _load_volume(self, volume_path):
        if self.volume_cache_size <= 0:
            return nib.load(str(volume_path))

        cached = self._volume_cache.get(volume_path)
        if cached is not None:
            self._volume_cache.move_to_end(volume_path)
            return cached

        img = nib.load(str(volume_path))
        self._volume_cache[volume_path] = img
        if len(self._volume_cache) > self.volume_cache_size:
            self._volume_cache.popitem(last=False)
        return img

    def _get_gaussian_kernel(self, sigma, device, dtype):
        if sigma <= 0:
            return None
        key = (float(sigma), device, dtype)
        cached = self._elastic_kernel_cache.get(key)
        if cached is not None:
            return cached
        radius = max(1, int(round(3 * sigma)))
        kernel_size = radius * 2 + 1
        coords = torch.arange(kernel_size, device=device, dtype=dtype) - radius
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel = kernel_2d.view(1, 1, kernel_size, kernel_size)
        self._elastic_kernel_cache[key] = kernel
        return kernel

    def _elastic_deform_pair(self, slice_tensor, mask_tensor):
        """Apply the same elastic deformation to the image (and optional mask).

        The displacement grid is sampled once and shared so the label slice stays
        registered to the image; the image is sampled bilinearly, the mask nearest.
        """
        if self.elastic_alpha <= 0 or self.elastic_sigma <= 0 or self.elastic_prob <= 0:
            return slice_tensor, mask_tensor
        if random.random() >= self.elastic_prob:
            return slice_tensor, mask_tensor

        device = slice_tensor.device
        dtype = slice_tensor.dtype
        _, height, width = slice_tensor.shape
        dx = torch.rand(1, 1, height, width, device=device, dtype=dtype) * 2 - 1
        dy = torch.rand(1, 1, height, width, device=device, dtype=dtype) * 2 - 1
        kernel = self._get_gaussian_kernel(self.elastic_sigma, device, dtype)
        if kernel is not None:
            padding = kernel.shape[-1] // 2
            dx = F.conv2d(dx, kernel, padding=padding)
            dy = F.conv2d(dy, kernel, padding=padding)
        dx = dx * self.elastic_alpha
        dy = dy * self.elastic_alpha

        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing='ij',
        )
        grid_x = grid_x + dx[0, 0]
        grid_y = grid_y + dy[0, 0]

        if width > 1:
            grid_x = 2.0 * grid_x / (width - 1) - 1.0
        else:
            grid_x = torch.zeros_like(grid_x)
        if height > 1:
            grid_y = 2.0 * grid_y / (height - 1) - 1.0
        else:
            grid_y = torch.zeros_like(grid_y)

        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        warped = F.grid_sample(
            slice_tensor.unsqueeze(0),
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )
        slice_tensor = warped.squeeze(0)
        if mask_tensor is not None:
            warped_mask = F.grid_sample(
                mask_tensor.unsqueeze(0),
                grid,
                mode='nearest',
                padding_mode='zeros',
                align_corners=True,
            )
            mask_tensor = warped_mask.squeeze(0)
        return slice_tensor, mask_tensor

    def __len__(self):
        return len(self.slice_mappings)

    def __getitem__(self, index):
        volume_path, slice_idx = self.slice_mappings[index]
        img = self._load_volume(volume_path)

        slice_2d = self._extract_slice(img.dataobj, slice_idx).astype(np.float32)
        if self.zscore:
            mean = float(slice_2d.mean())
            std = float(slice_2d.std())
            if std < 1e-6:
                std = 1.0
            slice_2d = (slice_2d - mean) / std

        slice_tensor = torch.from_numpy(slice_2d).unsqueeze(0)
        slice_tensor = TF.resize(
            slice_tensor,
            [self.input_size, self.input_size],
            interpolation=InterpolationMode.BILINEAR,
        )

        mask_tensor = None
        if self.seg_dir is not None:
            seg = self._load_volume(self._seg_paths[volume_path])
            seg_2d = self._extract_slice(seg.dataobj, slice_idx).astype(np.float32)
            mask_tensor = torch.from_numpy(seg_2d).unsqueeze(0)
            mask_tensor = TF.resize(
                mask_tensor,
                [self.input_size, self.input_size],
                interpolation=InterpolationMode.NEAREST,
            )

        if self.is_train:
            if self.hflip and random.random() < 0.5:
                slice_tensor = TF.hflip(slice_tensor)
                if mask_tensor is not None:
                    mask_tensor = TF.hflip(mask_tensor)
            apply_affine = (
                self.rotation_deg > 0
                or self.translate_frac > 0
                or self.scale_range != (1.0, 1.0)
            )
            if apply_affine:
                angle = random.uniform(-self.rotation_deg, self.rotation_deg) if self.rotation_deg > 0 else 0.0
                max_translate = self.translate_frac * self.input_size
                if max_translate > 0:
                    translate_x = random.uniform(-max_translate, max_translate)
                    translate_y = random.uniform(-max_translate, max_translate)
                else:
                    translate_x = 0.0
                    translate_y = 0.0
                if self.scale_range != (1.0, 1.0):
                    scale = random.uniform(self.scale_range[0], self.scale_range[1])
                else:
                    scale = 1.0
                affine_kwargs = dict(
                    angle=angle,
                    translate=[translate_x, translate_y],
                    scale=scale,
                    shear=[0.0, 0.0],
                    fill=0.0,
                )
                slice_tensor = TF.affine(
                    slice_tensor, interpolation=InterpolationMode.BILINEAR, **affine_kwargs,
                )
                if mask_tensor is not None:
                    mask_tensor = TF.affine(
                        mask_tensor, interpolation=InterpolationMode.NEAREST, **affine_kwargs,
                    )
            slice_tensor, mask_tensor = self._elastic_deform_pair(slice_tensor, mask_tensor)

        if mask_tensor is not None:
            return slice_tensor, self._patch_myocardium(mask_tensor)
        return slice_tensor, 0


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
