# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import random
from collections import OrderedDict
from pathlib import Path

import PIL
import nibabel as nib
import numpy as np
import torch
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


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
    ):
        self.data_dir = Path(data_dir)
        self.is_train = is_train
        self.input_size = int(input_size)
        self.train_ratio = float(train_ratio)
        self.split_seed = int(split_seed)
        self.zscore = bool(zscore)
        self.rotation_deg = float(rotation_deg)
        self.hflip = bool(hflip)
        self.volume_cache_size = int(volume_cache_size)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"NIfTI directory not found: {self.data_dir}")

        volume_paths = list(self.data_dir.rglob("*.nii")) + list(self.data_dir.rglob("*.nii.gz"))
        volume_paths = sorted(set(volume_paths))
        if not volume_paths:
            raise FileNotFoundError(f"No NIfTI files found under: {self.data_dir}")

        rng = random.Random(self.split_seed)
        rng.shuffle(volume_paths)
        split_index = int(len(volume_paths) * self.train_ratio)
        if split_index == 0 or split_index == len(volume_paths):
            raise ValueError("Train/val split would be empty; adjust train_ratio or dataset size.")

        if self.is_train:
            self.volume_paths = volume_paths[:split_index]
        else:
            self.volume_paths = volume_paths[split_index:]

        self.slice_mappings = []
        for volume_path in self.volume_paths:
            volume_shape = self._get_volume_shape(volume_path)
            depth = volume_shape[2]
            for slice_idx in range(depth):
                self.slice_mappings.append((volume_path, slice_idx))

        self._volume_cache = OrderedDict()

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

    def __len__(self):
        return len(self.slice_mappings)

    def __getitem__(self, index):
        volume_path, slice_idx = self.slice_mappings[index]
        img = self._load_volume(volume_path)
        data = img.dataobj

        if data.ndim == 4:
            slice_2d = np.asarray(data[:, :, slice_idx, 0])
        else:
            slice_2d = np.asarray(data[:, :, slice_idx])

        slice_2d = slice_2d.astype(np.float32)
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

        if self.is_train:
            if self.hflip and random.random() < 0.5:
                slice_tensor = TF.hflip(slice_tensor)
            if self.rotation_deg > 0:
                angle = random.uniform(-self.rotation_deg, self.rotation_deg)
                slice_tensor = TF.rotate(
                    slice_tensor,
                    angle=angle,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0.0,
                )

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
