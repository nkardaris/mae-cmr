# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch, evaluate_one_epoch
from util.datasets import NiftiSliceDataset


def get_args_parser():
    parser = argparse.ArgumentParser('MAE CMR pre-training', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

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

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--dataset', default='nifti', type=str, choices=['nifti', 'imagenet'],
                        help='dataset type')
    parser.add_argument('--data_path', default='/gpu-data3/nikos/datasets/cmr/heart_roi/control/', type=str,
                        help='dataset path (healthy controls)')
    parser.add_argument('--train_ratio', default=0.8, type=float,
                        help='train/val split ratio for NIfTI volumes')
    parser.add_argument('--split_id', default=-1, type=int,
                        help='split ID for predefined 5-fold split (-1 for random split)')
    parser.add_argument('--split_seed', default=0, type=int,
                        help='random seed for the NIfTI volume split (active only if split_id is -1)')
    parser.add_argument('--in_chans', default=1, type=int,
                        help='number of input channels')
    parser.add_argument('--seg_dir', default=None, type=str,
                        help='directory with paired segmentation masks; enables structural '
                             '(myocardium-only) masking instead of random masking')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='patch size used to build the per-patch myocardium indicator '
                             '(must match the model patch size)')
    parser.add_argument('--rotation_deg', default=10.0, type=float,
                        help='max rotation degrees for NIfTI training augmentation')
    parser.add_argument('--translate_frac', default=0.05, type=float,
                        help='max translation as a fraction of image size for NIfTI training augmentation')
    parser.add_argument('--scale_range', default=(0.9, 1.1), nargs=2, type=float, metavar=('MIN', 'MAX'),
                        help='min/max scale for NIfTI training augmentation')
    parser.add_argument('--elastic_alpha', default=20.0, type=float,
                        help='elastic deformation alpha in pixels')
    parser.add_argument('--elastic_sigma', default=5.0, type=float,
                        help='elastic deformation sigma in pixels')
    parser.add_argument('--elastic_prob', default=0.3, type=float,
                        help='probability of applying elastic deformation')
    parser.add_argument('--volume_cache_size', default=0, type=int,
                        help='number of NIfTI volumes to cache per worker (0 disables caching)')
    parser.add_argument('--hflip', action='store_true',
                        help='enable random horizontal flip for NIfTI training')
    parser.add_argument('--no_hflip', action='store_false', dest='hflip',
                        help='disable random horizontal flip for NIfTI training')
    parser.set_defaults(hflip=True)
    parser.add_argument('--zscore', action='store_true',
                        help='enable per-slice z-score normalization for NIfTI data')
    parser.add_argument('--no_zscore', action='store_false', dest='zscore',
                        help='disable per-slice z-score normalization for NIfTI data')
    parser.set_defaults(zscore=True)

    parser.add_argument('--early_stop_patience', default=20, type=int,
                        help='stop after this many epochs without val loss improvement (0 disables)')
    parser.add_argument('--early_stop_min_delta', default=1e-4, type=float,
                        help='minimum val loss improvement to reset early stopping')

    parser.add_argument('--output_dir', default='./output_dir_cmr',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir_cmr',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint (continues optimizer state and epoch)')
    parser.add_argument('--init_from', default=None, type=str,
                        help='initialize model weights only from this checkpoint (fresh optimizer/'
                             'schedule); use for stage B (LGE) after stage A (cine) pre-training')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def save_best_checkpoint(args, model_without_ddp, optimizer, loss_scaler, epoch):
    if not args.output_dir:
        return
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / 'checkpoint-best.pth'
    to_save = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': args,
    }
    if loss_scaler is not None:
        to_save['scaler'] = loss_scaler.state_dict()
    misc.save_on_master(to_save, checkpoint_path)


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # structural (myocardium-only) masking is enabled when a segmentation dir is given
    args.structural_masking = args.seg_dir is not None

    if args.dataset == 'nifti':
        dataset_train = NiftiSliceDataset(
            data_dir=args.data_path,
            is_train=True,
            input_size=args.input_size,
            train_ratio=args.train_ratio,
            split_id=args.split_id,
            split_seed=args.split_seed,
            zscore=args.zscore,
            rotation_deg=args.rotation_deg,
            hflip=args.hflip,
            volume_cache_size=args.volume_cache_size,
            translate_frac=args.translate_frac,
            scale_range=args.scale_range,
            elastic_alpha=args.elastic_alpha,
            elastic_sigma=args.elastic_sigma,
            elastic_prob=args.elastic_prob,
            seg_dir=args.seg_dir,
            patch_size=args.patch_size,
        )
        dataset_val = NiftiSliceDataset(
            data_dir=args.data_path,
            is_train=False,
            input_size=args.input_size,
            train_ratio=args.train_ratio,
            split_id=args.split_id,
            split_seed=args.split_seed,
            zscore=args.zscore,
            rotation_deg=0.0,
            hflip=False,
            volume_cache_size=args.volume_cache_size,
            translate_frac=0.0,
            scale_range=(1.0, 1.0),
            elastic_alpha=0.0,
            elastic_sigma=0.0,
            elastic_prob=0.0,
            seg_dir=args.seg_dir,
            patch_size=args.patch_size,
        )
        print(f"NIfTI train slices: {len(dataset_train)}, val slices: {len(dataset_val)}")
        print(f"Structural (myocardium) masking: {args.structural_masking}")
    else:
        if args.in_chans != 3:
            print("Overriding in_chans to 3 for ImageNet training")
            args.in_chans = 3

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_val = transforms.Compose([
            transforms.Resize(int(args.input_size / (224 / 256)), interpolation=3),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
        dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_val)
        print(dataset_train)
        print(dataset_val)

    # Save the train/val split info for NIfTI dataset
    if args.output_dir and misc.is_main_process() and args.dataset == "nifti":
        split_path = Path(args.output_dir) / "split.json"
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "split_id": args.split_id,
                    "split_seed": args.split_seed,
                    "train": [str(p) for p in dataset_train.volume_paths],
                    "val": [str(p) for p in dataset_val.volume_paths],
                }, f, indent=2)


    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
        print("Sampler_train = %s" % str(sampler_train))
        print("Sampler_val = %s" % str(sampler_val))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # define the model
    model = models_mae.__dict__[args.model](
        norm_pix_loss=args.norm_pix_loss,
        img_size=args.input_size,
        in_chans=args.in_chans,
    )

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    # Stage B (LGE) init: load weights only from the stage-A (cine) checkpoint, keeping a
    # fresh optimizer and LR schedule (unlike --resume, which continues epoch/optimizer state).
    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location='cpu')
        msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print(f"Initialized weights from {args.init_from}: {msg}")

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    best_val_loss = float('inf')
    best_epoch = -1
    patience_counter = 0

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        val_stats = evaluate_one_epoch(
            model, data_loader_val,
            device, epoch,
            args=args
        )
        val_loss = float(val_stats['loss'])
        improved = val_loss < (best_val_loss - args.early_stop_min_delta)
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            save_best_checkpoint(args, model_without_ddp, optimizer, loss_scaler, epoch)
        else:
            patience_counter += 1

        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,
                     'best_val_loss': best_val_loss,
                     'patience_counter': patience_counter}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.add_scalar('val_loss', val_stats['loss'], epoch)
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best val loss {best_val_loss:.6f} at epoch {best_epoch}."
            )
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        args.output_dir = Path(args.output_dir).joinpath(f"split_{args.split_id}" if args.split_id >= 0 else f"split_random_seed_{args.split_seed}")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        args.log_dir = Path(args.log_dir).joinpath(f"split_{args.split_id}" if args.split_id >= 0 else f"split_random_seed_{args.split_seed}")
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    main(args)
