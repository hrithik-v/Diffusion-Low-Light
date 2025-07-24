#!/usr/bin/env python3
"""
Validation script to compute average PSNR and SSIM over the validation dataset.
"""
import argparse
import os
import yaml
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

import datasets
from models import DenoisingDiffusion, DiffusiveRestoration
from utils.metrics import compute_psnr, compute_ssim

# Disable wandb logging
os.environ["WANDB_MODE"] = "disabled"

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Validation: compute PSNR and SSIM')
    parser.add_argument("--config", default='LOLv1.yml', type=str,
                        help="Path to the config file under configs/")
    parser.add_argument("--resume", required=True, type=str,
                        help="Path to the diffusion model checkpoint")
    parser.add_argument("--sampling_timesteps", type=int, default=10,
                        help="Number of sampling steps for diffusion")
    parser.add_argument('--seed', default=230, type=int,
                        help='Random seed')
    args = parser.parse_args()

    # load yaml config
    cfg_path = os.path.join("configs", args.config)
    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f)
    config = dict2namespace(raw)
    return args, config


def dict2namespace(cfg):
    ns = argparse.Namespace()
    for k, v in cfg.items():
        if isinstance(v, dict):
            setattr(ns, k, dict2namespace(v))
        else:
            setattr(ns, k, v)
    return ns


def main():
    args, config = parse_args_and_config()

    # device setup
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    config.device = device

    # seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # data loader
    print(f"=> using dataset '{config.data.val_dataset}' for validation")
    dataset = datasets.__dict__[config.data.type](config)
    train_loader, val_loader = dataset.get_loaders(parse_patches=False)
    # fallback to training set if no validation samples
    if len(val_loader.dataset) == 0:
        print("No validation images found, using training set for validation")
        loader = train_loader
    else:
        loader = val_loader

    # model setup
    diffusion = DenoisingDiffusion(args, config)
    model = DiffusiveRestoration(diffusion, args, config)
    model.diffusion.model.eval()

    psnr_vals = []
    ssim_vals = []

    with torch.no_grad():
        for x_batch, _ in tqdm(loader, desc="Validation"):  
            x_cond = x_batch[:, :3, :, :].to(device)
            gt = x_batch[:, 3:, :, :].to(device)
            b, _, h, w = x_cond.shape
            # pad to multiple of 32
            pad_h = int(32 * np.ceil(h / 32.0)) - h
            pad_w = int(32 * np.ceil(w / 32.0)) - w
            x_pad = F.pad(x_cond, (0, pad_w, 0, pad_h), 'reflect')

            # inference
            out = model.diffusive_restoration(x_pad)
            out = out[:, :, :h, :w]

            # crop 20% from each border (left, right, top, bottom) to avoid border artifacts
            # crop_w_start = int(0.3 * w)
            # crop_w_end = int(0.7 * w)
            # crop_h_start = int(0.3 * h)
            # crop_h_end = int(0.7 * h)
            # out_cropped = out[:, :, crop_h_start:crop_h_end, crop_w_start:crop_w_end]
            # gt_cropped = gt[:, :, crop_h_start:crop_h_end, crop_w_start:crop_w_end]
            out_cropped = out
            gt_cropped = gt
            # compute metrics on cropped images
            psnr = compute_psnr(out_cropped, gt_cropped)
            ssim = compute_ssim(out_cropped, gt_cropped)
            psnr_vals.append(psnr)
            ssim_vals.append(ssim)

    avg_psnr = np.mean(psnr_vals)
    avg_ssim = np.mean(ssim_vals)
    print(f"Average PSNR over validation set: {avg_psnr:.4f}")
    print(f"Average SSIM over validation set: {avg_ssim:.4f}")


if __name__ == '__main__':
    main()
