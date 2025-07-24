#!/usr/bin/env python3
"""
Inference script to generate and save a grid of images: original, ground truth, and generated outputs.
"""
import argparse
import os
import yaml
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.utils as tvu
import datasets
from models import DenoisingDiffusion, DiffusiveRestoration

# Disable wandb logging
os.environ["WANDB_MODE"] = "disabled"

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Inference: generate image grid')
    parser.add_argument("--config", default='LOLv1.yml', type=str,
                        help="Path to the config file under configs/")
    parser.add_argument('--resume', default='ckpt/Transf_LSUI/model_latest.pth.tar', type=str,
                        help='Path to the diffusion model checkpoint')
    parser.add_argument("--sampling_timesteps", type=int, default=10,
                        help="Number of sampling steps for diffusion")
    parser.add_argument("--image_folder", default='samples', type=str,
                        help="Directory to save output grid image")
    parser.add_argument('--seed', default=None, type=int,
                        help='Random seed')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to include in grid')
    args = parser.parse_args()

    # load yaml config
    with open(os.path.join("configs", args.config), "r") as f:
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

    # seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # data loader
    print(f"=> using dataset '{config.data.val_dataset}' for inference")
    dataset = datasets.__dict__[config.data.type](config)
    train_loader, val_loader = dataset.get_loaders(parse_patches=False)
    # if no validation samples, fallback to training set
    if len(val_loader.dataset) == 0:
        print("No validation images found, using training set for inference")
        loader = train_loader
    else:
        loader = val_loader

    # model
    print("=> creating diffusion restoration model")
    diffusion = DenoisingDiffusion(args, config)
    model = DiffusiveRestoration(diffusion, args, config)

    # collect images (flatten batches so each tensor is CxHxW)
    images = []
    count = 0
    with torch.no_grad():
        for batch in loader:
            x_batch, _ = batch
            b, _, h, w = x_batch.shape
            for i in range(b):
                if count >= args.num_samples:
                    break
                sample = x_batch[i:i+1].to(device)
                x_cond = sample[:, :3, :, :]
                gt = sample[:, 3:, :, :]
                # pad to multiple of 32
                pad_h = int(32 * np.ceil(h / 32.0)) - h
                pad_w = int(32 * np.ceil(w / 32.0)) - w
                x_pad = F.pad(x_cond, (0, pad_w, 0, pad_h), 'reflect')
                out = model.diffusive_restoration(x_pad)
                out = out[:, :, :h, :w]
                # convert to CPU and squeeze batch dimension
                images.extend([x_cond.cpu().squeeze(0), gt.cpu().squeeze(0), out.cpu().squeeze(0)])
                count += 1
            if count >= args.num_samples:
                break

    # save grid
    os.makedirs(args.image_folder, exist_ok=True)
    grid = tvu.make_grid(images, nrow=3, padding=2)
    # Extract path after 'ckpt/' for naming
    resume_rel = args.resume.split('ckpt/', 1)[-1].split(".")[:-2][0] #.replace('/', '_')
    out_path = f'samples/grid_{resume_rel}.png'

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tvu.save_image(grid, out_path)
    print(f"Saved grid of {args.num_samples} samples to {out_path}")


if __name__ == '__main__':
    main()
