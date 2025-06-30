import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from models import DenoisingDiffusion
import datasets
import utils

def dict2namespace(config):
    import argparse
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def load_config(config_path):
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return dict2namespace(config)

def save_grid(org_imgs, gt_imgs, gen_imgs, n, out_path):
    import imageio
    import os
    os.makedirs(out_path, exist_ok=True)
    rows = []
    for i in range(n):
        row = np.concatenate([
            np.clip(org_imgs[i], 0, 1),
            np.clip(gt_imgs[i], 0, 1),
            np.clip(gen_imgs[i], 0, 1)
        ], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)  # Stack rows vertically
    grid = (grid * 255).astype(np.uint8)
    imageio.imwrite(os.path.join(out_path, 'samples_grid_test.png'), grid)

def main():
    parser = argparse.ArgumentParser(description="Inference and visualization for DenoisingDiffusion")
    parser.add_argument("--config", type=str, default="configs/LOLv1.yml", help="Path to config YAML file")
    parser.add_argument("--ckpt", type=str, default="ckpt/234.pth.tar", help="Path to checkpoint file")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device

    # Ensure args has sampling_timesteps for model
    if not hasattr(args, 'sampling_timesteps'):
        args.sampling_timesteps = getattr(config, 'sampling_timesteps', 200)

    # Load dataset
    DATASET = datasets.__dict__[config.data.type](config)
    _, val_loader = DATASET.get_loaders()

    # Load model
    diffusion = DenoisingDiffusion(args, config)
    ckpt_path = args.ckpt
    if ckpt_path is None:
        # Try to find latest checkpoint
        ckpt_dir = getattr(config.data, 'ckpt_dir', 'checkpoints')
        if not os.path.exists(ckpt_dir):
            raise FileNotFoundError("No checkpoint directory found.")
        ckpts = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.pth') or f.endswith('.pt') or f.isdigit()]
        if not ckpts:
            raise FileNotFoundError("No checkpoint found in directory.")
        ckpt_path = max(ckpts, key=os.path.getctime)
        print(f"Using latest checkpoint: {ckpt_path}")
    diffusion.load_ddm_ckpt(ckpt_path, ema=True)

    # Randomly select indices for sampling
    val_dataset = val_loader.dataset
    total = len(val_dataset)
    np.random.seed()  # Use system time or OS entropy
    sample_indices = np.random.choice(total, size=args.num_samples, replace=False)

    org_imgs, gt_imgs, gen_imgs = [], [], []
    for idx in sample_indices:
        x, y = val_dataset[idx]
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            out = diffusion.model(x)
            pred_x = out["pred_x"].cpu().numpy()[0]
        org = x[0, :3, :, :].cpu().numpy()
        gt = x[0, 3:, :, :].cpu().numpy()
        org_imgs.append(np.transpose(org, (1,2,0)))
        gt_imgs.append(np.transpose(gt, (1,2,0)))
        gen_imgs.append(np.transpose(pred_x, (1,2,0)))

    # make directory if not exist
    os.makedirs("./samples/", exist_ok=True)
    save_grid(org_imgs, gt_imgs, gen_imgs, args.num_samples, out_path="./samples/")

if __name__ == "__main__":
    main()
