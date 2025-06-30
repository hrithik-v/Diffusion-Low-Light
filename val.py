import argparse
import os
import torch
import numpy as np
from models import DenoisingDiffusion
import datasets
import utils
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from tqdm import tqdm

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

def main():
    parser = argparse.ArgumentParser(description="Validation for DenoisingDiffusion")
    parser.add_argument("--config", type=str, default="configs/LOLv1.yml", help="Path to config YAML file")
    parser.add_argument("--ckpt", type=str, default="ckpt/234.pth.tar", help="Path to checkpoint file")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device

    # Ensure args has sampling_timesteps for model
    if not hasattr(args, 'sampling_timesteps'):
        args.sampling_timesteps = getattr(config, 'sampling_timesteps', 10)

    # Load dataset
    DATASET = datasets.__dict__[config.data.type](config)
    _, val_loader = DATASET.get_loaders()

    # Load model
    diffusion = DenoisingDiffusion(args, config)
    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_dir = getattr(config.data, 'ckpt_dir', 'checkpoints')
        if not os.path.exists(ckpt_dir):
            raise FileNotFoundError("No checkpoint directory found.")
        ckpts = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.pth') or f.endswith('.pt') or f.isdigit()]
        if not ckpts:
            raise FileNotFoundError("No checkpoint found in directory.")
        ckpt_path = max(ckpts, key=os.path.getctime)
        print(f"Using latest checkpoint: {ckpt_path}")
    diffusion.load_ddm_ckpt(ckpt_path, ema=True)

    psnr_list, ssim_list = [], []
    total_images = len(val_loader.dataset)
    processed = 0
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validating", unit="batch"):
            x = x.to(device)
            out = diffusion.model(x)
            pred_x = out["pred_x"].cpu().numpy()
            gt = x[:, 3:, :, :].cpu().numpy()
            for i in range(x.shape[0]):
                pred_img = np.clip(np.transpose(pred_x[i], (1,2,0)), 0, 1)
                gt_img = np.clip(np.transpose(gt[i], (1,2,0)), 0, 1)
                psnr = compare_psnr(gt_img, pred_img, data_range=1.0)
                # Use channel_axis for skimage >= 0.19, and set win_size if image is small
                min_side = min(gt_img.shape[0], gt_img.shape[1])
                win_size = 7 if min_side >= 7 else min_side if min_side % 2 == 1 else min_side - 1
                try:
                    ssim = compare_ssim(gt_img, pred_img, data_range=1.0, channel_axis=2, win_size=win_size)
                except TypeError:
                    # For older skimage, fallback to multichannel
                    ssim = compare_ssim(gt_img, pred_img, data_range=1.0, multichannel=True, win_size=win_size)
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                processed += 1
                # if processed % 10 == 0 or processed == total_images:
                #     tqdm.write(f"Processed {processed}/{total_images} images...")
    print(f"Validation PSNR: {np.mean(psnr_list):.4f}, SSIM: {np.mean(ssim_list):.4f}")

if __name__ == "__main__":
    main()
