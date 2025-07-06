import argparse
import os
import torch
import numpy as np
import torch.nn.functional as F
from pytorch_msssim import ssim
from models import DenoisingDiffusion
import datasets
from torch.utils.data import DataLoader
import utils

def dict2namespace(config):
    import argparse
    ns = argparse.Namespace()
    for k, v in config.items():
        if isinstance(v, dict):
            setattr(ns, k, dict2namespace(v))
        else:
            setattr(ns, k, v)
    return ns


def load_config(path):
    import yaml
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return dict2namespace(cfg)


    # python val.py --run_name 3rd_run --ckpt_step 300 --batch_size 8
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/LOLv1.yml')
    parser.add_argument('--run_name', type=str, required=True,
                        help='Name of the run (subfolder under ckpt/)')
    parser.add_argument('--ckpt_step', type=str,
                        help='Specific checkpoint step (e.g. 180). If omitted, use latest.')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sampling_timesteps', type=int, default=40,
                        help='Number of diffusion sampling steps (reverse)')
    args = parser.parse_args()

    # load config and device
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device

    # checkpoint selection
    ckpt_dir = os.path.join('ckpt', args.run_name)
    if args.ckpt_step:
        ckpt_file = f"{args.ckpt_step}.pth.tar"
    else:
        files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth') or f.endswith('.pth.tar')]
        ckpt_file = max(files, key=lambda f: os.path.getctime(os.path.join(ckpt_dir, f)))
    ckpt_path = os.path.join(ckpt_dir, ckpt_file)
    print(f"Using checkpoint: {ckpt_path}")

    # prepare data loader (override batch_size for validation)
    DATASET = datasets.__dict__[config.data.type](config)
    _, default_val_loader = DATASET.get_loaders()
    val_dataset = default_val_loader.dataset
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )

    # build model and load checkpoint
    diffusion = DenoisingDiffusion(args, config)
    diffusion.load_ddm_ckpt(ckpt_path, ema=True)
    # use model as defined in DenoisingDiffusion (already DataParallel if multi-GPU)
    model = diffusion.model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for inference.")
    model.eval()

    psnr_vals = []
    ssim_vals = []

    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            out = model(x)
            pred = out['pred_x']
            gt = x[:, 3:, :, :]

            # per-sample metrics
            for i in range(pred.shape[0]):
                p = pred[i:i+1]
                g = gt[i:i+1]
                mse = F.mse_loss(p, g)
                psnr = 10 * torch.log10(1.0 / mse)
                ssim_score = ssim(p, g, data_range=1.0)
                psnr_vals.append(psnr.item())
                ssim_vals.append(ssim_score.item())

    print(f"Validation PSNR: {np.mean(psnr_vals):.4f} dB")
    print(f"Validation SSIM: {np.mean(ssim_vals):.4f}")

if __name__ == '__main__':
    main()
