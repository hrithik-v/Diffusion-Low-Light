import torch
import numpy as np
import utils
import os
import torch.nn.functional as F
import utils.metrics as metrics
import wandb
from tqdm import tqdm
def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, val_loader):
        image_folder = os.path.join(self.args.image_folder, self.config.data.val_dataset)
        all_metrics = []
        with tqdm(total=len(val_loader), desc="Evaluation Progress") as pbar:
            with torch.no_grad():
                for i, (x, y) in enumerate(val_loader):
                    x_cond = x[:, :3, :, :].to(self.diffusion.device)
                    gt = x[:, 3:, :, :].to(self.diffusion.device)  # Ground truth
                    b, c, h, w = x_cond.shape
                    img_h_32 = int(32 * np.ceil(h / 32.0))
                    img_w_32 = int(32 * np.ceil(w / 32.0))
                    x_cond = F.pad(x_cond, (0, img_w_32 - w, 0, img_h_32 - h), 'reflect')
                    x_output = self.diffusive_restoration(x_cond)
                    x_output = x_output[:, :, :h, :w]
                    utils.logging.save_image(x_output, os.path.join(image_folder, f"{y[0]}.png"))
                    pbar.write(f"processing image {y[0]}")

                    metrics_result = metrics.evaluate_metrics(x_output, gt)
                    all_metrics.append(metrics_result)
                    # print(f"Metrics for {y[0]}: {metrics_result}")
                    wandb.log(metrics_result)

                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix(metrics=metrics_result)
                # Aggregate metrics over the dataset
            avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0]}
            # print(f"Average Metrics: {avg_metrics}")
            wandb.log({"average_metrics": avg_metrics})


    def diffusive_restoration(self, x_cond):
        x_output = self.diffusion.model(x_cond)
        return x_output["pred_x"]

