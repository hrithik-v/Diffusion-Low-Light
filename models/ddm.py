import os 
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import utils
from models.unet import DiffusionUNet
from models.icdt import ICDT, timestep_embedding
from models.wavelet import DWT, IWT
from pytorch_msssim import ssim
from models.mods import HFRM
from torchvision.models import vgg16
import cv2

import wandb
from utils import metrics
# from torchinfo import summary
from tqdm import tqdm
import torch


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)
def perceptual_loss(pred, gt, feature_extractor):
    pred_features = feature_extractor(pred)
    gt_features = feature_extractor(gt)
    return F.l1_loss(pred_features, gt_features)



class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Net(nn.Module):
    def __init__(self, args, config):
        super(Net, self).__init__()

        self.args = args
        self.config = config
        self.device = config.device


        self.high_enhance0 = HFRM(in_channels=3, out_channels=64)
        self.high_enhance1 = HFRM(in_channels=3, out_channels=64)
        
        icdt_img_size = config.data.patch_size // (2 ** config.data.dwt_levels)
        self.ICDT = ICDT(latent_dim=3, img_size=icdt_img_size, patch_size=4)
        # self.Unet = DiffusionUNet(config)

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

        # Freeze all parameters except HFRM modules
        # for name, param in self.named_parameters():
        #     if ("high_enhance0" in name) or ("high_enhance1" in name):
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def sample_training(self, x_cond, b, eta=0.):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        n, c, h, w = x_cond.shape
        seq_next = [-1] + list(seq[:-1])
        x = torch.randn(n, c, h, w, device=self.device)
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)

            # et = self.Unet(torch.cat([x_cond, xt], dim=1), t)
            t_embed = timestep_embedding(t, 256)
            et = self.ICDT(xt, x_cond, t_embed)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(x.device))

        return xs[-1]

    def forward(self, x):
        data_dict = {}
        dwt, idwt = DWT(), IWT()

        input_img = x[:, :3, :, :]
        n, c, h, w = input_img.shape
        input_img_norm = data_transform(input_img)
        input_dwt = dwt(input_img_norm)

        input_LL, input_high0 = input_dwt[:n, ...], input_dwt[n:, ...]

        input_high0 = self.high_enhance0(input_high0)

        input_LL_dwt = dwt(input_LL)
        input_LL_LL, input_high1 = input_LL_dwt[:n, ...], input_LL_dwt[n:, ...]
        input_high1 = self.high_enhance1(input_high1)

        b = self.betas.to(input_img.device)

        t = torch.randint(low=0, high=self.num_timesteps, size=(input_LL_LL.shape[0] // 2 + 1,)).to(self.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:input_LL_LL.shape[0]].to(x.device)
        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

        e = torch.randn_like(input_LL_LL)

        if self.training:
            gt_img_norm = data_transform(x[:, 3:, :, :])
            gt_dwt = dwt(gt_img_norm)
            gt_LL, gt_high0 = gt_dwt[:n, ...], gt_dwt[n:, ...]

            gt_LL_dwt = dwt(gt_LL)
            gt_LL_LL, gt_high1 = gt_LL_dwt[:n, ...], gt_LL_dwt[n:, ...]

            x_gt = gt_LL_LL * a.sqrt() + e * (1.0 - a).sqrt()
            # noise_output = self.Unet(torch.cat([input_LL_LL, x_gt], dim=1), t.float())
            t_embed = timestep_embedding(t.float(), 256)
            noise_output = self.ICDT(x_gt, input_LL_LL, t_embed)
            denoise_LL_LL = self.sample_training(input_LL_LL, b)

            pred_LL = idwt(torch.cat((denoise_LL_LL, input_high1), dim=0))

            pred_x = idwt(torch.cat((pred_LL, input_high0), dim=0))
            pred_x = inverse_data_transform(pred_x)

            data_dict["input_high0"] = input_high0
            data_dict["input_high1"] = input_high1
            data_dict["gt_high0"] = gt_high0
            data_dict["gt_high1"] = gt_high1
            data_dict["pred_LL"] = pred_LL
            data_dict["gt_LL"] = gt_LL
            data_dict["noise_output"] = noise_output
            data_dict["pred_x"] = pred_x
            data_dict["e"] = e

        else:
            denoise_LL_LL = self.sample_training(input_LL_LL, b)
            pred_LL = idwt(torch.cat((denoise_LL_LL, input_high1), dim=0))
            pred_x = idwt(torch.cat((pred_LL, input_high0), dim=0))
            pred_x = inverse_data_transform(pred_x)

            data_dict["pred_x"] = pred_x

        return data_dict


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = Net(args, config)
        # print(self.model)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)
        # summary(self.model, input_size=(1, 6, 256, 256))
        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.TV_loss = TVLoss()
    
        self.feature_extractor = vgg16(pretrained=True).features.eval()
        self.feature_extractor.to(self.device)
        # Only optimize parameters that require grad (should be HFRM modules)
        # self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-4)
        # self.scheduler = None  # No scheduler by default

        self.optimizer, self.scheduler = utils.optimize.get_optimizer(self.config, self.model.parameters())
      
        self.start_epoch, self.step = 0, 0

    def sobel_filter(self, x):
        # x: (N, C, H, W), expects float tensor
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        # Apply per channel
        edge_x = torch.nn.functional.conv2d(x, sobel_x.repeat(x.shape[1],1,1,1), padding=1, groups=x.shape[1])
        edge_y = torch.nn.functional.conv2d(x, sobel_y.repeat(x.shape[1],1,1,1), padding=1, groups=x.shape[1])
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        return edge

    def edge_loss(self, pred, gt):
        pred_edges = self.sobel_filter(pred)
        gt_edges = self.sobel_filter(gt)
        return F.l1_loss(pred_edges, gt_edges)
    
    def color_loss(self, pred, gt):
        """
        Color restoration loss using LAB color space.
        Assumes input tensors are normalized to [0,1].
        """
        pred_lab = self.rgb_to_lab(pred)
        gt_lab = self.rgb_to_lab(gt)
        
        l_loss = F.l1_loss(pred_lab[:, 0, :, :], gt_lab[:, 0, :, :])  # Luminance channel
        ab_loss = F.l1_loss(pred_lab[:, 1:, :, :], gt_lab[:, 1:, :, :])  # Chromaticity channels
        
        return l_loss + ab_loss

    def rgb_to_lab(self, img):
        """
        Convert normalized RGB tensors to LAB color space.
        """
        img_np = img.permute(0, 2, 3, 1).detach().cpu().numpy()  # Convert to [B, H, W, C]
        img_np = (img_np * 255).astype(np.uint8)
        lab_img = [cv2.cvtColor(img_np[i], cv2.COLOR_RGB2LAB) for i in range(img_np.shape[0])]
        lab_img = np.stack(lab_img, axis=0)
        lab_img = torch.from_numpy(lab_img).permute(0, 3, 1, 2).float().to(img.device)  # Convert to float tensor
        return lab_img
    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        # Load epoch state if present
        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch']
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint {} step {} epoch {}".format(load_path, self.step, getattr(self, 'start_epoch', 0)))

    def train(self, DATASET):
        # cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            tqdm.write(f"Epoch: {epoch + 1}/{self.config.training.n_epochs}")
            data_start = time.time()
            data_time = 0
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}") as pbar:
                for i, (x, y) in enumerate(train_loader):
                    x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                    data_time += time.time() - data_start
                    self.model.train()
                    self.step += 1

                    x = x.to(self.device)

                    output = self.model(x)

                    total_loss, noise_loss, photo_loss, frequency_loss, color_loss_val, perceptual_loss_val  = self.estimation_loss(x, output)

                    # loss = noise_loss + photo_loss + frequency_loss
                    loss = total_loss
                    if self.step % 10 == 0:
                        # print("step:{}, lr:{:.6f}, noise_loss:{:.4f}, photo_loss:{:.4f}, frequency_loss:{:.4f}, color_loss:{:.4f}, perceptual_loss:{:.4f}".format(
                        #     self.step,
                        #     self.scheduler.get_last_lr()[0],
                        #     noise_loss.item(),
                        #     photo_loss.item(),
                        #     frequency_loss.item(),
                        #     color_loss_val.item(),
                        #     perceptual_loss_val.item()
                        # ))
                        wandb.log({
                            "step": self.step,
                            "lr": self.scheduler.get_last_lr()[0],
                            "noise_loss": noise_loss.item(),
                            "photo_loss": photo_loss.item(),
                            "frequency_loss": frequency_loss.item(),
                            "color_loss": color_loss_val.item(),
                            "perceptual_loss": perceptual_loss_val.item(),
                            "total_loss": total_loss.item()
                        })
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.ema_helper.update(self.model)
                    pbar.update(1)
                    pbar.set_postfix(loss=total_loss.item())
                    data_start = time.time()

                    if self.step % self.config.training.validation_freq == 0 and self.step != 0:
                        # self.model.eval()
                        # self.sample_validation_patches(val_loader, self.step)
                        ckpt_filename = 'model_latest.pth.tar'
                        ckpt_path = os.path.join(self.config.data.ckpt_dir, ckpt_filename)
                        print(f"Saving checkpoint to: {ckpt_path}")
                        
                        # Save the checkpoint
                        utils.logging.save_checkpoint({
                            'step': self.step,
                            'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict(),
                            'ema_helper': self.ema_helper.state_dict(),
                            'params': self.args,
                            'config': self.config
                        }, filename=os.path.join(self.config.data.ckpt_dir, 'model_latest'))
                        
                        # Save to wandb as an artifact using the correct file path
                        # artifact = wandb.Artifact('model-checkpoint', type='model')
                        # artifact.add_file(ckpt_path)
                        # wandb.log_artifact(artifact)
                                                    
                self.scheduler.step()

    def estimation_loss(self, x, output):
        # Original outputs from the model
        input_high0, input_high1, gt_high0, gt_high1 = (
            output["input_high0"], output["input_high1"],
            output["gt_high0"], output["gt_high1"]
        )

        pred_LL, gt_LL, pred_x, noise_output, e = (
            output["pred_LL"], output["gt_LL"],
            output["pred_x"], output["noise_output"], output["e"]
        )

        # Ground truth image
        gt_img = x[:, 3:, :, :].to(self.device)

        # ============= Noise Loss ===================
        noise_loss = self.l2_loss(noise_output, e)

        # ============= Frequency Loss ===================
        # frequency_loss = 0.1 * (self.l2_loss(input_high0, gt_high0) +
        #                         self.l2_loss(input_high1, gt_high1) +
        #                         self.l2_loss(pred_LL, gt_LL)) + \
        #                 0.01 * (self.TV_loss(input_high0) +
        #                         self.TV_loss(input_high1) +
        #                         self.TV_loss(pred_LL))
        # Split L2 and TV more explicitly
        l2_HF = self.l2_loss(input_high0, gt_high0) + self.l2_loss(input_high1, gt_high1)
        tv_HF = self.TV_loss(input_high0) + self.TV_loss(input_high1)

        # Optional: LL supervision can stay light if haze is already solved
        l2_LL = self.l2_loss(pred_LL, gt_LL)
        tv_LL = self.TV_loss(pred_LL)


        # Edge loss for high-frequency components
        L_edge = self.edge_loss(input_high0, gt_high0) + self.edge_loss(input_high1, gt_high1)

        frequency_loss = (
            1.0 * l2_HF +          # ↑ from 0.1 to 1.0
            0.1 * tv_HF +          # ↑ from 0.01 to 0.1
            0.2 * l2_LL +          # optional — keep this weaker
            0.01 * tv_LL           # optional — very light TV on LL
        )
        frequency_loss += 0.2 * L_edge

        # ============= Photometric Loss ===================
        # Content loss (L1)
        content_loss = self.l1_loss(pred_x, gt_img)
        # SSIM loss
        ssim_loss = 1 - ssim(pred_x, gt_img, data_range=1.0).to(self.device)
        photo_loss = content_loss + ssim_loss

        # ============= Color Loss ===================
        color_loss_val = self.color_loss(pred_x, gt_img)

        # ============= Perceptual Loss ===================
        perceptual_loss_val = perceptual_loss(pred_x, gt_img, self.feature_extractor)

        # Combine losses
        total_loss = (
            noise_loss +
            0.5 * frequency_loss +
            0.5 * photo_loss +
            0.2 * color_loss_val +
            0.2 * perceptual_loss_val
        )

        return total_loss, noise_loss, photo_loss, frequency_loss, color_loss_val, perceptual_loss_val


    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.args.image_folder, self.config.data.type + str(self.config.data.patch_size))
        self.model.eval()
        all_metrics = []
        with torch.no_grad():
            tqdm.write(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y) in enumerate(val_loader):
                b, _, img_h, img_w = x.shape
                img_h_32 = int(32 * np.ceil(img_h / 32.0))
                img_w_32 = int(32 * np.ceil(img_w / 32.0))
                x = F.pad(x, (0, img_w_32 - img_w, 0, img_h_32 - img_h), 'reflect')

                # Input and Ground-Truth Split
                x_cond = x[:, :3, :, :].to(self.device)  # Input
                gt = x[:, 3:, :, :].to(self.device)  # Ground truth

                # Forward pass through the model
                output = self.model(x_cond)
                pred_x = output["pred_x"]
                pred_x = pred_x[:, :, :img_h, :img_w]

                # Save the restored image
                save_path = os.path.join(image_folder, str(step), f"{y[0]}.png")
                utils.logging.save_image(pred_x, save_path)
                # Evaluate metrics for the current sample
                sample_metrics = metrics.evaluate_metrics(pred_x, gt)
                all_metrics.append(sample_metrics)  # Append metrics for aggregation later

                # Print each sample's metrics for debugging (optional)
                # print(f"Metrics for {y[0]}: {sample_metrics}")

        # Aggregate metrics over the dataset
        avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0]}

        # Log average metrics to WandB
        tqdm.write(f"Step {step}: Average Metrics: {avg_metrics}")
        wandb.log({"average_metrics": avg_metrics, "step": step})


'''
class DummyConfig:
    class Diffusion:
        beta_schedule = "linear"
        beta_start = 0.0001
        beta_end = 0.02
        num_diffusion_timesteps = 1000
    class data:
        conditional = True
        pass

    class model:        
            in_channels=3
            out_ch=3
            ch= 64
            ch_mult= [1, 2, 3, 4]
            num_res_blocks= 2
            dropout= 0.0
            ema_rate= 0.999
            ema= True
            resamp_with_conv= True
    # class device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    diffusion = Diffusion()
    # device = Device()

class DummyArgs:
    sampling_timesteps = 100

# Initialize dummy arguments and configuration
args = DummyArgs()
config = DummyConfig()

# Create a dummy input tensor
# The model expects a 6-channel input during training (3 for input, 3 for ground truth)
dummy_input = torch.randn(1, 6, 64, 64).to(config.device)

# Initialize the Net class
net = Net(args, config).to(config.device)

# Forward pass through the network
output = net(dummy_input)

# Print the shapes of each layer's output
print("Input shape:", dummy_input.shape)
# The dummy code runs in training mode, which has more outputs in the dictionary
if net.training:
    print("Output shape:", output["pred_x"].shape)
    print("Input high0 shape:", output["input_high0"].shape)
    print("Input high1 shape:", output["input_high1"].shape)
    print("GT high0 shape:", output["gt_high0"].shape)
    print("GT high1 shape:", output["gt_high1"].shape)
    print("Pred LL shape:", output["pred_LL"].shape)
    print("GT LL shape:", output["gt_LL"].shape)
    print("Noise output shape:", output["noise_output"].shape)
    print("E shape:", output["e"].shape)
else:
    print("Output shape:", output["pred_x"].shape)

'''