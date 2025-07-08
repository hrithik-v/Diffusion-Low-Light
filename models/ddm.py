import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import utils
from models.wavelet import DWT, IWT
from pytorch_msssim import ssim
from models.mods import HFRM
from models.icdt import ICDT, timestep_embedding
import torchvision.models
import wandb
from tqdm import tqdm

def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


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
                self.shadow[name] = param.data.clone().to(param.device)

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

    def load_state_dict(self, state_dict, model):
        if isinstance(model, nn.DataParallel):
            model = model.module
        
        # Move shadow params to the same device as model params
        for name, param in model.named_parameters():
            if name in state_dict:
                state_dict[name] = state_dict[name].to(param.device)

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
        self.dwt_levels = config.data.dwt_levels

        self.high_enhancers = nn.ModuleList(
            [HFRM(in_channels=3, out_channels=64) for _ in range(self.dwt_levels)]
        )
        
        icdt_img_size = config.data.patch_size // (2 ** self.dwt_levels)
        self.ICDT = ICDT(latent_dim=3, img_size=icdt_img_size, patch_size=4)

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

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

            t_embed = timestep_embedding(t, 256)
            et = self.ICDT(xt, x_cond, t_embed)

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            # DDIM sampling with fixed variance
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

        # DWT Decomposition
        input_LL = input_img_norm
        input_highs = []
        for i in range(self.dwt_levels):
            input_dwt = dwt(input_LL)
            input_LL, input_high = input_dwt[:n, ...], input_dwt[n:, ...]
            input_high = self.high_enhancers[i](input_high)
            input_highs.append(input_high)

        final_LL_input = input_LL

        b = self.betas.to(input_img.device)

        t = torch.randint(low=0, high=self.num_timesteps, size=(final_LL_input.shape[0] // 2 + 1,)).to(self.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:final_LL_input.shape[0]].to(x.device)
        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

        e = torch.randn_like(final_LL_input)

        if self.training:
            gt_img_norm = data_transform(x[:, 3:, :, :])
            
            # GT DWT Decomposition
            gt_LL = gt_img_norm
            gt_highs = []
            for i in range(self.dwt_levels):
                gt_dwt = dwt(gt_LL)
                gt_LL, gt_high = gt_dwt[:n, ...], gt_dwt[n:, ...]
                gt_highs.append(gt_high)
            
            final_gt_LL = gt_LL

            # ---------Noise addition----------
            x_gt = final_gt_LL * a.sqrt() + e * (1.0 - a).sqrt()
            # ---------Predicted noise----------
            t_embed = timestep_embedding(t.float(), 256)
            noise_output = self.ICDT(x_gt, final_LL_input, t_embed)

            # ---------Denoising that added noise to get denoised LL Subbands----------
            denoised_LL = self.sample_training(final_LL_input, b)

            # Inverse DWT Reconstruction
            pred_LL = denoised_LL
            for i in range(self.dwt_levels - 1, -1, -1):
                pred_LL = idwt(torch.cat((pred_LL, input_highs[i]), dim=0))
            
            pred_x = inverse_data_transform(pred_LL)

            data_dict["input_highs"] = input_highs
            data_dict["gt_highs"] = gt_highs
            data_dict["final_gt_LL"] = final_gt_LL
            data_dict["noise_output"] = noise_output
            data_dict["pred_x"] = pred_x
            data_dict["e"] = e
            data_dict["denoised_LL"] = denoised_LL

        else:
            denoised_LL = self.sample_training(final_LL_input, b)
            
            # Inverse DWT Reconstruction
            pred_LL = denoised_LL
            for i in range(self.dwt_levels - 1, -1, -1):
                pred_LL = idwt(torch.cat((pred_LL, input_highs[i]), dim=0))

            pred_x = inverse_data_transform(pred_LL)

            data_dict["pred_x"] = pred_x

        return data_dict


class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg19(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg19(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg19(pretrained=True).features[9:18].eval())
        blocks.append(torchvision.models.vgg19(pretrained=True).features[18:27].eval())
        blocks.append(torchvision.models.vgg19(pretrained=True).features[27:36].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, x1, x2):
        mean_rgb1 = torch.mean(x1, [2, 3], keepdim=True)
        mean_rgb2 = torch.mean(x2, [2, 3], keepdim=True)

        d_rgb1 = torch.pow(mean_rgb1 - torch.mean(mean_rgb1, 1, keepdim=True), 2)
        d_rgb2 = torch.pow(mean_rgb2 - torch.mean(mean_rgb2, 1, keepdim=True), 2)

        return torch.sqrt(torch.pow(d_rgb1 - d_rgb2, 2).sum(1) + 1e-6).mean()


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        # Initialize model and move to device
        self.model = Net(args, config).to(self.device)
        # Use DataParallel only if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        # Use DataParallel for VGGPerceptualLoss if multiple GPUs are available
        perceptual = VGGPerceptualLoss().to(self.device)
        if torch.cuda.device_count() > 1:
            perceptual = nn.DataParallel(perceptual)
        self.perceptual_loss = perceptual
        self.color_loss = ColorLoss().to(self.device)
        self.TV_loss = TVLoss()

        self.optimizer, self.scheduler = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.ema_helper.load_state_dict(checkpoint['ema_helper'], self.model)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        # Restore epoch and step if present in checkpoint
        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch']
        if 'step' in checkpoint:
            self.step = checkpoint['step']
        # if 'optimizer' in checkpoint:
        # if 'scheduler' in checkpoint:
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint {} step {} epoch {}".format(load_path, self.step, getattr(self, 'start_epoch', 0)))

    def train(self, DATASET):
        # cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        scaler = torch.amp.GradScaler("cuda")

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            # print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.training.n_epochs}")):
                # print(f"x.shape: {x.shape}, y: {y}")
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                # print(f"Processed x.shape: {x.shape}, y: {y}")
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                x = x.to(self.device)

                with torch.amp.autocast("cuda"):
                    output = self.model(x) # [B, 2*C, H, W] for training, [B, C, H, W] for inference
                    noise_loss, photo_loss, frequency_loss, perceptual_loss, color_loss = self.estimation_loss(x, output)
                    # Balanced loss weighting (no vlb in old commit)
                    noise = noise_loss
                    photo = photo_loss
                    freq = frequency_loss
                    percep = perceptual_loss
                    color = color_loss
                    if noise is not None and hasattr(noise, 'dim') and noise.dim() > 0:
                        noise = noise.mean()
                    if photo is not None and hasattr(photo, 'dim') and photo.dim() > 0:
                        photo = photo.mean()
                    if freq is not None and hasattr(freq, 'dim') and freq.dim() > 0:
                        freq = freq.mean()
                    if percep is not None and hasattr(percep, 'numel') and percep.numel() > 1:
                        percep = percep.mean()
                    if color is not None and hasattr(color, 'dim') and color.dim() > 0:
                        color = color.mean()
                        
                    loss = (
                        noise + photo + freq +
                        0.2 * percep + 
                        0.05 * color
                    )
                    # Check for NaN values and skip if found
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: NaN or Inf detected in loss at step {self.step}")
                        print(f"noise: {noise.item()}, photo: {photo.item()}, freq: {freq.item()}, percep: {percep.item()}, color: {color.item()}")
                        continue

                if self.step % 10 == 0:
                    # print("step:{}, lr:{:.6f}, loss:{:.4f}, noise:{:.4f}, photo:{:.4f}, "
                    #       "freq:{:.4f}, percep:{:.4f}, color:{:.4f}".format(self.step, self.scheduler.get_last_lr()[0],
                    #                                  loss.item(),
                    #                                  noise_loss.item(), photo_loss.item(),
                    #                                  frequency_loss.item(),
                    #                                  percep.item(),
                    #                                  color_loss.item()))
                    # wandb logging
                    if wandb.run is not None:
                        wandb.log({
                            'loss': loss.item(),
                            'noise_loss': noise_loss.item(),
                            'photo_loss': photo_loss.item(),
                            'frequency_loss': frequency_loss.item(),
                            'perceptual_loss': percep.item(),
                            'color_loss': color_loss.item(),
                            'lr': self.scheduler.get_last_lr()[0],
                            'epoch': epoch,
                            'step': self.step
                        })

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()

                # Compute total gradient norm before clipping
                # scaler.unscale_(self.optimizer)
                # total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # if total_norm > 1.0:
                #     print(f"Warning: Gradient norm ({total_norm:.4f}) exceeded max_norm before clipping at step {self.step}")
                
                scaler.step(self.optimizer)
                scaler.update()
                self.ema_helper.update(self.model)
                data_start = time.time()

                # if self.step % self.config.training.validation_freq == 0 and self.step != 0:
                #     self.model.eval()
                #     self.sample_validation_patches(val_loader, self.step)

            if (epoch+1) % 4 == 0:
                utils.logging.save_checkpoint({'step': self.step, 'epoch': epoch + 1,
                                                'state_dict': self.model.state_dict(),
                                                'optimizer': self.optimizer.state_dict(),
                                                'scheduler': self.scheduler.state_dict(),
                                                'ema_helper': self.ema_helper.state_dict(),
                                                'params': self.args,
                                                'config': self.config},
                                                filename=os.path.join(self.config.data.ckpt_dir, "latest"))
            if (epoch+1) % 50 == 0:
                utils.logging.save_checkpoint({'step': self.step, 'epoch': epoch + 1,
                                                'state_dict': self.model.state_dict(),
                                                'optimizer': self.optimizer.state_dict(),
                                                'scheduler': self.scheduler.state_dict(),
                                                'ema_helper': self.ema_helper.state_dict(),
                                                'params': self.args,
                                                'config': self.config},
                                                filename=os.path.join(self.config.data.ckpt_dir, str(epoch + 1)))
                        
            self.scheduler.step()

    def estimation_loss(self, x, output):

        input_highs = output["input_highs"]
        gt_highs = output["gt_highs"]

        denoised_LL, pred_x, noise_output, e = output["denoised_LL"], output["pred_x"], output["noise_output"], output["e"]
        final_gt_LL = output["final_gt_LL"]

        gt_img = x[:, 3:, :, :].to(self.device)
        # =============noise loss==================
        noise_loss = self.l2_loss(noise_output, e)

        # =============frequency loss==================
        frequency_loss_l2 = self.l2_loss(denoised_LL, final_gt_LL)
        frequency_loss_tv = self.TV_loss(denoised_LL)

        for i in range(len(input_highs)):
            frequency_loss_l2 += self.l2_loss(input_highs[i], gt_highs[i])
            frequency_loss_tv += self.TV_loss(input_highs[i])

        # Normalize TV loss by number of high-frequency components
        # frequency_loss_tv = frequency_loss_tv / (len(input_highs) + 1)
        frequency_loss = 0.1 * frequency_loss_l2 + 0.01 * frequency_loss_tv

        # =============photo loss==================
        content_loss = self.l1_loss(pred_x, gt_img)
        if content_loss < 0 or content_loss > 1:
            print(f"Warning: Content loss value out of [0,1] range: {content_loss.item()}")
            print(f"pred_x: {pred_x.min().item()} to {pred_x.max().item()}, mean: {pred_x.mean().item()}, var: {pred_x.var().item()}")

        ssim_val = ssim(pred_x, gt_img, data_range=1.0).to(self.device)
        if ssim_val < 0 or ssim_val > 1:
            print(f"Warning: SSIM loss value out of [0,1] range: {ssim_val.item()}")
            print(f"pred_x: {pred_x.min().item()} to {pred_x.max().item()}, mean: {pred_x.mean().item()}, var: {pred_x.var().item()}")
        
        ssim_loss = 1 - torch.clamp(ssim_val, min=0, max=1)
        photo_loss = content_loss + ssim_loss

        # =============perceptual loss==================
        perceptual_loss = self.perceptual_loss(pred_x, gt_img)

        # =============color loss==================
        color_loss = self.color_loss(pred_x, gt_img)

        return noise_loss, photo_loss, frequency_loss, perceptual_loss, color_loss

    def sample_validation_patches(self, val_loader, step):
        self.model.eval()
        image_folder = os.path.join(self.args.image_folder, 
                                  f"{self.config.data.type}{self.config.data.patch_size}")
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                b, _, img_h, img_w = x.shape
                x = F.pad(x, 
                         (0, 32 * ((img_w + 31) // 32) - img_w,
                          0, 32 * ((img_h + 31) // 32) - img_h),
                         'reflect')
                out = self.model(x.to(self.device))
                pred_x = out["pred_x"][:, :, :img_h, :img_w]
                utils.logging.save_image(
                    pred_x, 
                    os.path.join(image_folder, str(step), f"{y[0]}.png")
                )
                break  # Process only one batch


