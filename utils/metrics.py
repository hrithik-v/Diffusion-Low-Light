import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
import math
import numpy as np
import cv2

# ====================================================
# Standard Image Quality Metrics
# ====================================================

def compute_psnr(pred, gt, data_range=1.0):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    pred, gt: Tensors of shape [B, C, H, W], values in [0,1] if data_range=1.0.
    """
    mse = F.mse_loss(pred, gt, reduction='mean')
    psnr = 10 * math.log10((data_range**2) / mse.item())
    return psnr

def compute_ssim(pred, gt, data_range=1.0):
    """
    Compute Structural Similarity Index (SSIM).
    pred, gt: Tensors of shape [B, C, H, W], values in [0,1].
    """
    return ssim(pred, gt, data_range=data_range).item()

# ====================================================
# Underwater-Specific Metrics: UIQM & UCIQE
#
# UIQM is defined as:
#   UIQM = c1*UICM + c2*UISM + c3*UIConM
# Constants from the literature:
#   c1 = 0.0282, c2 = 0.2953, c3 = 3.5753
#
# UCIQE is defined as:
#   UCIQE = w1*std_chroma + w2*contrast_l + w3*mean_sat
# Constants from the literature:
#   w1 = 0.4680, w2 = 0.2745, w3 = 0.2576
#
# References:
# - Panetta, Karen, et al. "Human-visual-system-inspired underwater image quality measures." 
#   IEEE Journal of Oceanic Engineering 41.3 (2015): 541-551.
# - Yang, Meng, and Arcot Sowmya. "An underwater color image quality evaluation metric." 
#   IEEE Transactions on Image Processing 24.12 (2015): 6062-6071.
# ====================================================

def _uicm(im):
    # UICM: Measure of colorfulness
    # im: numpy image in [0,1], shape [H,W,3], RGB
    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]

    rg = r - g
    yb = 0.5*(r + g) - b

    mu_rg = np.mean(rg)
    mu_yb = np.mean(yb)

    sigma_rg = np.std(rg)
    sigma_yb = np.std(yb)

    UICM = np.sqrt((mu_rg**2 + mu_yb**2)) + np.sqrt((sigma_rg**2 + sigma_yb**2))
    return UICM

def _uism(im):
    # UISM: Measure of sharpness
    # Based on Laplacian or gradient-based measure
    gray = cv2.cvtColor((im*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    # Standard deviation of Laplacian as a measure of sharpness
    UISM = np.std(lap)
    return UISM

def _uiconm(im):
    # UIConM: Measure of contrast based on the value channel
    hsv = cv2.cvtColor((im*255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    v = hsv[:,:,2]/255.0
    UIConM = np.std(v)
    return UIConM

def compute_uiqm(pred):
    """
    Compute UIQM for the first image in the batch.
    pred: Tensor [B, C, H, W], normalized to [0,1].
    Assumes B=1.
    """
    img = pred[0].permute(1,2,0).cpu().numpy() # [H,W,C]
    # Ensure float
    if img.dtype == np.uint8:
        img = img.astype(np.float32)/255.0

    UICM = _uicm(img)
    UISM = _uism(img)
    UIConM = _uiconm(img)

    # UIQM formula
    UIQM = (0.0282 * UICM) + (0.2953 * UISM) + (3.5753 * UIConM)
    return UIQM

def compute_uciqe(pred):
    """
    Compute UCIQE for the first image in the batch.
    pred: Tensor [B, C, H, W], normalized to [0,1].
    Assumes B=1.
    """
    img = pred[0].permute(1,2,0).cpu().numpy()
    if img.dtype == np.uint8:
        img = img.astype(np.float32)/255.0

    # Convert to LAB
    lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    L = lab[:,:,0].astype(np.float32)
    a = lab[:,:,1].astype(np.float32) - 128.0
    b = lab[:,:,2].astype(np.float32) - 128.0

    # Chroma
    chroma = np.sqrt(a**2 + b**2)
    std_chroma = np.std(chroma)

    # L contrast
    L_norm = L/255.0
    con_l = (np.max(L_norm)-np.min(L_norm)) / (np.max(L_norm)+np.min(L_norm) + 1e-8)

    # Saturation
    hsv = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    S = hsv[:,:,1].astype(np.float32)/255.0
    mean_sat = np.mean(S)

    # UCIQE formula
    # from Yang and Sowmya (2015)
    UCIQE = 0.4680*std_chroma + 0.2745*con_l + 0.2576*mean_sat
    return UCIQE

# ====================================================
# Utility Function for Integration
# ====================================================

def evaluate_metrics(pred, gt):
    """
    Compute a set of metrics for given prediction and ground-truth tensors.
    pred, gt: Tensors of shape [B, C, H, W], values in [0,1].
    Returns a dictionary of metric values.
    """
    metrics_dict = {}
    metrics_dict['PSNR'] = compute_psnr(pred, gt)
    metrics_dict['SSIM'] = compute_ssim(pred, gt)
    metrics_dict['UIQM'] = compute_uiqm(pred)
    metrics_dict['UCIQE'] = compute_uciqe(pred)
    return metrics_dict
