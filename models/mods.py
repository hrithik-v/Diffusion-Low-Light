# NEW VERSION
import torch
import torch.nn as nn
import math


class cross_attention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super(cross_attention, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, ctx):
        B, C, H, W = q.shape
        q = q.view(B, C, -1).permute(0, 2, 1)  # (B, HW, C)
        k = ctx.view(B, C, -1).permute(0, 2, 1)
        v = ctx.view(B, C, -1).permute(0, 2, 1)

        q = self.q_proj(q).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, -1, C)
        out = self.out_proj(out).permute(0, 2, 1).view(B, C, H, W)
        return out


class ResidualConv(nn.Module):
    def __init__(self, channels):
        super(ResidualConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.conv(x)


class DilatedResblock(nn.Module):
    def __init__(self, channels):
        super(DilatedResblock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 3, dilation=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return x + self.convs(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SimpleFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Takes two streams (HH + LH or HH + HL) → fuses them
        self.fuse = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x_stream, x_HH):
        # x_stream = x_LH or x_HL
        fused = torch.cat([x_stream, x_HH], dim=1)
        return self.fuse(fused)

class HFRM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HFRM, self).__init__()

        self.conv_head = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # Separate local + context branches
        self.local_HL = ResidualConv(out_channels)
        self.local_LH = ResidualConv(out_channels)
        self.local_HH = ResidualConv(out_channels)

        self.dilated_HL = DilatedResblock(out_channels)
        self.dilated_LH = DilatedResblock(out_channels)
        self.dilated_HH = DilatedResblock(out_channels)

        self.ca_LH = cross_attention(out_channels, num_heads=8)
        self.ca_HL = cross_attention(out_channels, num_heads=8)

        self.fuse_HH = nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)
        self.se = SEBlock(out_channels)
        self.conv_tail = nn.Conv2d(out_channels, in_channels, 3, padding=1)
        
        self.fuse_LH_HH = SimpleFusion(out_channels)
        self.fuse_HL_HH = SimpleFusion(out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x

        x = self.conv_head(x)
        x_HL, x_LH, x_HH = x[:b // 3], x[b // 3:2 * b // 3], x[2 * b // 3:]

        x_HL = self.local_HL(x_HL) + self.dilated_HL(x_HL)
        x_LH = self.local_LH(x_LH) + self.dilated_LH(x_LH)
        x_HH = self.local_HH(x_HH) + self.dilated_HH(x_HH)

        # hh_from_lh = self.ca_LH(x_LH, x_HH)
        # hh_from_hl = self.ca_HL(x_HL, x_HH)
        hh_from_lh = self.fuse_LH_HH(x_LH, x_HH)
        hh_from_hl = self.fuse_HL_HH(x_HL, x_HH)
        x_HH = self.fuse_HH(torch.cat([hh_from_lh, hh_from_hl], dim=1))
        x_HH = self.se(x_HH)

        out = self.conv_tail(torch.cat([x_HL, x_LH, x_HH], dim=0))
        return out + residual
