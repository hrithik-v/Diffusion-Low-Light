import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Replacement Helper Modules ---

class ChannelAttention(nn.Module):
    """A simple and effective channel attention module."""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden_planes = max(in_planes // ratio, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_planes, in_planes, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """A simple and effective spatial attention module."""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ResidualAttentionBlock(nn.Module):
    """The new, powerful building block for HFRM."""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False)
        )
        self.ca = ChannelAttention(dim)
        self.sa = SpatialAttention()

    def forward(self, x):
        res = x
        x = self.conv(x)
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x + res

# --- The New and Improved HFRM ---

class HFRM_v2(nn.Module):
    def __init__(self, in_channels=9, out_channels=64, n_blocks=4):
        """
        in_channels should be 3 (HL) + 3 (LH) + 3 (HH) = 9.
        out_channels is the internal dimension.
        """
        super().__init__()
        self.conv_head = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.body = nn.Sequential(
            *[ResidualAttentionBlock(out_channels) for _ in range(n_blocks)]
        )
        
        self.conv_tail = nn.Conv2d(out_channels, in_channels, 3, padding=1)

    def forward(self, x):
        # Split input into three subbands along batch dimension like original HFRM
        b, c, h, w = x.shape
        x_hl = x[:b//3]
        x_lh = x[b//3:2*b//3]
        x_hh = x[2*b//3:]
        # Early Fusion: concatenate channel-wise
        x_cat = torch.cat([x_hl, x_lh, x_hh], dim=1)  # shape [b//3, in_channels*3, h, w]
        residual = x_cat
        # Head, body, and tail
        out = self.conv_head(x_cat)
        out = self.body(out)
        out = self.conv_tail(out)
        out = out + residual
        # Restore original subband batch shape: split channel axis into three and concatenate along batch
        subbands = torch.chunk(out, 3, dim=1)  # list of tensors with shape [b//3, in_planes, h, w]
        out = torch.cat(subbands, dim=0)       # shape [b, in_planes, h, w]
        return out