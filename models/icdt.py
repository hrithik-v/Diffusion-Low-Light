import torch
import torch.nn as nn
import math
from einops import rearrange


# --- Helper Modules ---

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, patch_size, emb_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, C, H/p, W/p)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class AdaLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp_gamma = nn.Linear(dim, dim)
        self.mlp_beta = nn.Linear(dim, dim)

    def forward(self, x, t_embed):
        gamma = self.mlp_gamma(t_embed).unsqueeze(1)
        beta = self.mlp_beta(t_embed).unsqueeze(1)
        x = self.norm(x)
        return gamma * x + beta

class DiffusionTransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm1 = AdaLayerNorm(dim)
        self.norm2 = AdaLayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))

    def forward(self, x, t_embed):
        normed1 = self.norm1(x, t_embed)
        attn_out = self.self_attn(normed1, normed1, normed1)[0]
        x = x + self.alpha1 * attn_out

        normed2 = self.norm2(x, t_embed)
        mlp_out = self.mlp(normed2)
        x = x + self.alpha2 * mlp_out

        return x

# --- Main ICDT Model ---

class ICDT(nn.Module):
    def __init__(self, latent_dim=3, embed_dim=384, patch_size=4, img_size=32, depth=12, heads=6):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels=latent_dim * 2, patch_size=patch_size, emb_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2, embed_dim))

        self.time_embed = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(embed_dim, heads) for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, patch_size * patch_size * latent_dim * 2)
        self.patch_size = patch_size
        self.img_size = img_size
        self.latent_dim = latent_dim

    def forward(self, z_t, z_cond, t):
        x = torch.cat([z_t, z_cond], dim=1)
        x = self.patch_embed(x) + self.pos_embed

        t_embed = timestep_embedding(t, 256)
        t_embed = self.time_embed(t_embed)

        for blk in self.blocks:
            x = blk(x, t_embed)

        x = self.final_norm(x)
        x = self.out_proj(x)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                      h=self.img_size // self.patch_size, w=self.img_size // self.patch_size,
                      p1=self.patch_size, p2=self.patch_size, c=self.latent_dim * 2)
        return x

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

