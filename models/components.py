import math
import torch
from torch import nn
import torch.nn.functional as F

# Patch embedding: converts image to patch tokens
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, embed_dim=768, img_size=32):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                  # (B, C, H, W) -> (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # -> (B, num_patches, embed_dim)
        return x

# RMS LayerNorm
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(var + self.eps)
        return self.weight * x_normed.to(x.dtype)

# Rotary positional embedding
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=1000, device=None):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, dtype):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, pos_ids):
    cos = cos[pos_ids].unsqueeze(1)
    sin = sin[pos_ids].unsqueeze(1)
    q = q.view(*q.shape[:-1], -1, 2).transpose(-1, -2).reshape(q.shape)
    k = k.view(*k.shape[:-1], -1, 2).transpose(-1, -2).reshape(k.shape)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

# Config class for MLA
class MLAConfig:
    def __init__(self):
        self.hidden_size = 768
        self.num_heads = 6
        self.max_position_embeddings = 1024
        self.rope_theta = 1280
        self.attention_dropout = 0
        self.q_lora_rank = 1536
        self.qk_rope_head_dim = 64
        self.kv_lora_rank = 512
        self.v_head_dim = 128
        self.qk_nope_head_dim = 128
        self.attention_bias = False

# MLA + RoPE attention module
class MLA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.q_down = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
        self.q_norm = RMSNorm(config.q_lora_rank)
        self.q_up = nn.Linear(config.q_lora_rank, config.num_heads * self.q_head_dim, bias=False)

        self.kv_down = nn.Linear(config.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim, bias=config.attention_bias)
        self.kv_norm = RMSNorm(config.kv_lora_rank)
        self.kv_up = nn.Linear(config.kv_lora_rank, config.num_heads * (config.qk_nope_head_dim + config.v_head_dim), bias=False)

        self.rotary_emb = RotaryEmbedding(config.qk_rope_head_dim, config.max_position_embeddings, config.rope_theta)
        self.o_proj = nn.Linear(config.num_heads * config.v_head_dim, config.hidden_size, bias=config.attention_bias)
        self.dropout = config.attention_dropout

    def forward(self, x, mask=None, pos_ids=None):
        B, N, _ = x.size()
        q = self.q_up(self.q_norm(self.q_down(x))).view(B, N, self.num_heads, -1).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [128, 64], dim=-1)

        kv = self.kv_down(x)
        kv_content, k_pe = torch.split(kv, [512, 64], dim=-1)
        k_pe = k_pe.view(B, N, 1, 64).transpose(1, 2)
        kv_up = self.kv_up(self.kv_norm(kv_content)).view(B, N, self.num_heads, -1).transpose(1, 2)
        k_nope, v = torch.split(kv_up, [128, 128], dim=-1)

        cos, sin = self.rotary_emb(v, seq_len=N)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, pos_ids)

        q_full = torch.cat([q_nope, q_pe], dim=-1)
        k_full = torch.cat([k_nope, k_pe], dim=-1)

        scores = torch.matmul(q_full, k_full.transpose(-1, -2)) / math.sqrt(self.q_head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.dropout(F.softmax(scores, dim=-1, dtype=torch.float32), p=self.dropout, training=self.training)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, -1)
        return self.o_proj(out)
