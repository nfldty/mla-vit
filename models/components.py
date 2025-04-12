import math
import torch
from torch import nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, embed_dim=768, img_size=32):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # Calculate number of patches in the image
        self.num_patches = (img_size // patch_size) ** 2
        # Convolution layer to split image and map each patch to an embedding vector
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(2)  # Flatten to shape (batch_size, embed_dim, num_patches^2)
        # Transpose to (batch_size, num_patches^2, embed_dim)
        x = x.transpose(1, 2)
        return x  # Final shape: (batch_size, num_patches^2, embed_dim)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=1000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(t, self.inv_freq.to(t.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]  # Take first half
    x2 = x[..., x.shape[-1] // 2:]   # Take second half
    return torch.cat((-x2, x1), dim=-1)  # Swap and concatenate

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # Extract corresponding cos values and expand dims
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # Extract corresponding sin values and expand dims

    # Process q by splitting into two parts then rotating
    b, h, s, d = q.shape  # Get q shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)  # Split and rearrange

    # Similar processing for k
    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)  # Split and rearrange

    # Apply rotary position encoding
    q_embed = (q * cos) + (rotate_half(q) * sin)  # Query vector rotary position encoding
    k_embed = (k * cos) + (rotate_half(k) * sin)  # Key vector rotary position encoding

    return q_embed, k_embed

class MLAConfig:

    def __init__(self, hidden_size, num_heads, max_position_embeddings, rope_theta, attention_dropout, 
                 q_lora_rank, qk_rope_head_dim, kv_lora_rank, v_head_dim, qk_nope_head_dim, attention_bias):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.attention_bias = attention_bias

class MLA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        # Dimensions for query and key rope application
        self.qk_rope_head_dim = config.qk_rope_head_dim
        # For value compression vectors
        self.kv_lora_rank = config.kv_lora_rank
        # Dimension size for each head
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        
        self.q_down_proj = nn.Linear(
            self.hidden_size,
            self.q_lora_rank,
            bias=config.attention_bias,
        )
        self.q_down_layernorm = RMSNorm(self.q_lora_rank)
        self.q_up_proj = nn.Linear(
            self.q_lora_rank,
            self.num_heads * self.q_head_dim,
            # Final output needs splitting - part for nope, part for rope application
            bias=False,
        )
        
        # Similarly for kv
        self.kv_down_proj = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_down_layernorm = RMSNorm(self.kv_lora_rank)
        self.kv_up_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (
                    self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim
            ), 
            bias=False,
        )
        
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        
        # Initialize rope parameters
        self.rotary_emb = RotaryEmbedding(
            self.qk_rope_head_dim,
            self.max_position_embeddings,
            self.rope_theta,
        )

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        """
        MLA (Multi-head Linearized Attention) forward pass
        """
        bsz, q_len, _ = hidden_states.size()
        
        # 1. Query projection and split
        q = self.q_up_proj(
            self.q_down_layernorm(
                self.q_down_proj(hidden_states)
            )
        )
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1
        )
        
        # 2. Key/Value projection and split
        compressed_kv = self.kv_down_proj(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_up_proj(self.kv_down_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )
        k_nope, value_states = torch.split(
            kv,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1
        )

        # 3. Apply RoPE to position-dependent parts
        kv_seq_len = value_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        # Final Q, K, V shapes should all be (batch_size, num_heads, seq_len, head_dim)
        # Where q/k head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # v head_dim = self.v_head_dim

        # 4. Combine position-dependent and independent parts
        query_states = torch.empty(
            bsz, self.num_heads, q_len, self.q_head_dim,
            device=k_pe.device
        )
        query_states[:, :, :, :self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim:] = q_pe

        key_states = torch.empty(
            bsz, self.num_heads, q_len, self.q_head_dim,
            device=k_pe.device
        )
        key_states[:, :, :, :self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim:] = k_pe

        # 5. Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = attn_weights / math.sqrt(self.q_head_dim)

        if attention_mask is not None:
            attn_weights = torch.masked_fill(
                attn_weights,
                attention_mask == 0,
                float("-inf"),
            )

        # 6. Softmax and dropout
        attn_weights = F.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(
            attn_weights, p=self.attention_dropout, training=self.training)

        # 7. Compute attention output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights