import torch
import torch.nn as nn
import torch.nn.functional as F
from components import PatchEmbedding, MLA, MLAConfig

class ViT_MLA_RoPE(nn.Module):
    def __init__(self, num_classes=100):
        super(ViT_MLA_RoPE, self).__init__()
        # Patch embedding (4x4 patches, embed_dim=384)
        self.embed = PatchEmbedding(in_channels=3, patch_size=4, embed_dim=384)

        # Configure MLA with Rotary Positional Encoding (RoPE)
        config = MLAConfig(
            hidden_size=384,
            num_heads=6,
            max_position_embeddings=1024,
            rope_theta=1280,
            attention_dropout=0.0,
            q_lora_rank=1536,
            qk_rope_head_dim=64,  # Apply RoPE to 64 dims
            kv_lora_rank=512,
            v_head_dim=64,
            qk_nope_head_dim=64,  # Rest of head dim uses no position encoding
            attention_bias=False,
        )
        self.attn = MLA(config)

        # MLP classifier head
        self.mlp = nn.Sequential(
            nn.Linear(384 * 64, 768),
            nn.ReLU(),
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        x = self.embed(x)
        # MLA will internally apply RoPE to q/k
        x, _ = self.attn(x)
        x = x.flatten(1)
        return self.mlp(x)