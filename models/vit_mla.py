import torch
import torch.nn as nn
import torch.nn.functional as F
from components import PatchEmbedding, MLA, MLAConfig

class ViT_MLA(nn.Module):
    def __init__(self, num_classes=100):
        super(ViT_MLA, self).__init__()
        # Patch embedding (4x4 patches, embed_dim=384)
        self.embed = PatchEmbedding(in_channels=3, patch_size=4, embed_dim=384)

        # Configure MLA (no RoPE in this version)
        config = MLAConfig(
            hidden_size=384,
            num_heads=6,
            max_position_embeddings=1024,
            rope_theta=1280,
            attention_dropout=0.0,
            q_lora_rank=1536,
            qk_rope_head_dim=0,  # No RoPE
            kv_lora_rank=512,
            v_head_dim=64,
            qk_nope_head_dim=64,
            attention_bias=False,
        )
        self.attn = MLA(config)

        self.mlp = nn.Sequential(
            nn.Linear(384 * 64, 768),
            nn.ReLU(),
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.attn(x)
        x = x.flatten(1)
        return self.mlp(x)