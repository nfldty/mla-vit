import torch
import torch.nn as nn

class ViT_MHA(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_classes=100, embed_dim=384, num_heads=6, num_layers=6):
        super(ViT_MHA, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size
        self.linear_embedding = nn.Linear(self.patch_dim, embed_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=768, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, self.num_patches, -1)
        x = self.linear_embedding(patches) + self.position_embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer_encoder(x)
        return self.fc(x[:, 0])