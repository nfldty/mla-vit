import torch.nn as nn
import torch
class ViT(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_classes=10, embed_dim=768, num_heads=1, num_layers=1):
        super(ViT, self).__init__()
        
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size
        self.linear_embedding = nn.Linear(self.patch_dim, embed_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,  # FFN dimension in standard transformer
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)

        # Patch the image
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, self.num_patches, self.patch_dim)

        x = self.linear_embedding(patches) + self.position_embedding

        x = self.transformer_encoder(x)  # shape: (B, N, D)

        x = x.mean(dim=1)  # global average pooling over tokens

        return self.fc(x)

