from models.components import PatchEmbedding, MLA
import torch.nn as nn

class MLA_ViT(nn.Module):
    def __init__(self, config, in_channels, num_classes=10):
        super(MLA_ViT, self).__init__()
        self.fc1 = nn.Linear((32 // 4) ** 2 * 768, 768)  # 10 is the number of output classes
        self.embed = PatchEmbedding(in_channels, patch_size=4, embed_dim=768)
        self.fc2 = nn.Linear(768, num_classes)
        self.relu = nn.ReLU()
        self.lch = MLA(config)

    def forward(self, x):
        # Generate patch embeddings
        x = self.embed(x)
        # Process through MLA (Multi-head Linearized Attention)
        x, _ = self.lch(x)
        # Flatten x, removing all dimensions except batch_size
        x = x.flatten(1)  # Only flatten non-batch dimensions
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x