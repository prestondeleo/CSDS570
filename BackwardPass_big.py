import torch
import torch.nn as nn
import torch.nn.functional as F

class BackwardPass(nn.Module):
    def __init__(self, num_classes=10, image_size=28, T=1000, hidden_dim=64):
        super().__init__()
        self.image_size = image_size
        self.T = T

        self.conv1 = nn.Conv2d(3, hidden_dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim * 2)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        self.conv4 = nn.Conv2d(hidden_dim, 1, 3, padding=1)

        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.label_embedding = nn.Embedding(num_classes + 1, self.image_size * self.image_size)

        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, t, y):
        t = t.float() / self.T
        t_emb = t.view(-1, 1, 1, 1).expand_as(x)
        label_emb = self.label_embedding(y).view(-1, 1, self.image_size, self.image_size)

        x = torch.cat([x, t_emb, label_emb], dim=1)

        x1 = F.silu(self.bn1(self.conv1(x)))
        x2 = self.pool(x1)
        x2 = F.silu(self.bn2(self.conv2(x2)))

        x3 = self.upsample(x2)
        x3 = F.silu(self.bn3(self.conv3(x3)))

        x = x3 + x1
        x = self.conv4(x)

        return x
