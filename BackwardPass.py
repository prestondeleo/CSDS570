import torch
import torch.nn as nn
import torch.nn.functional as F

class BackwardPass(nn.Module):
    def __init__(self, num_classes=10, image_size = 28, T = 100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 1, 3, padding=1)
        self.image_size = image_size
        self.label_embedding = nn.Embedding(num_classes + 1, self.image_size * self.image_size)
        self.T = T

    def forward(self, input, t, y):
        t = t.float() / self.T
        t = t.view(-1, 1, 1, 1).expand_as(input)
        label_embedding = self.label_embedding(y).view(-1, 1, self.image_size, self.image_size)
        output = torch.cat([input, t, label_embedding], dim=1)
        output = F.relu(self.conv1(output))
        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))
        return self.conv4(output)