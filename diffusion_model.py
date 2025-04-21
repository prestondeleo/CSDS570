import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import BackwardPass
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
device = torch.device("mps" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    # from 0,1 to -1,1
    transforms.Lambda(lambda x: x * 2 - 1)
])

T = 1000 
img_size = 28
num_classes = 10
null_token = num_classes 

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'data')
dataset = datasets.MNIST(root=DATASET_DIR, train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
