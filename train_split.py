import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms 
import torchvision.datasets as datasets
import numpy as np
import torch.optim as optim
import torch.nn as nn
import pandas as pd
#import os
#import argparse
#from PIL import Image
import matplotlib.pyplot as plt
#import math
from torch.utils.data import Subset

transform = transforms.Compose([ transforms.ToTensor(),
transforms.Normalize(mean=[0.5], std=[0.5])])
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader= torch.utils.data.DataLoader(trainset, batch_size=16,
                                        shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                    download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                        shuffle=False, num_workers=2)
####THE TARGET CLASS (Hypothetical one image)
target_class = 3

# Get indices of only that class
train_indices_target = torch.where(torch.tensor(trainset.targets) == target_class)[0]
test_indices_target = torch.where(torch.tensor(testset.targets) == target_class)[0]

# Create subsets for that class
trainset_target = Subset(trainset, train_indices_target)
testset_target = Subset(testset, test_indices_target)

# Create data loaders from subsets
trainloader_target = torch.utils.data.DataLoader(trainset_target, batch_size=16, shuffle=True, num_workers=2)
testloader_target = torch.utils.data.DataLoader(testset_target, batch_size=16, shuffle=False, num_workers=2)

# Get indices of rest of data
train_indices_data = torch.where(torch.tensor(trainset.targets) != target_class)[0]
test_indices_data = torch.where(torch.tensor(testset.targets) != target_class)[0]

# Create subsetsof rest of data
trainset_data = Subset(trainset, train_indices_data)
testset_data = Subset(testset, test_indices_data)

# Create data loaders from subsets
trainloader_data = torch.utils.data.DataLoader(trainset_data, batch_size=16, shuffle=True, num_workers=2)
testloader_data = torch.utils.data.DataLoader(testset_data, batch_size=16, shuffle=False, num_workers=2)
