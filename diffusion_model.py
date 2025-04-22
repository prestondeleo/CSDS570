import torch
import torch.nn as nn
import torch.nn.functional as F
from dirs import *
import numpy as np
import matplotlib.pyplot as plt
import os
import BackwardPass
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader, Subset
import sys


T = 1000
img_size = 28
num_classes = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
null_token = num_classes
betas = torch.linspace(0.0001, 0.02, T)
alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, dim=0)


@torch.no_grad()
def cond_samp_CFG(model, shape, class_token=None, constantw_scale=2.0):
    x = torch.randn(shape, device=device)
    y = torch.full((shape[0],), class_token if class_token is not None else null_token, device=device, dtype=torch.long)
    y_null = torch.full_like(y, null_token)

    for t in reversed(range(T)):
        t_batch = torch.tensor([t] * shape[0], device=device)
        epsilon_unconditional = model(x, t_batch, y_null)
        epsilon_conditional = model(x, t_batch, y)
        epsilon = (1 + constantw_scale) * epsilon_conditional - constantw_scale * epsilon_unconditional

        beta = betas[t]
        alpha = alphas[t]
        alpha_cumprod = alphas_prod[t]

        one_over_sqrt_alpha = 1 / torch.sqrt(alpha)
        second_ddpm_sample_term = (1 - alpha) / torch.sqrt(1 - alpha_cumprod)
        mean = one_over_sqrt_alpha * (x - second_ddpm_sample_term * epsilon)
        if t > 0:
            z = torch.randn_like(x)
        else:
            z = torch.zeros_like(x)
        sigma = torch.sqrt(beta)
        x = mean + sigma * z
    return x

if __name__ == '__main__':
    assert len(sys.argv) == 2

    unknown = int(sys.argv[1])

    transform = transforms.Compose([
        transforms.ToTensor(),
        # from 0,1 to -1,1
        transforms.Lambda(lambda x: x * 2 - 1)
    ])

    full_train = datasets.MNIST(root=DATASET_DIR, train=True, transform=transform, download=True)
    indices = [i for i in range(len(full_train)) if full_train[i][1] != unknown]
    train_data = Subset(full_train, indices)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

    def get_alpha(a, t, x_shape):
        if a.device != t.device:
            a = a.to(t.device)
        return a.gather(-1, t).reshape(-1, 1, 1, 1).expand(x_shape)

    def forward_process_q(x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_prod = get_alpha(torch.sqrt(alphas_prod), t, x_0.shape)
        sqrt_one_minus_alpha_prod = get_alpha(torch.sqrt(1 - alphas_prod), t, x_0.shape)
        return sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise


    def train(model, dataloader, optimizer, epochs=1):
        model.train()
        for epoch in range(epochs):
            for x, y in dataloader:
                x = x.to(device)
                t = torch.randint(0, T, (x.size(0),), device=device).long()

                # CFG by taking null token with some probability
                if torch.rand(1).item() < 0.3:
                    y = torch.full_like(y, null_token)
                y = y.to(device)

                noise = torch.randn_like(x)
                x_noisy = forward_process_q(x, t, noise=noise)
                pred = model(x_noisy, t, y)
                loss = F.mse_loss(pred, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    ###Good
    model = BackwardPass.BackwardPass(num_classes=num_classes, image_size = 28, T = 1000).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model, train_loader, optimizer, epochs=20)
    ###Good

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'dif_bigger_{unknown}.pth'))

    samples = cond_samp_CFG(model, (16, 1, 28, 28), class_token=7, constantw_scale=7.5)  # Generate 7s from diffusion model
    samples = (samples.clamp(-1, 1) + 1) / 2
