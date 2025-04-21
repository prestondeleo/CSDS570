import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
import os
import matplotlib.pyplot as plt
import numpy as _

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'data')

class Encoder(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


# TODO can this be defined more as a Diffusion (higher fidelity)
# - Add a loss on how far same classes are
class Decoder(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld


def save_image_grid(original, reconstructed, epoch, output_dir='vae_outputs'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(5):
        axes[0, i].imshow(original[i].squeeze().cpu().numpy(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')

        axes[1, i].imshow(reconstructed[i].squeeze().cpu().numpy(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'epoch_{epoch+1}.png'))
    plt.close()


def main():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_data = datasets.MNIST(root=DATASET_DIR, train=True, transform=transform, download=True)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

    vae = VAE(latent_dim=20)
    optimizer = Adam(vae.parameters(), lr=1e-3)

    for epoch in range(10):
        vae.train()
        train_loss = 0
        for x, _ in train_loader:
            x = x
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(x)
            loss = vae_loss(recon_x, x, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        with torch.no_grad():
            sample_batch = next(iter(train_loader))[0][:5]
            recon_batch, _, _ = vae(sample_batch)
            save_image_grid(sample_batch, recon_batch, epoch)

        print(f"Epoch {epoch+1}, Loss: {train_loss / len(train_loader):.4f}")


if __name__ == '__main__':
    main()
