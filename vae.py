import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.optim.adam import Adam
import os
import matplotlib.pyplot as plt
import numpy as _
import torchvision

ROOT_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(ROOT_DIR, 'data')
os.makedirs(DATASET_DIR, exist_ok=True)
GENERATED_DIR = os.path.join(ROOT_DIR, 'generated')
os.makedirs(GENERATED_DIR, exist_ok=True)

# TODO do some operations in the latent space to make it more understandable
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(128 * 7 * 7, latent_dim)

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)
        self.deconv_stack = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(-1, 128, 7, 7)
        x = self.deconv_stack(x)
        return x

class VAE(nn.Module):
    def __init__(self, *, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.saved_dists = {}

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def save_distributions(self, data_loader):
        class_latents = {}
        self.eval()

        with torch.no_grad():
            for images, labels in data_loader:
                mu, _ = self.encoder(images)
                for latent, label in zip(mu, labels):
                    label = label.item()
                    if label not in class_latents:
                        class_latents[label] = []
                    class_latents[label].append(latent)

            for label in class_latents:
                if class_latents[label]:
                    latents = torch.stack(class_latents[label])
                    mean = torch.mean(latents, dim=0)
                    std = torch.std(latents, dim=0)
                    self.saved_dists[label] = {'mean': mean, 'std': std}

    def k_closest(self, unknown_sample: torch.Tensor, k: int):
        assert self.saved_dists is not None

        self.eval()
        with torch.no_grad():
            mu, _ = self.encoder(unknown_sample)
            if mu.dim() > 1:
                mu = mu.squeeze()

            distances = []
            for label, dist in self.saved_dists.items():
                mean = dist['mean']
                distance = torch.norm(mu - mean, p=2).item()
                distances.append((label, distance))

            distances.sort(key=lambda x: x[1])
        return distances[:k]

    def latent_distribution(self, k_closest_classes, sample, weight_of_sample):
        assert self.saved_dists is not None

        self.eval()
        with torch.no_grad():
            sample_mu, _ = self.encoder(sample)
            if sample_mu.dim() > 1:
                sample_mu = sample_mu.squeeze()

        labels, distances = zip(*k_closest_classes)
        distances = torch.tensor(distances) + 1e-8

        class_weights = 1.0 / distances
        class_weights /= class_weights.sum()

        means = torch.stack([self.saved_dists[label]['mean'] for label in labels])
        stds = torch.stack([self.saved_dists[label]['std'] for label in labels])

        combined_means = torch.cat([means, sample_mu.unsqueeze(0)], dim=0)

        sample_std = torch.mean(stds, dim=0)
        combined_stds = torch.cat([stds, sample_std.unsqueeze(0)], dim=0)

        combined_weights = torch.cat([class_weights, torch.tensor([weight_of_sample])])
        combined_weights /= combined_weights.sum()

        weighted_mean = torch.sum(combined_weights[:, None] * combined_means, dim=0)
        weighted_std = torch.sum(combined_weights[:, None] * combined_stds, dim=0)

        return weighted_mean, weighted_std

    def generate_samples(self, mean, std, num_samples):
        eps = torch.randn(num_samples, mean.size(0))
        z_samples = mean + eps * std

        self.eval()
        with torch.no_grad():
            generated = self.decoder(z_samples)

        return generated


def vae_loss(recon_x, x, mu, logvar):
    bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld

def save_image_grid(original, reconstructed, epoch, output_dir='vae_outputs'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    _, axes = plt.subplots(2, 5, figsize=(15, 6))
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
    num_epochs = 10
    unknown = 7

    full_train = datasets.MNIST(root=DATASET_DIR, train=True, transform=transform, download=True)
    indices = [i for i in range(len(full_train)) if full_train[i][1] != unknown]
    train_data = Subset(full_train, indices)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

    vae = VAE(latent_dim=20)
    optimizer = Adam(vae.parameters(), lr=1e-3)
    checkpoint_path = os.path.join(ROOT_DIR, 'vae_weights.pth')

    if os.path.exists(checkpoint_path):
        vae.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    else:
        for epoch in range(num_epochs):
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

            print(f'Epoch {epoch+1}, Loss: {train_loss / len(train_loader):.4f}')
    torch.save(vae.state_dict(), checkpoint_path)

    vae.eval()
    vae.save_distributions(train_loader)
    unknown_class_instance = None
    for d in full_train:
        if d[1] == unknown:
            unknown_class_instance = d
            break
    assert(unknown_class_instance is not None)
    k = 5
    num_samples = 20
    samp = unknown_class_instance[0].unsqueeze(0)
    k_closest_classes = vae.k_closest(samp, k)
    for c in k_closest_classes:
        print(f'label: {c[0]} | dist {c[1]}')
    mean, std = vae.latent_distribution(k_closest_classes, samp, 0.5)
    samples = vae.generate_samples(mean, std, num_samples)
    torchvision.utils.save_image(samples, os.path.join(GENERATED_DIR, f'{unknown}.png'),
                                 nrow=num_samples // 5)

if __name__ == '__main__':
    main()
