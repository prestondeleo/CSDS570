import torch
import sys
import torchvision
from diffusion_model import cond_samp_CFG
from BackwardPass import BackwardPass as Diffusion
from vae import VAE
from dirs import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def main(unknown: int):
    T = 1000
    img_size = 28
    num_classes = 10
    latent_vae = 20
    null_token = num_classes

    dif = Diffusion(num_classes, img_size, T)
    vae = VAE(latent_dim=latent_vae)
    vae_path = os.path.join(MODEL_DIR, f'vae_weights_{unknown}.pth')
    dif_path = os.path.join(MODEL_DIR, f'dif_{unknown}.pth')
    assert os.path.exists(vae_path)
    assert os.path.exists(dif_path)

    vae.load_state_dict(torch.load(vae_path, weights_only=True, map_location=torch.device('cpu')))
    dif.load_state_dict(torch.load(dif_path, weights_only=True, map_location=torch.device('cpu')))

    for u in torch.linspace(0, 10, 100):
        samples = cond_samp_CFG(dif, (5, 1, 28, 28), class_token=u.item(), constantw_scale=20)
        samples = (samples.clamp(-1, 1) + 1) / 2
        torchvision.utils.save_image(samples, os.path.join(GENERATED_DIR, f'{u.item()}_dif.png'),
                                     nrow=len(samples) // 5)

    return

    k = 5
    full_train = datasets.MNIST(root=DATASET_DIR, train=True, transform=transforms.Compose([
        transforms.ToTensor()
        ]), download=True)
    indices = [i for i in range(len(full_train)) if full_train[i][1] != unknown]
    train_data = Subset(full_train, indices)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    vae.eval()
    vae.save_distributions(train_loader)

    num_samples = 20
    unknown_class_instance = None
    for d in full_train:
        if d[1] == unknown:
            unknown_class_instance = d
            break
    assert(unknown_class_instance is not None)
    samp = unknown_class_instance[0].unsqueeze(0)
    k_closest_classes = vae.k_closest(samp, k)
    for c in k_closest_classes:
        print(f'label: {c[0]} | dist: {c[1]}')



if __name__ == '__main__':
    assert len(sys.argv) <= 2
    unknown = 0
    if len(sys.argv) == 2:
        unknown = int(sys.argv[1])
    main(unknown)
