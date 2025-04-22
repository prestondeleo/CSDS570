import torch
import sys
from torch import _has_compatible_shallow_copy_type
from BackwardPass import BackwardPass as Diffusion
from vae import VAE
from dirs import *

def main(unknown: int):
    T = 1000
    img_size = 28
    num_classes = 10
    latent_vae = 20
    null_token = num_classes

    dif = Diffusion(num_classes, img_size, T)
    vae = VAE(latent_dim=latent_vae)
    vae_path = f'vae_weights_{unknown}.pth'
    dif_path = f'dif_{unknown}.pth'
    assert os.path.exists(os.path.join(MODEL_DIR, dif_path))
    assert os.path.exists(os.path.join(MODEL_DIR, vae_path))

    vae.load_state_dict(torch.load(vae_path, weights_only=True))
    dif.load_state_dict(torch.load(dif_path, weights_only=True))

if __name__ == '__main__':
    assert len(sys.argv) <= 2
    unknown = 0
    if len(sys.argv) == 2:
        unknown = int(sys.argv[1])
    main(unknown)
