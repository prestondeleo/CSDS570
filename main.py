import torch
import sys
import torchvision
from diffusion_model import cond_samp_CFG
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
    vae_path = os.path.join(MODEL_DIR, f'vae_weights_{unknown}.pth')
    dif_path = os.path.join(MODEL_DIR, f'dif_{unknown}.pth')
    assert os.path.exists(vae_path)
    assert os.path.exists(dif_path)

    vae.load_state_dict(torch.load(vae_path, weights_only=True, map_location=torch.device('cpu')))
    dif.load_state_dict(torch.load(dif_path, weights_only=True, map_location=torch.device('cpu')))
    print(1)
    samples = cond_samp_CFG(dif, (10, 1, 28, 28), class_token=2, constantw_scale=20)
    print(2)
    samples = (samples.clamp(-1, 1) + 1) / 2
    torchvision.utils.save_image(samples, os.path.join(GENERATED_DIR, f'{unknown}_dif.png'),
                                 nrow=len(samples) // 5)

if __name__ == '__main__':
    assert len(sys.argv) <= 2
    unknown = 0
    if len(sys.argv) == 2:
        unknown = int(sys.argv[1])
    main(unknown)
