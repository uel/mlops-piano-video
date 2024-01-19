import torch
import hydra
from denoising_diffusion_pytorch import GaussianDiffusion, Unet
from torchvision.utils import save_image
import os

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PT = os.path.join(FILE_DIR, 'models', 'diffusion_model.pt')

@hydra.main(version_base=None, config_path="./config", config_name='config.yaml')
def generate(cfg):
    hp = cfg.hyperparameters # hyperparameters loaded from the config file
    num_images = hp.image_nos_to_generate # number of images to generate

    diffusion = torch.load(MODEL_PT) # load trained DDPM
    sample = diffusion.sample(batch_size=num_images) # generate images
    
    # save generated images as PNGs
    for img_num, img in enumerate(sample):
        save_image(img, os.path.join(FILE_DIR, f"./visualizations/img{img_num}.png"))

if __name__ == '__main__':
    generate()
    