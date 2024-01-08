import torch
import sys
from denoising_diffusion_pytorch import GaussianDiffusion, Unet
from torchvision.utils import save_image
import os

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    model_path = sys.argv[1]
    num_images = int(sys.argv[2]) # number of images to generate

    diffusion = torch.load(model_path) # load trained DDPM
    sample = diffusion.sample(batch_size=num_images) # generate images
    
    # save generated images as PNGs
    for img_num, img in enumerate(sample):
        save_image(img, os.path.join(FILE_DIR, f"./visualizations/img{img_num}.png"))