import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import hydra
import os

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

@hydra.main(config_name='config.yaml')
def main(cfg):
    hp = cfg.hyperparameters # hyperparameters loaded from the config file

    # define U-net backbone of the DDPM
    model = Unet(
        dim = cfg.hyperparameters.dim,
        dim_mults = tuple(cfg.hyperparameters.dim_mults),
        flash_attn = cfg.hyperparameters.flash_attn
    )

    # define the DDPM itself
    diffusion = GaussianDiffusion(
        model,
        image_size = hp.image_size,
        timesteps = hp.timesteps,
        sampling_timesteps = hp.sampling_timesteps
    )

    # create DDPM trainer
    trainer = Trainer(
        diffusion,
        os.path.join(FILE_DIR, '../data/processed/images_small'),
        train_batch_size = hp.train_batch_size,
        train_lr = hp.train_lr,
        train_num_steps = hp.train_num_steps,
        gradient_accumulate_every = hp.gradient_accumulate_every,
        ema_decay = hp.ema_decay,
        amp = hp.amp,
        calculate_fid = hp.calculate_fid,
        results_folder = os.path.join(FILE_DIR, '../reports'),
        num_fid_samples = hp.num_fid_samples,
        save_and_sample_every = hp.save_and_sample_every
    )

    trainer.train()

    torch.save(diffusion, os.join(FILE_DIR, './models/diffusion_model.pt'))

if __name__ == "__main__":
    main()