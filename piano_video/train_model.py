import torch
from torchvision import utils
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import num_to_groups, divisible_by
import hydra
import os
import math
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb
import datetime

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

@hydra.main(version_base=None, config_path=os.path.join(FILE_DIR, './config/'), config_name="config.yaml")
def main(cfg):
    
    # seeting up paths
    dataset_folder = os.path.join(FILE_DIR, '../data/processed/images_small')
    results_folder = os.path.join(FILE_DIR, '../reports', datetime.datetime.now().strftime('%d:%H-%M-%S'))
    tb_log = os.path.join(results_folder, 'tb') # tensorboard log dir

    # initalizing tensorboard, wandb and hyperparameters
    hp = cfg.hyperparameters # hyperparameters loaded from the config file
    writer = SummaryWriter(log_dir=tb_log) # initializing wandb
    run = wandb.init() # initializing wandb
    
    # fixing seed
    torch.manual_seed(hp.seed)

    # define U-net backbone of the DDPM
    model = Unet(
        dim=cfg.hyperparameters.dim,
        dim_mults=tuple(cfg.hyperparameters.dim_mults),
        flash_attn=cfg.hyperparameters.flash_attn
    )

    # define the DDPM itself
    diffusion = GaussianDiffusion(
        model,
        image_size=hp.image_size,
        timesteps=hp.timesteps,
        sampling_timesteps=hp.sampling_timesteps
    )

    # setting up the trainer class in denoising_diffusion_pytorch
    trainer = Trainer(
            diffusion,
            dataset_folder,
            train_batch_size = hp.train_batch_size,
            train_lr = hp.train_lr,
            train_num_steps = hp.train_num_steps,
            gradient_accumulate_every = hp.gradient_accumulate_every,
            ema_decay = hp.ema_decay,
            amp = hp.amp,
            calculate_fid = hp.calculate_fid,
            results_folder = results_folder,
            num_fid_samples = hp.num_fid_samples,
            save_and_sample_every = hp.save_and_sample_every
        )

    # explicit model training to be able to log progress
    if cfg.logging.log_on:
        accelerator = trainer.accelerator
        device = accelerator.device

        with tqdm(initial = trainer.step, total = trainer.train_num_steps, disable = not accelerator.is_main_process) as pbar:
            while trainer.step < trainer.train_num_steps:

                total_loss = 0.
                for _ in range(trainer.gradient_accumulate_every):
                    data = next(trainer.dl).to(device)

                    with trainer.accelerator.autocast():
                        loss = trainer.model(data)
                        loss = loss / trainer.gradient_accumulate_every
                        total_loss += loss.item()

                    trainer.accelerator.backward(loss)

                writer.add_scalar('Loss/train', total_loss, trainer.step) # logging loss with tensorboard
                run.log({"loss/train":total_loss}) # logging loss with wandb

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(trainer.model.parameters(), trainer.max_grad_norm)

                trainer.opt.step()
                trainer.opt.zero_grad()

                accelerator.wait_for_everyone()

                trainer.step += 1
                if accelerator.is_main_process:
                    trainer.ema.update()

                    if trainer.step != 0 and divisible_by(trainer.step, trainer.save_and_sample_every):
                        trainer.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = trainer.step // trainer.save_and_sample_every
                            batches = num_to_groups(trainer.num_samples, trainer.batch_size)
                            all_images_list = list(map(lambda n: trainer.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim = 0)

                        utils.save_image(all_images, str(trainer.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(trainer.num_samples)))

                        # whether to calculate fid

                        if trainer.calculate_fid:
                            fid_score = trainer.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')
                        if trainer.save_best_and_latest_only:
                            if trainer.best_fid > fid_score:
                                trainer.best_fid = fid_score
                                trainer.save("best")
                            trainer.save("latest")
                        else:
                            trainer.save(milestone)

                pbar.update(1)

            run.finish()
        accelerator.print('training complete')

    # training using the Trainer class for no logging
    else:
        trainer.train()

    # saving model
    torch.save(diffusion, os.path.join(FILE_DIR, './models/diffusion_model.pt'))

if __name__ == "__main__":
    main()
