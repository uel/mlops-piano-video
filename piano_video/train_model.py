import torch
import hydra

from imagen_pytorch import Unet, Imagen, ImagenTrainer
from imagen_pytorch import ElucidatedImagen

from data.data import Dataset
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import wandb
import datetime

PROJECT_DIR = Path(__file__).parent.parent.resolve()

@hydra.main(version_base=None, config_path=os.path.join(PROJECT_DIR, 'piano_video', 'config'), config_name="config.yaml")
def main(cfg):
    
    dataset_name = "images_512"
    model_dir = "512_landmarks"

    # setting up paths
    dataset_folder = os.path.join(PROJECT_DIR, 'data', 'processed', dataset_name)
    results_folder = os.path.join(PROJECT_DIR, 'reports', datetime.datetime.now().strftime('%d_%H_%M_%S'))
    tb_log = os.path.join(results_folder, 'tb') # tensorboard log dir

    # initalizing tensorboard, wandb and hyperparameters
    hp = cfg.hyperparameters # hyperparameters loaded from the config file
    writer = SummaryWriter(log_dir=tb_log) # initializing wandb
    run = wandb.init() # initializing wandb
    
    # fixing seed
    torch.manual_seed(hp.seed)

    unet1 = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = 1,
        layer_attns = (False, False, False, False),
        layer_cross_attns = (False, False, False, False),
        text_embed_dim=126*3,
        cond_dim=126*3
    )

    unet2 = Unet(
        dim = 32,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = (2, 4, 8, 8),
        layer_attns = (False, False, False, False),
        layer_cross_attns = (False, False, False, False),
        text_embed_dim=126*3,
        cond_dim=126*3
    )

    imagen = ElucidatedImagen(
        unets = (unet1, unet2),
        image_sizes = (128, 512),
        text_embed_dim=126*3,
        cond_drop_prob = 0.1,
        num_sample_steps = (64, 32), # number of sample steps - 64 for base unet, 32 for upsampler (just an example, have no clue what the optimal values are)
        sigma_min = 0.002,           # min noise level
        sigma_max = (80, 160),       # max noise level, @crowsonkb recommends double the max noise level for upsampler
        sigma_data = 0.5,            # standard deviation of data distribution
        rho = 7,                     # controls the sampling schedule
        P_mean = -1.2,               # mean of log-normal distribution from which noise is drawn for training
        P_std = 1.2,                 # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn = 80,                # parameters for stochastic sampling - depends on dataset, Table 5 in apper
        S_tmin = 0.05,
        S_tmax = 50,
        S_noise = 1.003,
    )

    # imagen, which contains the unet above
    # imagen = Imagen(
    #     unets = [unet1, unet2],
    #     image_sizes = [128, 512],
    #     timesteps = [1024, 512],
    #     text_embed_dim=126*3
    # )

    if torch.cuda.is_available():
        imagen.cuda()

    if os.path.exists(os.path.join("models", model_dir, "model.pt")):
        trainer = ImagenTrainer(imagen, split_valid_from_train=True)
        trainer.load(os.path.join("models", model_dir, "model.pt"))
        print("Loaded model from file.")
    else:
        # setting up the trainer class in denoising_diffusion_pytorch
        trainer = ImagenTrainer(
            imagen = imagen,
            split_valid_from_train = True
        )

    dataset = Dataset(dataset_folder, image_size = 512)

    trainer.add_train_dataset(dataset, batch_size = 16)
    trainer.create_valid_iter()

    for unet in [2]:
        for i in range(5000):
            loss = trainer.train_step(unet_number = unet, max_batch_size = 4)
            print(f'loss/train: {loss}')
            writer.add_scalar('loss/train', loss, i) 
            run.log({"loss/train": loss})

            if i and not (i % 100):
                valid_loss = trainer.valid_step(unet_number = 1, max_batch_size = 4)
                print(f'valid loss: {valid_loss}')
                writer.add_scalar('loss/valid', valid_loss, i)
                run.log({"loss/valid": valid_loss})
            
            if not (i % 511) and trainer.is_main:
                _, valid_landmarks = next(trainer.valid_dl_iter)
                images = trainer.sample(batch_size = 1, return_pil_images = True, text_embeds=valid_landmarks[:4])
                # images[0].save(f'./sample-{i // 100}.png')
                run.log({"samples": [wandb.Image(image) for image in images]})
                
        os.makedirs(os.path.join(PROJECT_DIR, 'models', model_dir), exist_ok=True)
        trainer.save(os.path.join(PROJECT_DIR, 'models', model_dir, 'model.pt'))


    # trainer.create_valid_iter()
    # for i in range(10):
    #     _, valid_landmarks = next(trainer.valid_dl_iter)
    #     images = trainer.sample(text_embeds=valid_landmarks[:4], batch_size = 1, return_pil_images = True)
    #     wandb.log({"samples": [wandb.Image(image) for image in images]})
        # images[0].save(f'./sample-{i // 100}.png')


if __name__ == "__main__":
    main()
