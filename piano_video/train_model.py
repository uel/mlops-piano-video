import torch
import hydra

from imagen_pytorch import Unet, Imagen, ImagenTrainer

from data.data import Dataset
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import wandb
import datetime

PROJECT_DIR = Path(__file__).parent.parent.resolve()

@hydra.main(version_base=None, config_path=os.path.join(PROJECT_DIR, 'piano_video', 'config'), config_name="config.yaml")
def main(cfg):
    
    dataset_name = "images_128"
    model_dir = "128_landmarks"

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

    if os.path.exists(os.path.join("models", model_dir, "model.pt")):
        trainer = ImagenTrainer(imagen_checkpoint_path=os.path.join("models", model_dir, "model.pt"), split_valid_from_train=True)
        print("Loaded model from file.")
    else:
        # unets for unconditional imagen
        unet = Unet(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            num_resnet_blocks = 1,
            layer_attns = (False, True, True, True),
            layer_cross_attns = (False, True, True, True),
            text_embed_dim=126,
            cond_dim=126
        )

        # imagen, which contains the unet above
        imagen = Imagen(
            unets = unet,
            image_sizes = 128,
            timesteps = 1000,
            text_embed_dim=126
        )

        if torch.cuda.is_available():
            imagen.cuda()

        # setting up the trainer class in denoising_diffusion_pytorch
        trainer = ImagenTrainer(
            imagen = imagen,
            split_valid_from_train = True
        )

    dataset = Dataset(dataset_folder, image_size = 128)

    trainer.add_train_dataset(dataset, batch_size = 32)

    validation_landmarks = None

    # for i in range(5000):
    #     loss = trainer.train_step(unet_number = 1, max_batch_size = 4)
    #     print(f'loss/train: {loss}')
    #     writer.add_scalar('loss/train', loss, i) 
    #     run.log({"loss/train": loss})

    #     if i and not (i % 100):
    #         valid_loss = trainer.valid_step(unet_number = 1, max_batch_size = 4)
    #         print(f'valid loss: {valid_loss}')
    #         writer.add_scalar('loss/valid', valid_loss, i)
    #         run.log({"loss/valid": valid_loss})

    for i in range(10):
        _, valid_landmarks = next(trainer.valid_dl_iter)
        images = trainer.sample(text_embeds=valid_landmarks, batch_size = 1, return_pil_images = True)
        wandb.log({"samples": [wandb.Image(image) for image in images]})
        # images[0].save(f'./sample-{i // 100}.png')

    # saving model
    # os.makedirs(os.path.join(PROJECT_DIR, 'models', model_dir), exist_ok=True)
    # trainer.save(os.path.join(PROJECT_DIR, 'models', model_dir, 'model.pt'))

if __name__ == "__main__":
    main()
