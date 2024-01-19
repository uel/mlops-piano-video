from piano_video.train_model import Unet, GaussianDiffusion, Trainer
from hydra import initialize, compose
from unittest.mock import MagicMock
from tests import _PROJECT_ROOT
import torch
import os
import pytest

def test_model_instantiation():
    # test model instantiaion with confing.yaml hyperparameters
    with initialize(version_base=None, config_path='../piano_video/config'):
        cfg = compose(config_name='config')
        try:
            # define U-net backbone of the DDPM
            model = Unet(
                dim=cfg.hyperparameters.dim,
                dim_mults=tuple(cfg.hyperparameters.dim_mults),
                flash_attn=cfg.hyperparameters.flash_attn
            )

            # define the DDPM itself
            diffusion = GaussianDiffusion(
                model,
                image_size=cfg.hyperparameters.image_size,
                timesteps=cfg.hyperparameters.timesteps,
                sampling_timesteps=cfg.hyperparameters.sampling_timesteps
            )
        except Exception as e:
            pytest.fail(f'Model instantiation failed with error: {str(e)}')

@pytest.mark.skipif(not os.path.exists(os.path.join(_PROJECT_ROOT, 'data', 'processed', 'images_small')), reason='Data files not found')
def test_trainer_instantiation():
    # create mock GaussianDiffusion object to serve as a parameter for Trainer
    mock_diffusion = MagicMock(spec=GaussianDiffusion)
    mock_diffusion.parameters.return_value =  iter([torch.tensor(0)])
    mock_diffusion.image_size = 64
    mock_diffusion.is_ddim_sampling = True
    mock_diffusion.channels = 3

    # test Trainer instantiaion with confing.yaml hyperparameters
    with initialize(version_base=None, config_path='../piano_video/config'):
        cfg = compose(config_name='config')
        try:
            trainer = Trainer(
                mock_diffusion,
                os.path.join(_PROJECT_ROOT, 'data', 'processed', 'images_small'),
                train_batch_size = cfg.hyperparameters.train_batch_size,
                train_lr = cfg.hyperparameters.train_lr,
                train_num_steps = cfg.hyperparameters.train_num_steps,
                gradient_accumulate_every = cfg.hyperparameters.gradient_accumulate_every,
                ema_decay = cfg.hyperparameters.ema_decay,
                amp = cfg.hyperparameters.amp,
                calculate_fid = cfg.hyperparameters.calculate_fid,
                results_folder = os.path.join(_PROJECT_ROOT, 'results'),
                num_fid_samples = cfg.hyperparameters.num_fid_samples,
                save_and_sample_every = cfg.hyperparameters.save_and_sample_every
            )
        except Exception as e:
            pytest.fail(f'Trainer instantiation failed with error: {str(e)}')

        assert isinstance(trainer, Trainer)

@pytest.mark.skipif(not os.path.exists(os.path.join(_PROJECT_ROOT, 'data', 'processed', 'images_small')), reason='Data files not found')
def test_training_loop():
    with initialize(version_base=None, config_path='../piano_video/config'):
        cfg = compose(config_name='config')
        try:
            # define U-net backbone of the DDPM
            model = Unet(
                dim=cfg.hyperparameters.dim,
                dim_mults=tuple(cfg.hyperparameters.dim_mults),
                flash_attn=cfg.hyperparameters.flash_attn
            )

            # define the DDPM itself
            diffusion = GaussianDiffusion(
                model,
                image_size=cfg.hyperparameters.image_size,
                timesteps=cfg.hyperparameters.timesteps,
                sampling_timesteps=cfg.hyperparameters.sampling_timesteps
            )
            initial_parameters = diffusion.parameters()

            # define Trainer object
            trainer = Trainer(
                diffusion,
                os.path.join(_PROJECT_ROOT, 'data', 'processed', 'images_small'),
                train_batch_size = cfg.hyperparameters.train_batch_size,
                train_lr = cfg.hyperparameters.train_lr,
                train_num_steps = cfg.hyperparameters.train_num_steps,
                gradient_accumulate_every = cfg.hyperparameters.gradient_accumulate_every,
                ema_decay = cfg.hyperparameters.ema_decay,
                amp = cfg.hyperparameters.amp,
                calculate_fid = cfg.hyperparameters.calculate_fid,
                results_folder = os.path.join(_PROJECT_ROOT, 'results'),
                num_fid_samples = cfg.hyperparameters.num_fid_samples,
                save_and_sample_every = cfg.hyperparameters.save_and_sample_every
            )

            # test one step of the training loop
            accelerator = trainer.accelerator
            device = accelerator.device
            for _ in range(trainer.gradient_accumulate_every):
                data = next(trainer.dl).to(device)

                with trainer.accelerator.autocast():
                    loss = trainer.model(data)
                    loss = loss / trainer.gradient_accumulate_every

                trainer.accelerator.backward(loss)

            trainer.opt.step()
            trainer.opt.zero_grad()

            torch.save(diffusion, os.path.join(_PROJECT_ROOT, 'models', 'test_model.pt'))
        except Exception as e:
            pytest.fail(f'Training loop test failed with error: {str(e)}')

        assert diffusion.parameters() != initial_parameters, 'Model parameters were not updated'

def test_model_inference():
    with initialize(version_base=None, config_path='../piano_video/config'):
        cfg = compose(config_name='config')
        try:
            diffusion = torch.load(os.path.join(_PROJECT_ROOT, 'models', 'test_model.pt')) # load trained DDPM
            sample = diffusion.sample(batch_size=1) # generate images

        except Exception as e:
            pytest.fail(f'Model inference test failed with error: {str(e)}')

        image_size = cfg.hyperparameters.image_size
        assert list(sample[0].shape) == [3, image_size, image_size], 'Output image has incorrect shape'