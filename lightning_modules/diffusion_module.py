import os.path
from .loss import get_loss_funcs
from lightning import LightningModule
from diffusers import make_diffusion_scheduler
from model import make_model
from utils.torch_utils import randn_tensor
from lightning_modules import make_optimizer, make_scheduler
import numpy as np
import torch


class DiffusionModule(LightningModule):
    def __init__(self, train_cfg, scheduler_cfg, model_cfg, loss_cfg=None, guidance_on=None, mask_cond_prob=0.1, guidance_scale=2.5, inference_steps=1000, inference_timesteps=None):
        super().__init__()
        self.train_cfg = train_cfg
        if loss_cfg is not None:
            self.loss_funcs = get_loss_funcs(loss_cfg)
            self.loss_weight = {key: loss_cfg[key]['weight'] for key in loss_cfg}
        else:
            self.loss_funcs = None
        self.num_train_timesteps = scheduler_cfg.kwargs['num_train_timesteps']
        self.scheduler = make_diffusion_scheduler(scheduler_cfg)
        self.denoiser = make_model(model_cfg)
        self.guidance_on = guidance_on
        if isinstance(self.guidance_on, str):
            self.guidance_on = [self.guidance_on]
        self.save_hyperparameters()
        self.mean = None
        self.std = None

    def training_step(self, batch, batch_idx):
        data, conditions = batch
        conditions = self.mask_conditions(conditions)
        x_0 = data
        loss, sample = self._diffusion_train(x_0, conditions)
        self.log('diffusion_loss', loss)

        return loss

    def _diffusion_train(self, data, conditions, **kwargs):
        noise = torch.randn_like(data)
        timesteps = torch.randint(0, self.num_train_timesteps, (data.shape[0],), device=data.device)
        timesteps = timesteps.long()
        noisy_data = self.scheduler.add_noise(data, noise, timesteps)

        model_output = self.denoiser(noisy_data, conditions, timesteps, **kwargs['model_args'])
        simple_loss = None
        sample = None
        if self.scheduler.config.prediction_type == 'epsilon':
            simple_loss = self.loss_funcs['diffusion_loss'](model_output, noise, **kwargs['loss_args'])
            sample = self.scheduler.predict_x_start_from_epsilon(noisy_data, model_output, timesteps)
        elif self.scheduler.config.prediction_type == 'sample':
            simple_loss = self.loss_funcs['diffusion_loss'](model_output, data, **kwargs['loss_args'])
            sample = model_output

        return simple_loss, sample

    def diffuse(self, noise, conditions, model_args, **kwargs):
        nil_conditions = self.mask_conditions(conditions, force_mask=True)
        self.scheduler.set_timesteps(self.hparams.inference_steps, noise.device)
        bs = noise.shape[0]
        timesteps = self.scheduler.timesteps
        x_t = noise
        for i, t in enumerate(timesteps):
            t = t.expand(bs)
            model_output = self.denoiser(x_t, conditions, t, **model_args)
            if not kwargs.get('uncond', False):
                no_cond_output = self.denoiser(x_t, nil_conditions, t, **model_args)
                model_output = no_cond_output + self.hparams.guidance_scale * (model_output - no_cond_output)
            res = self.scheduler.step(model_output, t[0], x_t, return_dict=True, **kwargs['scheduler_args'])
            x_t = res.prev_sample
        return x_t

    def inpaint(self, src_sample, inpaint_mask, conditions, model_args):
        nil_conditions = self.mask_conditions(conditions, force_mask=True)
        self.scheduler.set_timesteps(self.hparams.inference_steps, src_sample.device)
        bs = src_sample.shape[0]
        timesteps = self.scheduler.timesteps
        noise = randn_tensor(src_sample.shape, device=src_sample.device)
        x_t = noise.clone()
        for i, t in enumerate(timesteps):
            t = t.expand(bs)
            noised_src = self.scheduler.add_noise(src_sample, noise, t)
            x_t = noised_src * (1. - inpaint_mask) + x_t * inpaint_mask
            model_output = self.denoiser(x_t, conditions, t, **model_args)
            no_cond_output = self.denoiser(x_t, nil_conditions, t, **model_args)
            model_output = no_cond_output + self.hparams.guidance_scale * (model_output - no_cond_output)
            prev_x = self.scheduler.step(model_output, t, x_t, return_dict=False)
            x_t = prev_x[0]
        # x_t = src_sample * (1. - inpaint_mask) + x_t * inpaint_mask
        return x_t

    def configure_optimizers(self):
        optimizer = make_optimizer(self.train_cfg, self.denoiser)
        scheduler = make_scheduler(self.train_cfg.scheduler, optimizer)
        return [optimizer], [scheduler]

    def zscore_normalize(self, data):
        mean = torch.tensor(self.mean, device=data.device, dtype=data.dtype)
        std = torch.tensor(self.std, device=data.device, dtype=data.dtype)
        return (data - mean) / std

    def zscore_denormalize(self, data):
        mean = torch.tensor(self.mean, device=data.device, dtype=data.dtype)
        std = torch.tensor(self.std, device=data.device, dtype=data.dtype)
        return data * std + mean

    def mask_conditions(self, conditions, force_mask=False):
        masked_conditions = conditions.copy()
        bs = list(conditions.values())[0].shape[0]
        device = list(conditions.values())[0].device
        if force_mask:
            mask = torch.ones(bs, device=device)
        else:
            mask = torch.bernoulli(torch.ones(bs, device=device) * self.hparams.mask_cond_prob)
        # 1-> use null_cond, 0-> use real cond

        for key in conditions:
            if key in self.guidance_on:
                new_mask_shape = [-1] + [1] * (len(conditions[key].shape)-1)
                masked_conditions[key] = conditions[key] * (1. - mask).view(*new_mask_shape)

        return masked_conditions

    def load_stats(self, path, prefix):
        self.mean = np.load(os.path.join(path, f'{prefix}mean.npy'))
        self.std = np.load(os.path.join(path, f'{prefix}std.npy'))
        # self.std[np.where(self.std < 1e-4)] = 1e-4


