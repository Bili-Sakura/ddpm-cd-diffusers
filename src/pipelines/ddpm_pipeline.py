"""
DDPM pipeline using HuggingFace diffusers.

The GaussianDiffusion class wraps the custom SR3-style UNet and integrates
with diffusers.DDPMScheduler for noise scheduling.
"""

import logging
import math
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from diffusers import DDPMScheduler

from src.models.base_model import BaseModel
import src.models.networks as networks

logger = logging.getLogger('base')


def exists(x):
    return x is not None


def default(val, d):
    return val if exists(val) else (d() if callable(d) else d)


def noise_like(shape, device, repeat=False):
    def repeat_noise():
        return torch.randn((1, *shape[1:]), device=device).repeat(
            shape[0], *((1,) * (len(shape) - 1))
        )

    def noise():
        return torch.randn(shape, device=device)

    return repeat_noise() if repeat else noise()


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion model that uses diffusers.DDPMScheduler for noise scheduling.

    The UNet is used both for training (denoising) and as a feature extractor
    for the change detection head.
    """

    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        self.scheduler = None
        self.num_timesteps = 0
        # sqrt_alphas_cumprod_prev is kept for SR3-style continuous feature extraction
        self.sqrt_alphas_cumprod_prev = None

        if schedule_opt is not None:
            pass  # will be set via set_new_noise_schedule

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        """
        Configure the noise schedule using diffusers.DDPMScheduler.
        Replaces the manual beta schedule computation with the diffusers implementation.
        """
        beta_schedule = schedule_opt['schedule']
        # diffusers uses 'squaredcos_cap_v2' for cosine; map accordingly
        diffusers_schedule_map = {
            'linear': 'linear',
            'cosine': 'squaredcos_cap_v2',
            'quad': 'linear',   # fallback
            'warmup10': 'linear',
            'warmup50': 'linear',
            'const': 'linear',
            'jsd': 'linear',
        }
        diffusers_schedule = diffusers_schedule_map.get(beta_schedule, 'linear')

        self.scheduler = DDPMScheduler(
            num_train_timesteps=schedule_opt['n_timestep'],
            beta_start=schedule_opt['linear_start'],
            beta_end=schedule_opt['linear_end'],
            beta_schedule=diffusers_schedule,
            clip_sample=False,
        )
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        self.num_timesteps = schedule_opt['n_timestep']

        # Compute sqrt_alphas_cumprod_prev for SR3-style continuous feature extraction
        alphas_cumprod = self.scheduler.alphas_cumprod.cpu().numpy()
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1.0, alphas_cumprod))

    def q_sample(self, x_start, t, noise=None):
        """Add noise to x_start at timestep t using DDPMScheduler."""
        noise = default(noise, lambda: torch.randn_like(x_start))
        return self.scheduler.add_noise(x_start, noise, t)

    def p_mean_variance(self, x, t, clip_denoised=True, condition_x=None):
        t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        # Compute noise prediction from denoise_fn
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t]]
        ).to(x.device).view(1, 1, 1, 1).expand(x.shape[0], -1, -1, -1)
        if condition_x is not None:
            noise_pred = self.denoise_fn(
                torch.cat([condition_x, x], dim=1),
                continuous_sqrt_alpha_cumprod.squeeze(-1).squeeze(-1).squeeze(-1)
            )
        else:
            noise_pred = self.denoise_fn(
                x,
                continuous_sqrt_alpha_cumprod.squeeze(-1).squeeze(-1).squeeze(-1)
            )
        # Use scheduler to compute the previous sample
        scheduler_output = self.scheduler.step(noise_pred, t, x)
        prev_sample = scheduler_output.prev_sample
        return prev_sample

    @torch.no_grad()
    def p_sample_loop(self, in_channels, img_size, continous=False):
        device = self.scheduler.alphas_cumprod.device
        sample_inter = max(1, self.num_timesteps // 10)

        img = torch.randn((1, in_channels, img_size, img_size), device=device)
        ret_img = img
        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc='sampling loop time step',
            total=self.num_timesteps,
        ):
            continuous_sqrt_alpha_cumprod = torch.FloatTensor(
                [self.sqrt_alphas_cumprod_prev[i]]
            ).to(device)
            noise_pred = self.denoise_fn(img, continuous_sqrt_alpha_cumprod)
            t_tensor = torch.full((1,), i, device=device, dtype=torch.long)
            img = self.scheduler.step(noise_pred, i, img).prev_sample
            if i % sample_inter == 0:
                ret_img = torch.cat([ret_img, img], dim=0)

        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def sampling_imgs(self, in_channels, img_size, continous=False):
        return self.p_sample_loop(in_channels, img_size, continous)

    @torch.no_grad()
    def feats(self, x, t, noise=None):
        """
        Extract intermediate UNet features at a given diffusion timestep.

        Args:
            x: input image tensor (B, C, H, W)
            t: integer diffusion timestep
            noise: optional pre-sampled noise

        Returns:
            fe: list of encoder feature maps
            fd: list of decoder feature maps (reversed)
        """
        x_start = x
        b = x_start.shape[0]

        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t - 1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b,
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = (
            continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1) * x_start
            + (1 - continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1) ** 2).sqrt() * noise
        )

        fe, fd = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod, feat_need=True)
        return fe, fd

    def p_losses(self, x_in, noise=None):
        x_start = x_in['img']
        b, c, h, w = x_start.shape

        t = np.random.randint(1, self.num_timesteps + 1)

        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t - 1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b,
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = (
            continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1) * x_start
            + (1 - continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1) ** 2).sqrt() * noise
        )
        x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        loss = self.loss_func(noise, x_recon)
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)


class DDPM(BaseModel):
    """
    DDPM model wrapper providing training and inference interfaces.
    """

    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None

        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train'
        )
        if self.opt['phase'] == 'train':
            self.netG.train()
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k)
                        )
            else:
                optim_params = list(self.netG.parameters())

            if opt['train']['optimizer']['type'] == 'adam':
                self.optG = torch.optim.Adam(
                    optim_params, lr=opt['train']['optimizer']['lr']
                )
            elif opt['train']['optimizer']['type'] == 'adamw':
                self.optG = torch.optim.AdamW(
                    optim_params, lr=opt['train']['optimizer']['lr']
                )
            else:
                raise NotImplementedError(
                    'Optimizer [{:s}] not implemented'.format(
                        opt['train']['optimizer']['type']
                    )
                )
            self.log_dict = OrderedDict()
        self.load_network()
        self.print_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        l_pix = self.netG(self.data)
        b, c, h, w = self.data['img'].shape
        l_pix = l_pix.sum() / int(b * c * h * w)
        l_pix.backward()
        self.optG.step()
        self.log_dict['l_pix'] = l_pix.item()

    def test(self, in_channels, img_size, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.sampled_img = self.netG.module.sampling_imgs(
                    in_channels, img_size, continous
                )
            else:
                self.sampled_img = self.netG.sampling_imgs(
                    in_channels, img_size, continous
                )
        self.netG.train()

    def get_feats(self, t):
        """Extract encoder and decoder features from the diffusion UNet."""
        self.netG.eval()
        A = self.data['A']
        B = self.data['B']
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                fe_A, fd_A = self.netG.module.feats(A, t)
                fe_B, fd_B = self.netG.module.feats(B, t)
            else:
                fe_A, fd_A = self.netG.feats(A, t)
                fe_B, fd_B = self.netG.feats(B, t)
        self.netG.train()
        return fe_A, fd_A, fe_B, fd_B

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['SAM'] = self.sampled_img.detach().float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(
                self.netG.__class__.__name__, self.netG.module.__class__.__name__
            )
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n)
        )
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'],
            'I{}_E{}_gen.pth'.format(iter_step, epoch),
        )
        opt_path = os.path.join(
            self.opt['path']['checkpoint'],
            'I{}_E{}_opt.pth'.format(iter_step, epoch),
        )
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        opt_state = {
            'epoch': epoch,
            'iter': iter_step,
            'scheduler': None,
            'optimizer': None,
        }
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)
        logger.info('Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path)
            )
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(gen_path), strict=False)
            if self.opt['phase'] == 'train':
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
