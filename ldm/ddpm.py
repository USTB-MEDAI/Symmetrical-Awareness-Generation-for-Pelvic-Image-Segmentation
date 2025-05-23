import math
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
# import pytorch_lightning as pl
import lightning as pl
import os
from torch.optim.lr_scheduler import LambdaLR
from torchvision.models import resnet50
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
from torch import clamp
import SimpleITK as sitk
import torch.fft as fft
import torch.functional as F
from PIL import Image
from monai.transforms import SaveImage

from .lr_scheduler import LambdaLinearScheduler
from .unet import UNetModel

from scipy.ndimage import gaussian_filter
from .util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from .ema import LitEma
from .autoencoderkl.distributions import normal_kl, DiagonalGaussianDistribution
from .autoencoderkl.autoencoder import AutoencoderKL, VQModelInterface, IdentityFirstStage
from .ddpm_utils import make_beta_schedule, extract_into_tensor, noise_like
from .ddim import DDIMSampler
from .Medicalnet.Vit import load_weight_for_vit_encoder, vit_encoder_b
from .condition_extractor import UnetEncoder
from .TextEncoder import FrozenCLIPEmbedder
from .TextEncoder_UniCLIP import Text_Encoder
# from .hotmap import AttentionHook, overlay_2d_slice, interactive_3d_overlay
from .CoordinateEmbeder import CoordinateEmbedder
# current_time_step = 1000

__conditioning_keys__ = {"concat": "c_concat", "crossattn": "c_crossattn", "adm": "y"}
class resnet_duplicate(nn.Module):
    def __init__(self,cond_type,ckpt_path):
        super().__init__()
        self.resnet = resnet50()
        # for name in self.resnet.named_modules():
        #     print(name)
        # self.resnet.layer4 = nn.Sequential()
        # self.resnet.avgpool = nn.Sequential()
        # self.resnet.fc = nn.Sequential()
        if cond_type=="pretrained_resnet":
            ckpt = torch.load(ckpt_path)
            self.resnet.load_state_dict(ckpt,strict=False)
            self.resnet.eval()
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # test_tensor = torch.randn((1,3,256,256))
        # print(test_tensor.shape)
        # for name,module in self.resnet._modules.items():
        #     print(name)
        #     print(module)
        #     test_tensor=module(test_tensor)
        #     print(test_tensor.shape)
        # for name in self.resnet.named_modules():
        #     print(name)
        self.fc = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 8, kernel_size=1, stride=1, padding=0),
        )
    def forward(self,x):
        for name,module in self.resnet._modules.items():
            if name=="layer4":
                break
            x=module(x)
        x = self.fc(x)
        _,_,h,w =  x.shape
        x = x.unsqueeze(-1)
        # x = torch.cat([x]*h,dim=-1)
        x = x.repeat(1,1,1,1,h)
        return x


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2

def fc_encoder(x):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 获取模型权重的设备
    x = x.to(device)  # 将输入张量移动到相同的设备
    
    if x.shape[2] != 768:
        raise ValueError(f"Input tensor must have 768 features in the second dimension, got {x.shape[1]}")
    # 定义一个全连接层将文本嵌入映射到目标维度
    fc = nn.Linear(768, 4 * 16 * 16 * 16)  # 输入维度为768，输出维度为4×16×16×16
    text_mapped = fc(x)  # 形状变为 (1, 77, 4×16×16×16)
    # 重塑为与图片相同的维度
    text_mapped = text_mapped.view(1, 77, 4, 16, 16, 16)  # 重塑为 (1, 77, 4, 16, 16, 16)
    text_mapped = text_mapped.mean(dim=1)  # 在时间步维度（dim=1）取平均，形状变为 (1, 4, 16, 16, 16)
    return text_mapped


class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(
        self,
        unet_config,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        ckpt_path=None,
        ignore_keys=[],
        load_only_unet=False,
        monitor="val/loss",
        use_ema=True,
        first_stage_key="image",
        image_size=256,
        channels=3,
        log_every_t=100,
        clip_denoised=True,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        given_betas=None,
        original_elbo_weight=0.0,
        base_learning_rate=1e-5,
        v_posterior=0.0,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        l_simple_weight=1.0,
        conditioning_key=None,
        parameterization="eps",  # all assuming fixed variance schedules
        scheduler_config=None,
        use_positional_encodings=False,
        learn_logvar=False,
        logvar_init=0.0,
        batch_size=8,
        pad_channel=8,
        root_path=None,
    ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        self.current_time_step = 1000
        self.learning_rate = base_learning_rate
        self.batch_size = batch_size
        self.pad_channel = pad_channel
        self.root_path = root_path
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,)).to(self.device)
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

    def register_schedule(
        self, given_betas=None, beta_schedule="linear", timesteps=1000, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
    ):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s
            )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1.0 - alphas_cumprod_prev) / (
            1.0 - alphas_cumprod
        ) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer("posterior_log_variance_clipped", to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer("posterior_mean_coef1", to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)))
        self.register_buffer(
            "posterior_mean_coef2", to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod))
        )

        if self.parameterization == "eps":
            lvlb_weights = self.betas**2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2.0 * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = (
            self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(sd, strict=False)
        )
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        print(f"Sampling from time step {t}")
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        print(f"Sampling {shape} from time step 0 to {self.num_timesteps}")
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc="Sampling t", total=self.num_timesteps):
            print(f"""Sampling time step {i}""")
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        print(f"Sampling {batch_size} from time step 0 to {self.num_timesteps}")
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), return_intermediates=return_intermediates)
    
    #生成在特定时间步 t 下的扩散样本 x_t, 给无噪声图像x_strat加噪声, t表示从无噪声图像到最终噪声图像的中间步骤
    def q_sample(self, x_start, t, noise=None):
        # print(f"Sampling from time step {t}")
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction="none")
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = "train" if self.training else "val"

        loss_dict.update({f"{log_prefix}/loss_simple": loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f"{log_prefix}/loss_vlb": loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f"{log_prefix}/loss": loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, "b h w c -> b c h w")
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        losses, _ = self.shared_step(batch)
        loss, loss_dict = losses

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        # 记录全局训练步骤数以跟踪训练进度
        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + "_ema": loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)
        torch.cuda.empty_cache()

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, "n b c h w -> b n c h w")
        denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), "1 -> b", b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


class LatentDiffusion(DDPM):
    """main class"""

    def __init__(
        self,
        first_stage_config,
        cond_stage_config,
        coord_stage_config,
        num_timesteps_cond=None,
        cond_stage_key="image",
        cond_stage_trainable=True,
        concat_mode=True,
        cond_stage_forward=None,
        conditioning_key=None,
        scale_factor=1.0,
        scale_by_std=False,
        high_low_mode=False,
        cond_nums=[1],
        *args,
        **kwargs,
    ):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs["timesteps"]
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = "concat" if concat_mode else "crossattn"
        if cond_stage_config == "__is_unconditional__":
            conditioning_key = None
        # * condition_key should be concat here
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)  # * load autoencoderkl
        self.instantiate_cond_stage(cond_stage_config)  # * load cond stage model
        self.instantiate_coord_stage(coord_stage_config)  # * load coord stage model
        self.cond_stage_forward = cond_stage_forward  # * false
        self.clip_denoised = False  # * false?
        self.bbox_tokenizer = None
        self.high_low_mode = high_low_mode
        self.cond_nums = cond_nums

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

    def make_cond_schedule(
        self,
    ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[: self.num_timesteps_cond] = ids

    def lowpass_torch(self, input, limit):
        pass1 = torch.abs(fft.rfftfreq(input.shape[-1])) < limit
        pass2 = torch.abs(fft.fftfreq(input.shape[-2])) < limit
        kernel = torch.outer(pass2, pass1).to(input)
        fft_input = fft.rfftn(input)
        return fft.irfftn(fft_input * kernel, s=input.shape[-3:])

    def highpass_torch(self, input, limit):
        pass1 = torch.abs(fft.rfftfreq(input.shape[-1])) > limit
        pass2 = torch.abs(fft.fftfreq(input.shape[-2])) > limit
        kernel = torch.outer(pass2, pass1).to(input)
        fft_input = fft.rfftn(input)
        return fft.irfftn(fft_input * kernel, s=input.shape[-3:])

    def _high_low_loss(self, rec, target, low_limit, high_limit):
        rec_low = self.lowpass_torch(rec, low_limit)
        target_low = self.lowpass_torch(target, low_limit)

        rec_high = self.highpass_torch(rec, high_limit)
        target_high = self.highpass_torch(target, high_limit)

        return F.mse_loss(rec_low, target_low) + F.mse_loss(rec_high, target_high)

    # @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if (
            self.scale_by_std
            and self.current_epoch == 0
            and self.global_step == 0
            and batch_idx == 0
            and not self.restarted_from_ckpt
            and self.global_rank == 0
        ):
            assert self.scale_factor == 1.0, "rather not use custom rescaling and std-rescaling simultaneously"
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            # x = super().get_input(batch, self.first_stage_key)
            # x, _ = batch
            x = batch["image"]
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer("scale_factor", 1.0 / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(
        self, given_betas=None, beta_schedule="linear", timesteps=1000, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
    ):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        # model = instantiate_from_config(config)
        model = AutoencoderKL(**config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        # self.cond_stage_model = self.first_stage_model
        # self.cond_stage_model = self.cond_stage_model.eval()
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
        
        torch.cuda.empty_cache()

    def instantiate_cond_stage(self, config):
        # model = instantiate_from_config(config)
        model = Text_Encoder(**config)
        # checkpoint = torch.load(config.cond_ckpt_path)
        # 将权重加载到模型中
        # model.load_state_dict(checkpoint['state_dict'])
        self.cond_stage_model = model
        #self.cond_stage_model = model.train()
        # self.cond_stage_model.train = disabled_train
        # self.cond_stage_model = self.first_stage_model
        # self.cond_stage_model = self.cond_stage_model.eval()
        # for param in self.cond_stage_model.parameters():
        #     param.requires_grad = False
        
        torch.cuda.empty_cache()
            
        # if not self.cond_stage_trainable:
        #     if config == "__is_first_stage__":
        #         print("Using first stage also as cond stage.")
        #         self.cond_stage_model = self.first_stage_model
        #     elif config == "__is_unconditional__":
        #         print(f"Training {self.__class__.__name__} as an unconditional model.")
        #         self.cond_stage_model = None
        #         # self.be_unconditional = True
        #     else:
        #         model = instantiate_from_config(config)
        #         self.cond_stage_model = model.eval()
        #         self.cond_stage_model.train = disabled_train
        #         for param in self.cond_stage_model.parameters():
        #             param.requires_grad = False
        # else:
        #     assert config != "__is_first_stage__"
        #     assert config != "__is_unconditional__"
        #     model = instantiate_from_config(config)
        #     self.cond_stage_model = model
    def instantiate_coord_stage(self, config):
        Embedder = CoordinateEmbedder(**config)
        self.coord_stage_model = Embedder
        

    def _get_denoise_row_from_list(self, samples, desc="", force_no_decoder_quantization=False):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device), force_not_quantize=force_no_decoder_quantization))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, "n b c h w -> b n c h w")
        denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.first_stage_model, "encode") and callable(self.first_stage_model.encode):
                c = self.first_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.first_stage_model(c)
        else:
            assert hasattr(self.first_stage_model, self.cond_stage_forward)
            c = getattr(self.first_stage_model, self.cond_stage_forward)(c)
        return c


    def condition_vit_encode(self, cond):
        """
        using vit backbone to encode conditioning x-ray imgs.
        backbone checkpoint from https://github.com/duyhominhnguyen/LVM-Med
        input: (1,1,256,256) 
        output: (1,4,16,16,16) match the latent code z.
        """
        # * repeat second channel
        cond = cond.repeat(1, 3, 1, 1)
        cond = self.cond_stage_model(cond)
        #cond = self.cond_stage_model(cond)
        return cond

    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(
            weighting,
            self.split_input_params["clip_min_weight"],
            self.split_input_params["clip_max_weight"],
        )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(
                L_weighting, self.split_input_params["clip_min_tie_weight"], self.split_input_params["clip_max_tie_weight"]
            )

            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(
                kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                dilation=1,
                padding=0,
                stride=(stride[0] * uf, stride[1] * uf),
            )
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(
                kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                dilation=1,
                padding=0,
                stride=(stride[0] // df, stride[1] // df),
            )
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting
    
    def check_interference(self, text_emb, coords_emb):
        # 计算特征相似度矩阵
        sim = torch.einsum('btd,bcd->btc', 
            text_emb / text_emb.norm(dim=-1, keepdim=True),
            coords_emb / coords_emb.norm(dim=-1, keepdim=True))
        print(f"特征相似度矩阵 (应接近0):\n{sim.mean(dim=0)}")

    # 检查坐标编码区分度
    def check_coord_discrimination(self, coords_emb, labels):
        # labels: 1=左髋, 2=右髋, 0=骶骨
        left_feat = coords_emb[labels==1].mean(dim=0)
        right_feat = coords_emb[labels==2].mean(dim=0)
        print(f"左右髋骨特征距离: {torch.cosine_similarity(left_feat, right_feat, dim=0):.3f} (应<-0.5)")

    @torch.no_grad()
    def get_input(
        self,
        batch,
        k,
        return_first_stage_outputs=False,
        force_c_encode=False,
        cond_key=None,
        return_original_cond=False,
        bs=None,
    ):
        # x = super().get_input(batch, k)
        # x, cond1, cond2 = batch["image"].as_tensor(), batch["cond1"].as_tensor(), batch["cond2"].as_tensor()
        x = batch["image"].as_tensor()
        # if x.shape[0] != 8 and not return_original_cond :
        #     #重复最后一个数据到batch_size
        #     x = torch.cat([x, x[:(8 - x.shape[0])]], dim=0)
        if 1 in self.cond_nums:
            cond1 = batch["cond1"]
            coord = batch["coord"]
            label = batch["label"]
            # print(f"label:{label}, coord:{coord}")
            # if len(cond1) != 8 and not return_original_cond :
            #     # 填充数据到batch_size
            #     cond1 = cond1 + cond1[:(8 - len(cond1))]
            # print(f"cond1 len: {len(cond1)}")
            # print(f"cond1 : {cond1}")
        else:
            cond1 = None
        if 2 in self.cond_nums:
            cond2 = batch["cond2"].as_tensor()
        else:
            cond2 = None
        if 3 in self.cond_nums:
            cond3 = batch["cond3"].as_tensor()
        else:
            cond3 = None

        if bs is not None:
            x = x[:bs]
        
        ###
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()  # *  input image -> first stage -> sample -> z
        ###
        
        cond = cond1
        
        if self.high_low_mode:
            high_c = self.highpass_torch(cond, limit=0.04)
            high_c = self.get_learned_conditioning(high_c)
            low_c = self.lowpass_torch(cond, limit=0.04)
            low_c = self.get_learned_conditioning(low_c)

            c = self.get_learned_conditioning(cond)
            c = torch.concat([c, high_c, low_c], dim=1)
        else:
            # c = self.get_learned_conditioning(cond)
            # cond_cat = []
            if 1 in self.cond_nums:
                # print(f"cond1: {cond1}")
                # print(f"coord: {coord}")
                encoder_token = self.encode_cond_stage(cond1)
                text_emb = encoder_token.to(self.device)
                coords_emb = self.encode_coord_stage(coord)
                # print(f"text_emb:{text_emb}")
                # print(f"coords_emb:{coords_emb}")
                # print("Text norm:", text_emb.norm(), "Coord norm:", coords_emb.norm())
                # 检查特征竞争
                # self.check_interference(text_emb, coords_emb)
                # self.check_coord_discrimination(coords_emb, label)
                #c = text_emb + coords_emb
                # print(f"text_emb shape: {text_emb.shape}")
                # print(f"coords_emb shape: {coords_emb.shape}")
                c = torch.concat([text_emb, coords_emb], dim=1)
            # if 2 in self.cond_nums:
            #     cond2 = self.condition_vit_encode(cond2)
            #     cond_cat.append(cond2)
            # if 3 in self.cond_nums:
            #     cond3 = self.condition_vit_encode(cond3)
            #     cond_cat.append(cond3)
            
            #c = torch.cat(cond_cat, dim=1)
            
            # print(f"c shape: {c.shape}") 
            # print(f"z shape: {z.shape}")
                
        #zc = torch.concat([z, c], dim=1)
        out = [z, c]
        # print(f"out1_len: {len(out)}") #2
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
            # print(f"out2_len: {len(out)}")
        if return_original_cond:
            # out.append(cond)
            out.extend(cond1)
        # print(f"out: {out}")
        # print(f"out_len: {len(out)}") #
        return out

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 12:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, "b h w c -> b c h w").contiguous()

        z = 1.0 / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [
                        self.first_stage_model.decode(
                            z[:, :, :, :, i], force_not_quantize=predict_cids or force_not_quantize
                        )
                        for i in range(z.shape[-1])
                    ]
                else:
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i]) for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    # same as above but without decorator
    def differentiable_decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, "b h w c -> b c h w").contiguous()

        z = 1.0 / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [
                        self.first_stage_model.decode(
                            z[:, :, :, :, i], force_not_quantize=predict_cids or force_not_quantize
                        )
                        for i in range(z.shape[-1])
                    ]
                else:
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i]) for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                df = self.split_input_params["vqf"]
                self.split_input_params["original_image_size"] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                output_list = [self.first_stage_model.encode(z[:, :, :, :, i]) for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded

            else:#跑的是这个分支
                return self.first_stage_model.encode(x)
        else:
            return self.first_stage_model.encode(x)
        
    def encode_cond_stage(self, x):
        return self.cond_stage_model.forward(x)
    
    def encode_coord_stage(self, coord):
        return self.coord_stage_model.forward(coord)

    def shared_step(self, batch, **kwargs):
        x,  c = self.get_input(batch, self.first_stage_key)  # * x: fisrt_stage gauss sample  ,c origin image
        gt = batch[self.first_stage_key].as_tensor()
        loss = self(x, c, gt)
        return loss, c

    def forward(self, x, c, gt, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        # 随机生成0-1000张量, [2,,,,]
        # if self.model.conditioning_key is not None:
        #     assert c is not None
        #     if self.cond_stage_trainable:
        #         c = self.get_learned_conditioning(c)
        #     if self.shorten_cond_schedule:  # TODO: drop this option
        #         tc = self.cond_ids[t].to(self.device)
        #         c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))  # * c -> c_tc(noisy)
        return self.p_losses(x, c, gt, t, *args, **kwargs)

    def _rescale_annotations(self, bboxes, crop_coordinates):  # TODO: move to dataset
        def rescale_bbox(bbox):
            x0 = clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2])
            y0 = clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3])
            w = min(bbox[2] / crop_coordinates[2], 1 - x0)
            h = min(bbox[3] / crop_coordinates[3], 1 - y0)
            return x0, y0, w, h

        return [rescale_bbox(b) for b in bboxes]

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = "c_concat" if self.model.conditioning_key == "concat" else "c_crossattn"
            cond = {key: cond}
        # print("@@@@@@@@@@@@")
        if hasattr(self, "split_input_params"):   #没有执行这个分支
            print("##########")
            assert len(cond) == 1  # todo can only deal with one conditioning atm
            assert not return_ids
            ks = self.split_input_params["ks"]  # eg. (128, 128)
            stride = self.split_input_params["stride"]  # eg. (64, 64)

            h, w = x_noisy.shape[-2:]

            fold, unfold, normalization, weighting = self.get_fold_unfold(x_noisy, ks, stride)

            z = unfold(x_noisy)  # (bn, nc * prod(**ks), L)
            # Reshape to img shape
            z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
            z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]

            if (
                self.cond_stage_key in ["image", "LR_image", "segmentation", "bbox_img"] and self.model.conditioning_key
            ):  # todo check for completeness
                c_key = next(iter(cond.keys()))  # get key
                c = next(iter(cond.values()))  # get value
                assert len(c) == 1  # todo extend to list with more than one elem
                c = c[0]  # get element

                c = unfold(c)
                c = c.view((c.shape[0], -1, ks[0], ks[1], c.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                cond_list = [{c_key: [c[:, :, :, :, i]]} for i in range(c.shape[-1])]

            elif self.cond_stage_key == "coordinates_bbox":
                assert "original_image_size" in self.split_input_params, "BoudingBoxRescaling is missing original_image_size"

                # assuming padding of unfold is always 0 and its dilation is always 1
                n_patches_per_row = int((w - ks[0]) / stride[0] + 1)
                full_img_h, full_img_w = self.split_input_params["original_image_size"]
                # as we are operating on latents, we need the factor from the original image size to the
                # spatial latent size to properly rescale the crops for regenerating the bbox annotations
                num_downs = self.first_stage_model.encoder.num_resolutions - 1
                rescale_latent = 2 ** (num_downs)

                # get top left postions of patches as conforming for the bbbox tokenizer, therefore we
                # need to rescale the tl patch coordinates to be in between (0,1)
                tl_patch_coordinates = [
                    (
                        rescale_latent * stride[0] * (patch_nr % n_patches_per_row) / full_img_w,
                        rescale_latent * stride[1] * (patch_nr // n_patches_per_row) / full_img_h,
                    )
                    for patch_nr in range(z.shape[-1])
                ]

                # patch_limits are tl_coord, width and height coordinates as (x_tl, y_tl, h, w)
                patch_limits = [
                    (x_tl, y_tl, rescale_latent * ks[0] / full_img_w, rescale_latent * ks[1] / full_img_h)
                    for x_tl, y_tl in tl_patch_coordinates
                ]
                # patch_values = [(np.arange(x_tl,min(x_tl+ks, 1.)),np.arange(y_tl,min(y_tl+ks, 1.))) for x_tl, y_tl in tl_patch_coordinates]

                # tokenize crop coordinates for the bounding boxes of the respective patches
                patch_limits_tknzd = [
                    torch.LongTensor(self.bbox_tokenizer._crop_encoder(bbox))[None].to(self.device) for bbox in patch_limits
                ]  # list of length l with tensors of shape (1, 2)
                print(patch_limits_tknzd[0].shape)
                # cut tknzd crop position from conditioning
                assert isinstance(cond, dict), "cond must be dict to be fed into model"
                cut_cond = cond["c_crossattn"][0][..., :-2].to(self.device)
                print(cut_cond.shape)

                adapted_cond = torch.stack([torch.cat([cut_cond, p], dim=1) for p in patch_limits_tknzd])
                adapted_cond = rearrange(adapted_cond, "l b n -> (l b) n")
                print(adapted_cond.shape)
                adapted_cond = self.get_learned_conditioning(adapted_cond)
                print(adapted_cond.shape)
                adapted_cond = rearrange(adapted_cond, "(l b) n d -> l b n d", l=z.shape[-1])
                print(adapted_cond.shape)

                cond_list = [{"c_crossattn": [e]} for e in adapted_cond]

            else:
                cond_list = [cond for i in range(z.shape[-1])]  # Todo make this more efficient

            # apply model by loop over crops
            output_list = [self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]
            assert not isinstance(
                output_list[0], tuple
            )  # todo cant deal with multiple model outputs check this never happens

            o = torch.stack(output_list, axis=-1)
            o = o * weighting
            # Reverse reshape to img shape
            o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
            # stitch crops together
            x_recon = fold(o) / normalization

        else:
            x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / extract_into_tensor(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)
    
    def threshold_segmentation(data, method='otsu'):
        """
        使用阈值分割将图像数据转换为二值图像
        :param data: 输入图像数据（numpy数组）
        :param method: 阈值分割方法，可选值为 'otsu', 'mean', 'isodata'
        :return: 二值化的图像数据
        """
        if method == 'otsu':
            # 最大类间方差法（OTSU）
            threshold = threshold_otsu(data)
        elif method == 'mean':
            # 均值阈值
            threshold = threshold_mean(data)
        elif method == 'isodata':
            # ISODATA阈值
            threshold = threshold_isodata(data)
        else:
            raise ValueError("不支持的阈值分割方法")
        
        # 应用阈值分割
        binary_data = data > threshold
        return binary_data.astype(np.uint8)


    #高斯分解的多尺度3D Otsu阈值分割算法


    def gaussian_decomposition(self, image, sigmas):
        """
        对图像进行高斯分解，得到不同尺度的子图像。
        :param image: 输入3D灰度图像
        :param sigmas: 高斯核的标准差列表，用于多尺度分解
        :return: 包含不同尺度子图像的列表
        """
        decomposed_images = []
        for sigma in sigmas:
            # 应用高斯滤波
            blurred = gaussian_filter(image, sigma=sigma)
            decomposed_images.append(blurred)
        return decomposed_images

    def calculate_3d_otsu_threshold(self, image):
        """
        计算3D Otsu阈值。
        :param image: 输入的单个尺度的3D灰度图像
        :return: 计算得到的Otsu阈值
        """
        # Flatten the image to 1D array
        pixels = image.flatten()
        
        # Compute histogram
        hist, bins = np.histogram(pixels, bins=256, range=[0, 256])
        
        # Compute normalized histogram
        hist = hist.astype(float) / hist.sum()
        
        # Compute cumulative sums
        omega1 = np.cumsum(hist)
        omega2 = 1 - omega1
        
        # Avoid division by zero
        omega2[omega2 == 0] = 1e-10
        
        # Compute means
        mu1 = np.cumsum(hist * np.arange(256))
        mu_t = mu1[-1]
        mu1 = mu1 / omega1
        mu2 = (mu_t - mu1 * omega1) / omega2
        
        # Compute between-class variance
        variance = omega1 * omega2 * (mu1 - mu2) ** 2
        
        # Find the threshold that maximizes the variance
        threshold = np.argmax(variance)
        
        return threshold

    def fuse_thresholds(self, thresholds, weights=None):
        """
        融合多个阈值。
        :param thresholds: 不同尺度下的阈值列表
        :param weights: 融合时的权重，默认为均等权重
        :return: 融合后的最终阈值
        """
        if weights is None:
            weights = np.ones(len(thresholds)) / len(thresholds)
        else:
            weights = np.array(weights)
            weights /= weights.sum()
        
        fused_threshold = np.dot(thresholds, weights)
        return fused_threshold

    def multiscale_3d_otsu_segmentation(self, image, sigmas):
        """
        多尺度3D Otsu阈值分割主函数。
        :param image: 输入3D灰度图像
        :param sigmas: 高斯核的标准差列表，用于多尺度分解
        :return: 分割后的二值图像
        """
        # 高斯分解
        decomposed_images = self.gaussian_decomposition(image, sigmas)
        
        # 计算每个尺度下的Otsu阈值
        thresholds = []
        for decomposed_image in decomposed_images:
            threshold = self.calculate_3d_otsu_threshold(decomposed_image)
            thresholds.append(threshold)
        
        # 融合阈值
        fused_threshold = self.fuse_thresholds(thresholds)
        
        # 应用最终阈值进行分割
        segmented_image = np.where(image > fused_threshold, 1, 0).astype(np.uint8)
        
        return segmented_image

    def dice_loss(self, pred_mask, true_mask, x_noisy, model_output, i, smooth=1e-6):
        output_folder = '/disk/syz/Xray-Diffsuion/image'
        pred_mask = self.decode_first_stage(torch.tensor(pred_mask))
        pred_mask = pred_mask.cpu().float()
        true_mask = true_mask.cpu().float()
        #加噪数据   
        x_noisy_mask = self.decode_first_stage(x_noisy)
        x_noisy_mask = x_noisy_mask.cpu().float()
        x_noisy_mask = x_noisy_mask[0, 1, :, :, :]
        #预测噪声
        # model_output = self.decode_first_stage(model_output)
        model_output = model_output.cpu().float()
        model_output = model_output[0, 1, :, :, :]
        
        #二值化
        pred_mask = pred_mask[0, 1, :, :, :]
        true_mask = true_mask[0, 1, :, :, :]
        pred_mask = pred_mask.squeeze()
        true_mask = true_mask.squeeze()
        x_noisy_mask = x_noisy_mask.squeeze()
        model_output = model_output.squeeze()
        sigmas = [1, 2, 4]
        pred_mask = self.multiscale_3d_otsu_segmentation(pred_mask, sigmas)
        true_mask = self.multiscale_3d_otsu_segmentation(true_mask, sigmas)

        # 输出二值化图像切片
        # print(f"pred_mask.shape:{pred_mask.shape},true_mask.shape:{true_mask.shape}")  #(1,128,128,128)
        slice_index = pred_mask.shape[1] // 2  # 获取中间切片
        image_slice = pred_mask[slice_index, :, :]  # 转换为 numpy 数组
        image_slice_notrans = true_mask[slice_index, :, :] # 转换为 numpy 数组
        x_noisy_mask = x_noisy_mask[slice_index, :, :] # 转换为 numpy 数组
        model_output = model_output[8, :, :] # 转换为 numpy 数组

        # 使用 matplotlib 保存为 JPG 图像
        plt.imsave(os.path.join(output_folder, f'pred_mask{i}.jpg'), image_slice, cmap='gray')
        plt.imsave(os.path.join(output_folder, f'true_mask{i}.jpg'), image_slice_notrans, cmap='gray')
        plt.imsave(os.path.join(output_folder, f'x_noisy_mask{i}.jpg'), x_noisy_mask, cmap='gray')
        plt.imsave(os.path.join(output_folder, f'model_output{i}.jpg'), model_output.detach().numpy(), cmap='gray')
        
        # 预测Mask经过Sigmoid处理（假设输出未归一化）

        # 展平张量
        pred_flat = pred_mask.flatten()
        target_flat = true_mask.flatten()
        # print(f"pred_flat.shape:{pred_flat.shape},true_flat.shape:{true_flat.shape}")  #(2,128,128,128)
        
        # 计算交集和并集
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        print(f"intersection:{intersection},union:{union}, pred_flat.sum():{pred_flat.sum()}, target_flat.sum():{target_flat.sum()}")
        # Dice系数
        dice = (2.0 * intersection + smooth) / (union + smooth)
        
        # 返回Dice损失（Dice）
        return 1 - dice
    
    #sigmoidal权重函数
    def sigmoidal_function(t):
        k = 10
        c = 990
        y = 1 / (1 + np.exp(-k * (t - c)))
        return y
    
    def p_losses(self, x_start, cond, gt, t, noise=None):
        # torch.randn_like(x_start)会生成一个与x_start具有相同尺寸（shape）的张量，其元素是从标准正态分布（均值为0，标准差为1）中随机抽取的。
        noise = default(noise, lambda: torch.randn_like(x_start))  
        #已经压缩数据 加噪 根据给定的时间步 t，将 x_start 添加噪声得到 x_noisy
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)  # * x -> x_t (noisey)
        #去噪
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        
        prefix = "train" if self.training else "val"
        # 如果参数化方式为 "x0"，则目标值为原始输入 x_start
        if self.parameterization == "x0":
            target = x_start
        # 如果参数化方式为 "eps"，则目标值为噪声 eps
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()
        # loss_dict.update({f"{prefix}/cts":self.current_epoch})
        # 计算mse损失
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3, 4])
        loss_dict.update({f"{prefix}/loss_simple": loss_simple.mean()})

        self.logvar = self.logvar.to(self.device)
        logvar_t = self.logvar[t].to(self.device)
        #计算加权后的损失
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f"{prefix}/loss_gamma": loss.mean()})
            loss_dict.update({"logvar": self.logvar.data.mean()})
        #将损失乘以权重 1.0
        loss = self.l_simple_weight * loss.mean()
        #计算mse损失
        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3, 4))
        # 使用时间步 t 对应的权重 lvlb_weights[t] 加权 loss_vlb
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f"{prefix}/loss_vlb": loss_vlb})
        loss += self.original_elbo_weight * loss_vlb
        loss_dict.update({f"{prefix}/loss": loss})
        
        return loss, loss_dict

    def p_mean_variance(
        self,
        x,
        c,
        t,
        clip_denoised: bool,
        return_codebook_ids=False,
        quantize_denoised=False,
        return_x0=False,
        score_corrector=None,
        corrector_kwargs=None,
    ):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        x,
        c,
        t,
        clip_denoised=False,
        repeat_noise=False,
        return_codebook_ids=False,
        quantize_denoised=False,
        return_x0=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
    ):
        print(f"current_time_step:{current_time_step}")
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(
            x=x,
            c=c,
            t=t,
            clip_denoised=clip_denoised,
            return_codebook_ids=return_codebook_ids,
            quantize_denoised=quantize_denoised,
            return_x0=return_x0,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
        )
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(
        self,
        cond,
        shape,
        verbose=True,
        callback=None,
        quantize_denoised=False,
        img_callback=None,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        batch_size=None,
        x_T=None,
        start_T=None,
        log_every_t=None,
    ):
        global current_time_step
        
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {
                    key: cond[key][:batch_size]
                    if not isinstance(cond[key], list)
                    else list(map(lambda x: x[:batch_size], cond[key]))
                    for key in cond
                }
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        print (f"start_T:{start_T}, timesteps:{timesteps}")
        iterator = (
            tqdm(reversed(range(0, timesteps)), desc="Progressive Generation", total=timesteps)
            if verbose
            else reversed(range(0, timesteps))
        )
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            current_time_step = i  # 获取当前训练到的时间步
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != "hybrid"
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(
                img,
                cond,
                ts,
                clip_denoised=self.clip_denoised,
                quantize_denoised=quantize_denoised,
                return_x0=True,
                temperature=temperature[i],
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
            )
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(
        self,
        cond,
        shape,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        start_T=None,
        log_every_t=None,
    ):
        print(f"current_time_step2:{current_time_step}")
        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
            
        print(f"start_T:{start_T}, timesteps:{timesteps}")
        iterator = (
            tqdm(reversed(range(0, timesteps)), desc="Sampling t", total=timesteps)
            if verbose
            else reversed(range(0, timesteps))
        )

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            print(f"sampling t:{i}")
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != "hybrid"
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts, clip_denoised=self.clip_denoised, quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(
        self,
        cond,
        batch_size=16,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        shape=None,
        **kwargs,
    ):
        print(f"current_time_step3:{current_time_step}")
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {
                    key: cond[key][:batch_size]
                    if not isinstance(cond[key], list)
                    else list(map(lambda x: x[:batch_size], cond[key]))
                    for key in cond
                }
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(
            cond,
            shape,
            return_intermediates=return_intermediates,
            x_T=x_T,
            verbose=verbose,
            timesteps=timesteps,
            quantize_denoised=quantize_denoised,
            mask=mask,
            x0=x0,
        )

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        print("sampling with ddim")
        if ddim:
            print("ddim is on")
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size, return_intermediates=True, **kwargs)

        return samples, intermediates

    def validation_step(self, batch, batch_idx):
        # _, loss_dict_no_ema, cond = self.shared_step(batch)
        if batch_idx == 1:
            val_losses, cond = self.shared_step(batch)
            _, loss_dict_no_ema = val_losses
            with self.ema_scope():
                # _, loss_dict_ema = self.shared_step(batch)
                val_losses, _ = self.shared_step(batch)
                _, loss_dict_ema = val_losses
                loss_dict_ema = {key + "_ema": loss_dict_ema[key] for key in loss_dict_ema}
            self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
            self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
            if batch_idx == 1:
                ddim_sampler = DDIMSampler(self)
                shape = (self.batch_size, self.channels, self.image_size, self.image_size, self.image_size)
                print(f"ddim shape :{shape}")
                from torch.utils.data import DataLoader, TensorDataset

                cond_z, _ = ddim_sampler.sample(50, batch_size=self.batch_size, shape=shape, conditioning=cond, verbose=False)
                reconstructions = self.decode_first_stage(cond_z)
                reconstructions = torch.clamp(reconstructions, min=-1, max=1)
                reconstructions = (reconstructions + 1) * 127.5
                # 假设 reconstructions 的形状是 [B, C, D, H, W]
                B_r, C, D, H, W = reconstructions.shape

                # 展开 D 维度，变成 [B * D, C, H, W]
                reconstructions = reconstructions.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
                # reconstructions = reconstructions.squeeze(0).permute(1, 0, 2, 3)
                reconstructions = reconstructions.type(torch.uint8)
                grid = make_grid(reconstructions)
                self.logger.experiment.add_image("val_rec", grid, self.global_step)
                
                x = batch["image"]
                B_x, C, D, H, W = x.shape
                x = torch.clamp(x, min=-1, max=1)
                x = (x + 1) * 127.5
                x = x.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
                # x = x.squeeze(0).permute(1, 0, 2, 3)
                # x = x.permute(1, 0, 2, 3)
                x = x.type(torch.uint8)
                grid = make_grid(x)
                self.logger.experiment.add_image("val_gt", grid, self.global_step)

        # torch.cuda.empty_cache()

    def img_saver(self, img, post_fix, i_type=".nii", meta_data=None, random_num=None,**kwargs):
        """
        save img to self.root_path with post_fix

        Args:
            img (torch.Tensor): [description]
            post_fix (str): [description]
            type (str, optional): [description]. Defaults to "nii".
            meta_data ([type], optional): [description]. Defaults to None.
        """
        # if hasattr(img, "meta") and meta_data is None:
        #     meta_data = img.meta
        # elif meta_data is None:
        #     print("img dosen't has meta attribution use `None` as meta_data")
        # else:
        #     print("use the input meta data")

        assert i_type in [".nii", ".nii.gz", ".jpg",".txt"], "Only .nii or .jpg suffix file supported now"
        assert post_fix in ["origin_x", "ae_rec", "rec","label_x","label_rec","text","xray_left", "xray_right", "rec"], "unsupported post_fix"
        
        if post_fix is not "text":
            img = img.squeeze(0)
            
        writer = "NibabelWriter" if "nii" in i_type else "PILWriter"
        out_ext = ".nii.gz" if "nii" in i_type else ".jpg"
        
        out_ext = f"{random_num}{out_ext}"
        if post_fix in ["ae_rec","rec"]:
            img = torch.clamp(img, min=-1, max=1)
            img = (img + 1) * 127.5
            # MAX = torch.max(img)
            # MIN = torch.min(img)
            # img = 2*(img-MAX)/(MAX-MIN)-1
            # img = (img + 1) * 127.5
        elif post_fix in ["label_rec"]:
            #img = img.to(torch.uint8)
            img = torch.clamp(img, min=-1, max=1) 
            #img_cpu = img.cpu().numpy()
            #img = (img + 1) * 0.5
            # with open(os.path.join(self.root_path, "label_rec.txt"), "a") as f:
            #      f.write(str(img_cpu))
            #      f.write("\n")
            img = torch.where(img > 0.5, 1, 0)
            img = img.to(torch.uint8)
        elif post_fix in ["label_x"]:
            img = torch.where(img >= 1, 1, 0)
            img = img.to(torch.uint8)
        elif post_fix == "text":
            print(f"type of img: {type(img)}")
            if isinstance(img, str):
                print(f"type of img is str, save text to file")
                file_path = os.path.join(self.root_path, f"{meta_data}{random_num}.{i_type}")
                with open(file_path, 'w') as file:
                    file.write(img)
                print(f"Text saved to {file_path}")
            return
        else:
            img = torch.clamp(img, min=-1, max=1)
            img = (img + 1) * 127.5

        saver = SaveImage(
            output_dir=self.root_path,
            output_ext=out_ext,
            output_postfix=post_fix,
            separate_folder=False,
            output_dtype=np.uint8,
            resample=False,
            squeeze_end_dims=True,
            writer=writer,
            **kwargs,
        )
        saver(img, meta_data=meta_data)

    def tensorboard_save(self, img, figure_name, step):
        img = img.squeeze(0)
        img = torch.clamp(img, min=-1, max=1)
        img = (img + 1) * 127.5
        img = img.permute(1, 0, 2, 3)
        img = img.type(torch.uint8)
        grid = make_grid(img)
        self.logger.experiment.add_image(f"{figure_name}", grid, step)
    
    def find_cross_attn_layers(self, model):
        cross_attn_layers = []
        for name, module in model.named_modules():
            # 匹配典型的交叉注意力层命名模式
            if "attn2" in name and "to_out" not in name:
                cross_attn_layers.append(name)
        print("Detected cross-attention layers:", cross_attn_layers)
        return cross_attn_layers
    
    def hotmap_register_hooks(self, model, layer_paths):
        hooks = []
        for path in layer_paths:
            layer = dict([*model.named_modules()])[path]
            hook = AttentionHook()
            handle = layer.register_forward_hook(hook.hook_fn)
            hooks.append((handle, hook))
        return hooks
    
    def hotmap_prepare_data(self, volume, attention_map):
        """处理原始3D数据和注意力热力图"""
        # 假设输入volume形状 (1,2,128,128,128)，取第一个样本
        volume = volume[0].detach().cpu().numpy()
        # 合并双通道（取均值或选择特定通道）
        gray_volume = np.mean(volume, axis=0)  # 形状 (128,128,128)
        
        # 归一化处理
        gray_volume = (gray_volume - gray_volume.min()) / (gray_volume.max() - gray_volume.min())
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        return gray_volume, attention_map    
    
    def test_step(self, batch, batch_idx):
        # * 检测所有交叉注意力层
        # self.target_layer = self.find_cross_attn_layers(self.model.diffusion_model)
        # self.attention_hooks = self.hotmap_register_hooks(self.model.diffusion_model, self.target_layer)
        
        import random
        random_num = random.sample(range(1, 100), 1)

        inputs = batch["image"]
        print(f'inputs shape: {inputs.shape}')
        meta_data = inputs.meta
        mask_meta_data = batch["mask"].meta
        file_path = meta_data["filename_or_obj"]
        file_name = os.path.basename(file_path)  # 获取文件名
        file_name_without_extension = os.path.splitext(file_name)[0]  # 去掉扩展名的文件名
        # meta_data=None
        # if 1 in self.cond_nums:
        #     cond1_meta_data = batch["cond1"].meta
        # if 2 in self.cond_nums:
        #     cond2_meta_data = batch["cond2"].meta
        # if 3 in self.cond_nums:
        #     cond3_meta_data = batch["cond3"].meta
        #print(f"self.get_input_len(batch: {self.get_input(batch, self.first_stage_key, return_first_stage_outputs=True, return_original_cond=True)}")
        _, c, x, x_rec, origin_cond = self.get_input(
            batch, self.first_stage_key, return_first_stage_outputs=True, return_original_cond=True
        )
        channel1 = x[:, 0, :, :, :]
        channel2 = x[:, 1, :, :, :]
                
        self.img_saver(channel1, post_fix="origin_x", meta_data=meta_data, random_num=random_num)
        self.img_saver(channel2, post_fix="label_x", meta_data=mask_meta_data, random_num=random_num)
        
           # self.tensorboard_save(x_rec, "rec", 0)
        # self.img_saver(x, "origin_x", meta_data=meta_data, random_num=random_num)
        # self.img_saver(x_rec, "ae_rec", meta_data=meta_data, random_num=random_num)
        if 1 in self.cond_nums:
            self.img_saver(origin_cond, post_fix="text", i_type=".txt", meta_data=file_name_without_extension, random_num=random_num)
        if 2 in self.cond_nums:
            self.img_saver(origin_cond[1], "xray_1", i_type=".jpg", meta_data=cond2_meta_data)
        if 3 in self.cond_nums:
            self.img_saver(origin_cond[2], "xray_2", i_type=".jpg", meta_data=cond3_meta_data)
            

        ddim_sampler = DDIMSampler(self)
        shape = (self.batch_size, self.channels, self.image_size, self.image_size, self.image_size)
        # cond_z = self.p_sample_loop(cond=c, shape=shape)
        cond_z, _ = ddim_sampler.sample(50, batch_size=1, shape=shape, conditioning=c, verbose=False)
        print(f"cond_z shape: {cond_z.shape}")
        reconstructions = self.decode_first_stage(cond_z)
        print(f"reconstructions shape: {reconstructions.shape}")

        # import nibabel as nib

        # nib_rec = nib.Nifti1Image(reconstructions[0, 0, ...].unsqueeze(-1).cpu().numpy(), np.eye(4))
        # nib.save(nib_rec, os.path.join(self.root_path, "nib_rec.nii.gz"))
        
        #self.img_saver(reconstructions, "rec", meta_data=meta_data, random_num=random_num)
        
        res_channel1 = reconstructions[:, 0, :, :, :]
        res_channel2 = reconstructions[:, 1, :, :, :]
        
        self.img_saver(res_channel1, post_fix="rec",meta_data=meta_data, random_num=random_num)
        self.img_saver(res_channel2, post_fix="label_rec",meta_data=mask_meta_data, random_num=random_num)
        
        #*hotmap
        # 获取最后一个注意力图
        # attn = self.attention_hooks[-1][1].attention_maps[-1][0].float().numpy()
        # print(f"attn shape: {attn.shape}")
        # d, h, w = 4, 4, 8
        # attn_3d = attn.mean(0).reshape(d, h, w)  # 形状 (4,4,8)
        # # 上采样到原始输入尺寸 (128,128,128)
        # import torch.nn.functional as F
        # attn_3d_upsampled = F.interpolate(torch.from_numpy(attn_3d).unsqueeze(0).unsqueeze(0), size=(128, 128, 128), mode='trilinear').squeeze() # 去掉 batch 和 channel 维度
        # print(f"attn_3d_upsampled shape: {attn_3d_upsampled.shape}")
        # attn_3d_upsampled = attn_3d_upsampled.detach().cpu().numpy()  # 转为 numpy 数组

        # # 归一化处理
        # gray_vol, norm_attn = self.hotmap_prepare_data(x, attn_3d_upsampled)
        
        # # 生成可视化
        # fig = overlay_2d_slice(gray_vol, norm_attn)
        # self.logger.experiment.add_figure(
        #     "attention_overlay", fig, self.global_step
        # )
        # # 保存3D交互可视化
        # html_path = f"3d_attention_{batch_idx}.html"
        # interactive_3d_overlay(gray_vol, norm_attn).write_html(html_path, include_plotlyjs=True)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:  # false
            print("Diffusion model optimizing logvar")
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:  # true
            # assert "target" in self.scheduler_config
            # scheduler = instantiate_from_config(self.scheduler_config)
            scheduler = LambdaLinearScheduler(**self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [{"scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule), "interval": "step", "frequency": 1}]
            
            return [opt], scheduler
        # if self.use_scheduler:
        #     # 使用 StepLR 替换原来的调度器
        #     scheduler = torch.optim.lr_scheduler.StepLR(
        #         optimizer=opt,
        #         step_size=self.scheduler_config["step_size"],  # 每30个epoch衰减一次
        #         gamma=self.scheduler_config["gamma"],  # 每次衰减为原来的0.5
        #         # gamma=0.5,  # 每次衰减为原来的0.5
        #     )
        #     return [opt], [{"scheduler": scheduler, "interval": "step"}]
        return opt


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        # self.diffusion_model = instantiate_from_config(diff_model_config)
        self.diffusion_model = UNetModel(**diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, "concat", "crossattn", "hybrid", "adm"]

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:
            print("No cond @@@@@@")
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == "concat":
            print("Cond concat @@@@@@")
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == "crossattn":
            # print("Cond crossattn @@@@@@")
            # print(f"c_crossattn: {c_crossattn}")
            cc = torch.cat(c_crossattn, 1)
            # print(f"cc: {cc.shape}")
            # print(f"x: {x.shape}")
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == "hybrid":
            print("Cond hybrid @@@@@@")
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == "adm":
            print("Cond adm @@@@@@")
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            print("Cond not implemented @@@@@@@")
            raise NotImplementedError()

        return out


class Layout2ImgDiffusion(LatentDiffusion):
    # TODO: move all layout-specific hacks to this class
    def __init__(self, cond_stage_key, *args, **kwargs):
        assert cond_stage_key == "coordinates_bbox", 'Layout2ImgDiffusion only for cond_stage_key="coordinates_bbox"'
        super().__init__(cond_stage_key=cond_stage_key, *args, **kwargs)

    def log_images(self, batch, N=8, *args, **kwargs):
        logs = super().log_images(batch=batch, N=N, *args, **kwargs)

        key = "train" if self.training else "validation"
        dset = self.trainer.datamodule.datasets[key]
        mapper = dset.conditional_builders[self.cond_stage_key]

        bbox_imgs = []
        map_fn = lambda catno: dset.get_textual_label(dset.get_category_id(catno))
        for tknzd_bbox in batch[self.cond_stage_key][:N]:
            bboximg = mapper.plot(tknzd_bbox.detach().cpu(), map_fn, (256, 256))
            bbox_imgs.append(bboximg)

        cond_img = torch.stack(bbox_imgs, dim=0)
        logs["bbox_image"] = cond_img
        return logs
