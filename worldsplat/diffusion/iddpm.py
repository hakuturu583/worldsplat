"""Improved DDPM (IDDPM) with classifier-free guidance for sampling.

This module provides the IDDPM class (built on SpacedDiffusion / GaussianDiffusion)
and the forward_with_cfg helper used during CFG sampling.  All diffusion
primitives (beta schedules, q-sampling, p-sampling, VLB, training losses) are
self-contained here so that no external ``ldm.*`` imports are needed.
"""

import enum
import math
from functools import partial

import numpy as np
import torch
import torch as th
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def mean_flat(tensor, mask=None):
    """Take the mean over all non-batch dimensions."""
    if mask is None:
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    else:
        assert tensor.dim() == 5
        assert tensor.shape[2] == mask.shape[1]
        tensor = rearrange(tensor, "b c t h w -> b t (c h w)")
        denom = mask.sum(dim=1) * tensor.shape[-1]
        loss = (tensor * mask.unsqueeze(2)).sum(dim=1).sum(dim=1) / denom
        return loss


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """Extract values from a 1-D numpy array for a batch of indices."""
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + th.zeros(broadcast_shape, device=timesteps.device)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """Compute KL divergence between two Gaussians."""
    tensor = next(
        (obj for obj in (mean1, logvar1, mean2, logvar2) if isinstance(obj, th.Tensor)),
        None,
    )
    assert tensor is not None
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x, device=tensor.device)
        for x in (logvar1, logvar2)
    ]
    return 0.5 * (
        -1.0 + logvar2 - logvar1 + th.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """Fast approximation of the standard normal CDF."""
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """Log-likelihood of a discretized Gaussian (for uint8 images scaled to [-1, 1])."""
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ModelMeanType(enum.Enum):
    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()


class ModelVarType(enum.Enum):
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()
    RESCALED_MSE = enum.auto()
    KL = enum.auto()
    RESCALED_KL = enum.auto()

    def is_vb(self):
        return self in [LossType.KL, LossType.RESCALED_KL]


# ---------------------------------------------------------------------------
# Beta schedule helpers
# ---------------------------------------------------------------------------

def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == "quad":
        betas = (
            np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """Get a pre-defined beta schedule by name ('linear' or 'squaredcos_cap_v2')."""
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        return get_beta_schedule(
            "linear",
            beta_start=scale * 0.0001,
            beta_end=scale * 0.02,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
    elif schedule_name == "squaredcos_cap_v2":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


# ---------------------------------------------------------------------------
# GaussianDiffusion
# ---------------------------------------------------------------------------

class GaussianDiffusion:
    """Core Gaussian diffusion utilities for training and sampling."""

    def __init__(self, *, betas, model_mean_type, model_var_type, loss_type):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type

        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = (
            np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
            if len(self.posterior_variance) > 1
            else np.array([])
        )
        self.posterior_mean_coef1 = betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = th.randn_like(x_start)
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, clip_denoised=True, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, t, **model_kwargs)
        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)
        elif self.model_var_type in [ModelVarType.FIXED_LARGE, ModelVarType.FIXED_SMALL]:
            model_variance, model_log_variance = {
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)
        else:
            model_variance = th.zeros_like(model_output)
            model_log_variance = th.zeros_like(model_output)

        def process_xstart(x):
            return x.clamp(-1, 1) if clip_denoised else x

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
        else:
            pred_xstart = process_xstart(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "extra": extra,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_sample(self, model, x, t, clip_denoised=True, model_kwargs=None, mask=None):
        """Sample x_{t-1} from the model at the given timestep."""
        if mask is not None:
            if mask.shape[0] != x.shape[0]:
                mask = mask.repeat(2, 1)
            mask_t = (mask * len(self.betas)).to(torch.int)

            x0 = x.clone()
            x_noise = x0 * _extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) + torch.randn_like(
                x
            ) * _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape)

            mask_t_equall = (mask_t == t.unsqueeze(1))[:, None, :, None, None]
            x = torch.where(mask_t_equall, x_noise, x0)

            mask_t_upper = (mask_t > t.unsqueeze(1))[:, None, :, None, None]
            batch_size = x.shape[0]
            model_kwargs["x_mask"] = mask_t_upper.reshape(batch_size, -1).to(torch.bool)

        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        noise = th.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

        if mask is not None:
            mask_t_lower = (mask_t < t.unsqueeze(1))[:, None, :, None, None]
            sample = torch.where(mask_t_lower, x0, sample)

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(self, model, shape, noise=None, clip_denoised=True, model_kwargs=None, device=None, progress=False, mask=None):
        """Generate samples from the model via ancestral sampling."""
        final = None
        for sample in self.p_sample_loop_progressive(
            model, shape, noise=noise, clip_denoised=clip_denoised, model_kwargs=model_kwargs,
            device=device, progress=progress, mask=mask,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(self, model, shape, noise=None, clip_denoised=True, model_kwargs=None, device=None, progress=False, mask=None):
        if device is None:
            device = next(model.parameters()).device
        img = noise if noise is not None else th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(model, img, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs, mask=mask)
                yield out
                img = out["sample"]

    def _vb_terms_bpd(self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None, mask=None):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
        out = self.p_mean_variance(model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        kl = normal_kl(true_mean, true_log_variance_clipped, out["mean"], out["log_variance"])
        kl = mean_flat(kl, mask=mask) / np.log(2.0)
        decoder_nll = -discretized_gaussian_log_likelihood(x_start, means=out["mean"], log_scales=0.5 * out["log_variance"])
        decoder_nll = mean_flat(decoder_nll, mask=mask) / np.log(2.0)
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, timestep, model_kwargs=None, noise=None, mask=None):
        """Compute training losses for a single timestep."""
        t = timestep
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        if mask is not None:
            t0 = torch.zeros_like(t)
            x_t0 = self.q_sample(x_start, t0, noise=noise)
            x_t = torch.where(mask[:, None, :, None, None], x_t, x_t0)

        terms = {}

        if self.loss_type in [LossType.KL, LossType.RESCALED_KL]:
            terms["loss"] = self._vb_terms_bpd(
                model=model, x_start=x_start, x_t=x_t, t=t, clip_denoised=False, model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type in [LossType.MSE, LossType.RESCALED_MSE]:
            model_output = model(x_t, t, **model_kwargs)
            if isinstance(model_output, dict) and model_output.get("x", None) is not None:
                output = model_output["x"]
            else:
                output = model_output

            if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
                B, C = x_t.shape[:2]
                assert output.shape == (B, C * 2, *x_t.shape[2:])
                output, model_var_values = th.split(output, C, dim=1)
                frozen_out = th.cat([output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out, **kwargs: r,
                    x_start=x_start, x_t=x_t, t=t, clip_denoised=False, mask=mask,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert output.shape == target.shape == x_start.shape
            loss = (target - output) ** 2
            terms["mse"] = mean_flat(loss, mask=mask)
            terms["loss"] = terms["mse"] + terms["vb"] if "vb" in terms else terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms


# ---------------------------------------------------------------------------
# SpacedDiffusion (timestep sub-sampling)
# ---------------------------------------------------------------------------

def space_timesteps(num_timesteps, section_counts):
    """Create a set of timesteps to use from an original diffusion process."""
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(f"cannot create exactly {num_timesteps} steps with an integer stride")
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"cannot divide section of {size} steps into {section_count}")
        frac_stride = 1 if section_count <= 1 else (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class _WrappedModel:
    """Maps SpacedDiffusion timestep indices back to the original schedule."""

    def __init__(self, model, timestep_map, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.original_num_steps = original_num_steps

    def __call__(self, x, timestep, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=timestep.device, dtype=timestep.dtype)
        new_ts = map_tensor[timestep]
        return self.model(x, timestep=new_ts, **kwargs)


class SpacedDiffusion(GaussianDiffusion):
    """A diffusion process that sub-samples timesteps from a base schedule."""

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(self, model, *args, **kwargs):
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(model, self.timestep_map, self.original_num_steps)

    def _scale_timesteps(self, t):
        return t


# ---------------------------------------------------------------------------
# IDDPM (public API)
# ---------------------------------------------------------------------------

class IDDPM(SpacedDiffusion):
    """Improved DDPM scheduler with classifier-free guidance sampling."""

    def __init__(
        self,
        num_sampling_steps=None,
        timestep_respacing=None,
        noise_schedule="linear",
        use_kl=False,
        sigma_small=False,
        predict_xstart=False,
        learn_sigma=True,
        rescale_learned_sigmas=False,
        diffusion_steps=1000,
        cfg_scale=4.0,
        cfg_channel=None,
    ):
        betas = get_named_beta_schedule(noise_schedule, diffusion_steps)
        if use_kl:
            loss_type = LossType.RESCALED_KL
        elif rescale_learned_sigmas:
            loss_type = LossType.RESCALED_MSE
        else:
            loss_type = LossType.MSE
        if num_sampling_steps is not None:
            assert timestep_respacing is None
            timestep_respacing = str(num_sampling_steps)
        if timestep_respacing is None or timestep_respacing == "":
            timestep_respacing = [diffusion_steps]
        super().__init__(
            use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
            betas=betas,
            model_mean_type=(ModelMeanType.EPSILON if not predict_xstart else ModelMeanType.START_X),
            model_var_type=(
                (ModelVarType.FIXED_LARGE if not sigma_small else ModelVarType.FIXED_SMALL)
                if not learn_sigma
                else ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
        )

        self.cfg_scale = cfg_scale
        self.cfg_channel = cfg_channel

    def sample(
        self,
        model,
        device,
        y_embedder,
        z_size=None,
        z=None,
        prompts=None,
        caption_feature=None,
        attention_mask=None,
        additional_args=None,
        dtype=torch.float32,
        text_encoder=None,
        num_frames=None,
        mask=None,
    ):
        """Generate samples using DDPM ancestral sampling with classifier-free guidance."""
        n = len(prompts)
        if num_frames is not None:
            n = n // num_frames
        if z is None:
            z = torch.randn(n, *z_size, device=device, dtype=dtype)
        n = len(prompts)
        z = torch.cat([z, z], 0)

        if text_encoder is not None:
            assert prompts is not None
            model_args = text_encoder.encode(prompts)
            y_null = y_embedder.y_embedding[None].repeat(n, 1, 1)[:, None]
            model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        else:
            assert caption_feature is not None and attention_mask is not None
            y_null = y_embedder.y_embedding[None].repeat(
                caption_feature.shape[0] * caption_feature.shape[1], 1, 1
            )[:, None].reshape(*caption_feature.shape)
            model_args = {}
            model_args["y"] = torch.cat([caption_feature, y_null], 0)
            model_args["mask"] = attention_mask

        if additional_args is not None:
            model_args.update(additional_args)

        forward = partial(forward_with_cfg, model, cfg_scale=self.cfg_scale, cfg_channel=self.cfg_channel)
        samples = self.p_sample_loop(
            forward,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_args,
            progress=True,
            device=device,
            mask=mask,
        )
        samples, _ = samples.chunk(2, dim=0)
        return samples


def forward_with_cfg(model, x, timestep, y, cfg_scale, cfg_channel=None, **kwargs):
    """Model forward pass with classifier-free guidance applied."""
    half = x[: len(x) // 2]
    combined = torch.cat([half, half], dim=0)
    if "x_mask" in kwargs and kwargs["x_mask"] is not None:
        if len(kwargs["x_mask"]) != len(x):
            kwargs["x_mask"] = torch.cat([kwargs["x_mask"], kwargs["x_mask"]], dim=0)
    model_out = model.forward(combined, timestep, y, **kwargs)
    model_out = model_out["x"] if isinstance(model_out, dict) else model_out
    if cfg_channel is None:
        cfg_channel = model_out.shape[1] // 2
    eps, rest = model_out[:, :cfg_channel], model_out[:, cfg_channel:]
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    eps = torch.cat([half_eps, half_eps], dim=0)
    return torch.cat([eps, rest], dim=1)
