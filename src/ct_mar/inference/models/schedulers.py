# Copyright (c) MONAI Consortium / Diffusers authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _cosine_beta(num_train_timesteps: int, s: float = 8e-3) -> torch.Tensor:
    x = torch.linspace(0, num_train_timesteps, num_train_timesteps + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / num_train_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0.0, 0.9999)
    return betas.float()


def _linear_beta(num_train_timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)


class DDPMScheduler(nn.Module):
    """
    DDPMScheduler supporting cosine schedule and v_prediction for 3D LDM inference.
    """

    def __init__(
        self,
        num_train_timesteps: int = 500,
        schedule: str = "cosine",
        prediction_type: str = "v_prediction",
        clip_sample: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_train_timesteps = int(num_train_timesteps)
        self.schedule = schedule
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample

        if schedule == "cosine":
            betas = _cosine_beta(self.num_train_timesteps)
        elif schedule in ("linear", "linear_beta"):
            betas = _linear_beta(self.num_train_timesteps)
        else:
            raise ValueError(f"Unsupported schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("one", torch.tensor(1.0))

        self.num_inference_steps: int = self.num_train_timesteps
        self.timesteps: torch.Tensor = torch.arange(self.num_train_timesteps - 1, -1, -1, dtype=torch.long)

    def set_timesteps(self, num_inference_steps: int, device: torch.device | None = None) -> None:
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps` ({num_inference_steps}) cannot be larger than "
                f"`num_train_timesteps` ({self.num_train_timesteps})."
            )
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
        if device is not None:
            self.timesteps = self.timesteps.to(device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = int(timestep)
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else self.one
        beta_prod_t = 1.0 - alpha_prod_t
        beta_prod_t_prev = 1.0 - alpha_prod_t_prev

        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / (alpha_prod_t ** 0.5)
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)

        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * self.betas[t]) / beta_prod_t
        current_sample_coeff = (self.alphas[t] ** 0.5) * beta_prod_t_prev / beta_prod_t

        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        if t > 0:
            variance = torch.clamp((1.0 - alpha_prod_t_prev) / beta_prod_t * self.betas[t], min=1e-20)
            noise = torch.randn(model_output.size(), dtype=model_output.dtype, device=model_output.device, generator=generator)
            pred_prev_sample = pred_prev_sample + (variance ** 0.5) * noise

        return pred_prev_sample, pred_original_sample

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to the original samples according to the diffusion schedule."""
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1.0 - alphas_cumprod[timesteps]) ** 0.5

        while sqrt_alpha_prod.ndim < original_samples.ndim:
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the velocity target for v-prediction models."""
        alphas_cumprod = self.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        timesteps = timesteps.to(sample.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1.0 - alphas_cumprod[timesteps]) ** 0.5

        while sqrt_alpha_prod.ndim < sample.ndim:
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        return sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample

