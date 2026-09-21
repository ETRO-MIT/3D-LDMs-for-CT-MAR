from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm


class LatentDiffusionInferer(nn.Module):
    """
    Handles latent space encoding, diffusion reverse-step sampling, and decoding
    for large-volume 3D CT metal artifact reduction.
    """

    def __init__(
        self,
        scheduler: nn.Module,
        scale_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.scheduler = scheduler
        self.scale_factor = float(scale_factor)

    @torch.no_grad()
    def encode(self, autoencoder: nn.Module, image: torch.Tensor) -> torch.Tensor:
        """Compress full-resolution CT image tensor [B, 1, D, H, W] to latent representation."""
        return autoencoder.encode_stage_2_inputs(image) * self.scale_factor

    @torch.no_grad()
    def decode(self, autoencoder: nn.Module, latent: torch.Tensor) -> torch.Tensor:
        """Decompress latent representation back to CT image tensor [B, 1, D, H, W]."""
        return autoencoder.decode_stage_2_outputs(latent / self.scale_factor)

    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        autoencoder_model: nn.Module,
        diffusion_model: nn.Module,
        conditioning: torch.Tensor,
        context: torch.Tensor | None = None,
        scheduler: nn.Module | None = None,
        mode: str = "concat",
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Denoises latent noise conditioned on the artifacted CT latent (and optional metadata context),
        then decodes the restored latent back to 3D image space.

        Args:
            input_noise: Initial Gaussian noise [B, C, D, H, W] matching latent shape.
            autoencoder_model: VQVAE model for decoding.
            diffusion_model: 3D UNet or MetadataConditionedDiffusionModel.
            conditioning: Latent representation of the artifacted CT [B, C, D, H, W].
            context: Optional cross-attention tokens for metadata [B, 3, context_dim].
            scheduler: DDPMScheduler. Defaults to self.scheduler.
            mode: 'concat' (anatomy only) or 'concat_crossattn' (anatomy + metadata).
            verbose: If True, display a progress bar.

        Returns:
            Restored 3D CT tensor [B, 1, D, H, W] with values in [-1.0, 1.0].
        """
        sched = scheduler or self.scheduler
        image = input_noise
        timesteps = sched.timesteps
        device = input_noise.device

        iterator = tqdm(timesteps, desc="Sampling steps", disable=not verbose)
        for t in iterator:
            t_tensor = torch.tensor((int(t),), device=device)

            if mode in ("concat", "concat_crossattn"):
                model_input = torch.cat([image, conditioning], dim=1)
                model_output = diffusion_model(
                    model_input,
                    timesteps=t_tensor,
                    context=context if mode == "concat_crossattn" else None,
                )
            else:
                model_output = diffusion_model(
                    image,
                    timesteps=t_tensor,
                    context=context if context is not None else conditioning,
                )

            image, _ = sched.step(model_output, int(t), image)

        # Decode latent back to full-resolution CT volume
        restored = self.decode(autoencoder_model, image)
        return restored

