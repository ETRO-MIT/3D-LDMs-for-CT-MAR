from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml
import torch
import torch.nn as nn
import numpy as np

from .models import (
    VQVAE,
    DiffusionModelUNet,
    MetadataConditioningEncoder,
    MetadataConditionedDiffusionModel,
    DDPMScheduler,
    parse_metadata_json,
)
from .inferer import LatentDiffusionInferer
from .transforms import (
    DEFAULT_TARGET_SIZE,
    denormalize_to_hu,
    normalize_ct_hu,
    pad_or_crop_3d,
    restore_to_original_shape,
)
from .io import export_slice_comparison_png, load_nifti, save_nifti


def _remove_module_prefix(state_dict: dict) -> dict:
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def _resolve_checkpoint(path: str | Path) -> Path:
    p = Path(path)
    if p.is_dir():
        for name in ("best_model.pth", "final_model.pth", "checkpoint.pth", "model.pth"):
            cand = p / name
            if cand.exists():
                return cand
        raise FileNotFoundError(f"No checkpoint file found in directory: {p}")
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {p}")
    return p


class MARPipeline:
    """
    Unified inference pipeline for Large-Volume 3D CT Metal Artifact Suppression.

    Supports:
    - Model 1: Anatomy-Conditioned LDM ('anatomy')
    - Model 2: Anatomy-Metadata Conditioned LDM ('anatomy_metadata')
    """

    def __init__(
        self,
        vqvae_config: str | Path,
        vqvae_checkpoint: str | Path | None,
        ldm_config: str | Path,
        ldm_checkpoint: str | Path | None,
        model_type: str = "anatomy",
        device: str | torch.device | None = None,
        scale_factor: float = 1.0,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.scale_factor = scale_factor
        self.model_type = model_type.lower()

        # Load VQ-VAE
        with open(vqvae_config, "r", encoding="utf-8") as f:
            v_cfg = yaml.safe_load(f)
        self.vqvae = VQVAE(**v_cfg["stage1"]["params"]).to(self.device)
        if vqvae_checkpoint is not None:
            ckpt_path = _resolve_checkpoint(vqvae_checkpoint)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            self.vqvae.load_state_dict(_remove_module_prefix(state), strict=False)
        self.vqvae.eval()

        # Load LDM
        with open(ldm_config, "r", encoding="utf-8") as f:
            l_cfg = yaml.safe_load(f)
        ldm_params = l_cfg["ldm"]["params"]

        if self.model_type in ("anatomy_metadata", "metadata", "concat_crossattn"):
            backbone = DiffusionModelUNet(**ldm_params)
            meta_cfg = l_cfg["ldm"].get("metadata_conditioning", {})
            ctx_dim = int(meta_cfg.get("context_dim", ldm_params.get("cross_attention_dim", 128)))
            self.meta_encoder = MetadataConditioningEncoder(
                context_dim=ctx_dim,
                region_vocab=meta_cfg.get("region_vocab"),
                side_vocab=meta_cfg.get("side_vocab"),
                metal_name_vocab=meta_cfg.get("metal_name_vocab"),
            )
            self.ldm = MetadataConditionedDiffusionModel(
                diffusion_model=backbone,
                metadata_encoder=self.meta_encoder,
            ).to(self.device)
            self.mode = "concat_crossattn"
        else:
            self.meta_encoder = None
            self.ldm = DiffusionModelUNet(**ldm_params).to(self.device)
            self.mode = "concat"

        if ldm_checkpoint is not None:
            ckpt_path = _resolve_checkpoint(ldm_checkpoint)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if isinstance(ckpt, dict) and "diffusion" in ckpt:
                state = ckpt["diffusion"]
            else:
                state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            self.ldm.load_state_dict(_remove_module_prefix(state), strict=False)
        self.ldm.eval()

        # Scheduler and Inferer
        sched_params = l_cfg["ldm"].get("scheduler", {})
        self.scheduler = DDPMScheduler(**sched_params)
        self.inferer = LatentDiffusionInferer(scheduler=self.scheduler, scale_factor=self.scale_factor)

    @classmethod
    def from_pretrained(
        cls,
        model_type: str = "anatomy",
        checkpoint_dir: str | Path = "checkpoints",
        config_dir: str | Path = "configs/inference",
        device: str | torch.device | None = None,
    ) -> MARPipeline:
        """Convenience loader looking for default YAMLs and checkpoint filenames."""
        cfg_dir = Path(config_dir)
        ckpt_dir = Path(checkpoint_dir)

        vqvae_cfg = cfg_dir / "vqvae_ds4.yaml"
        vqvae_ckpt = ckpt_dir / "vqvae_checkpoint.pth"

        if model_type.lower() in ("anatomy_metadata", "metadata"):
            ldm_cfg = cfg_dir / "anatomy_metadata_ldm.yaml"
            ldm_ckpt = ckpt_dir / "anatomy_metadata_ldm_checkpoint.pth"
        else:
            ldm_cfg = cfg_dir / "anatomy_ldm.yaml"
            ldm_ckpt = ckpt_dir / "anatomy_ldm_checkpoint.pth"

        return cls(
            vqvae_config=vqvae_cfg,
            vqvae_checkpoint=vqvae_ckpt if vqvae_ckpt.exists() else None,
            ldm_config=ldm_cfg,
            ldm_checkpoint=ldm_ckpt if ldm_ckpt.exists() else None,
            model_type=model_type,
            device=device,
        )

    def suppress(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        num_inference_steps: int = 500,
        region: str | None = None,
        side: str | None = None,
        metal_name: str | None = None,
        metadata_json: str | Path | None = None,
        restore_original_shape: bool = True,
        save_preview: bool = True,
        seed: int | None = None,
        verbose: bool = True,
    ) -> Path:
        """
        Run metal artifact suppression on an input 3D CT volume.

        Args:
            image_path: Path to the artifacted NIfTI volume (.nii or .nii.gz).
            output_dir: Directory where restored output is written.
            num_inference_steps: Diffusion sampling steps (e.g. 500 or faster for preview).
            region: Anatomical region ('hip', 'spine', 'knee', 'shoulder', 'unknown').
            side: Implant laterality ('left', 'right', 'midline', 'bilateral', 'unknown').
            metal_name: Material ('titanium', 'iron', 'unknown').
            metadata_json: Optional sidecar JSON file containing placement metadata.
            restore_original_shape: If True, place restored crop back in original dimensions.
            save_preview: If True, export a side-by-side axial slice comparison PNG.
            seed: Optional random seed for reproducible sampling.
            verbose: If True, display progress bar.

        Returns:
            Path to the saved restored NIfTI volume.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        input_p = Path(image_path)
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        # 1. Load input CT
        vol_hu, affine, header = load_nifti(input_p)

        # 2. Normalize to [-1.0, 1.0] and crop/pad to (448, 448, 256)
        vol_norm = normalize_ct_hu(vol_hu)
        crop_norm, meta = pad_or_crop_3d(vol_norm, DEFAULT_TARGET_SIZE)

        # 3. Prepare PyTorch tensors [B=1, C=1, D, H, W]
        cond_tensor = torch.from_numpy(crop_norm).unsqueeze(0).unsqueeze(0).to(self.device)

        # 4. Prepare metadata context if using Model 2
        context = None
        if self.mode == "concat_crossattn":
            if metadata_json is not None or input_p.with_suffix(".json").exists():
                json_file = metadata_json or (
                    input_p.name.removesuffix(".nii.gz").removesuffix(".nii") + ".json"
                )
                if Path(json_file).exists():
                    extracted = parse_metadata_json(json_file)
                    region = region or extracted["region"]
                    side = side or extracted["side"]
                    metal_name = metal_name or extracted["metal_name"]

            reg = region or "unknown"
            sd = side or "unknown"
            mtl = metal_name or "unknown"

            context = self.meta_encoder.encode_tokens_from_strings(
                regions=[reg],
                sides=[sd],
                metal_names=[mtl],
                device=self.device,
            )

        # 5. Set inference steps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # 6. Encode condition latent and generate initial noise
        with torch.no_grad():
            cond_latent = self.inferer.encode(self.vqvae, cond_tensor)
            input_noise = torch.randn_like(cond_latent)

            # 7. Sample restored volume
            restored_tensor = self.inferer.sample(
                input_noise=input_noise,
                autoencoder_model=self.vqvae,
                diffusion_model=self.ldm,
                conditioning=cond_latent,
                context=context,
                scheduler=self.scheduler,
                mode=self.mode,
                verbose=verbose,
            )

        restored_crop = restored_tensor[0, 0].detach().cpu().numpy()
        restored_hu_crop = denormalize_to_hu(restored_crop)

        # 8. Shape restoration
        if restore_original_shape:
            final_hu = restore_to_original_shape(
                processed_volume=restored_hu_crop,
                original_shape=vol_hu.shape,
                meta=meta,
                fill_from=vol_hu,
            )
        else:
            final_hu = restored_hu_crop

        # 9. Save output NIfTI
        stem = input_p.name.removesuffix(".nii.gz").removesuffix(".nii")
        output_file = out_p / f"{stem}_restored_{self.model_type}.nii.gz"
        save_nifti(output_file, final_hu, affine=affine, header=header)

        # 10. Optional slice preview PNG
        if save_preview:
            preview_file = out_p / f"{stem}_preview_{self.model_type}.png"
            export_slice_comparison_png(
                preview_file,
                original_hu=vol_hu,
                restored_hu=final_hu,
            )

        return output_file

