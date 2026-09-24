from __future__ import annotations

import argparse
import warnings
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Silence PyTorch AMP deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch\.cuda\.amp.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*GradScaler.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*autocast.*")

from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    RandFlipd,
    RandSpatialCropd,
    ScaleIntensityRanged,
    SpatialPadd,
    ThresholdIntensityd,
)
import numpy as np
import pandas as pd

from ct_mar.inference.models import VQVAE
from ct_mar.training.discriminator import PatchDiscriminator
from ct_mar.inference.transforms import CT_HU_MAX_METAL, CT_HU_MIN


def get_clean_dataloader(csv_path: str | Path, batch_size: int = 15, roi: tuple = (128, 128, 128), num_workers: int = 4) -> DataLoader:
    df = pd.read_csv(csv_path)
    col = "clean_image_path" if "clean_image_path" in df.columns else df.columns[0]
    data = [{"image": str(row[col])} for _, row in df.iterrows() if str(row[col]).strip()]

    transforms = Compose(
        [
            LoadImaged(keys=["image"], image_only=True),
            EnsureChannelFirstd(keys=["image"]),
            Lambdad(keys=["image"], func=lambda x: np.nan_to_num(x, nan=0.0, posinf=CT_HU_MAX_METAL, neginf=CT_HU_MIN)),
            ThresholdIntensityd(keys=["image"], threshold=CT_HU_MAX_METAL, above=False, cval=CT_HU_MAX_METAL),
            ThresholdIntensityd(keys=["image"], threshold=CT_HU_MIN, above=True, cval=CT_HU_MIN),
            ScaleIntensityRanged(keys=["image"], a_min=CT_HU_MIN, a_max=CT_HU_MAX_METAL, b_min=-1.0, b_max=1.0, clip=True),
            SpatialPadd(keys=["image"], spatial_size=roi, mode="constant", constant_values=-1.0),
            RandFlipd(keys=["image"], spatial_axis=1, prob=0.5),
            RandSpatialCropd(keys=["image"], roi_size=roi, random_size=False),
            EnsureTyped(keys=["image"], data_type="tensor", track_meta=False),
            Lambdad(keys=["image"], func=lambda x: x.as_tensor() if hasattr(x, "as_tensor") else torch.as_tensor(x)),
        ]
    )
    dataset = Dataset(data=data, transform=transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 1 VQ-VAE-GAN on clean 3D CT volumes.")
    parser.add_argument("--config_file", type=Path, required=True, help="Path to VQGAN YAML config.")
    parser.add_argument("--train_ids", type=Path, required=True, help="CSV manifest with clean CT image paths.")
    parser.add_argument("--output_dir", type=Path, default=Path("runs/vqgan"), help="Directory to save checkpoints.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--n_epochs", type=int, default=222, help="Number of training epochs.")
    parser.add_argument("--adv_start", type=int, default=25, help="Epoch to start adversarial training.")
    parser.add_argument("--eval_freq", type=int, default=5, help="Evaluation and checkpoint interval.")
    parser.add_argument("--roi", type=int, nargs=3, default=[128, 128, 128], help="Spatial crop size.")
    parser.add_argument("--num_workers", type=int, default=4, help="Worker count.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = VQVAE(**cfg["stage1"]["params"]).to(device)
    disc = PatchDiscriminator(**cfg.get("discriminator", {}).get("params", {"spatial_dims": 3, "num_channels": 64, "num_layers_d": 3, "in_channels": 1, "out_channels": 1})).to(device)

    opt_g = torch.optim.Adam(model.parameters(), lr=float(cfg["stage1"].get("base_lr", 3e-5)))
    opt_d = torch.optim.Adam(disc.parameters(), lr=float(cfg["stage1"].get("disc_lr", 1e-4)))

    scaler_g = GradScaler(enabled=device.type == "cuda")
    scaler_d = GradScaler(enabled=device.type == "cuda")

    loader = get_clean_dataloader(args.train_ids, batch_size=args.batch_size, roi=tuple(args.roi), num_workers=args.num_workers)

    print(f"Starting VQ-VAE-GAN training: epochs={args.n_epochs}, device={device}")
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        disc.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.n_epochs}")

        for batch in pbar:
            images = batch["image"]
            if hasattr(images, "as_tensor"):
                images = images.as_tensor()
            images = images.to(device)

            # Generator step
            opt_g.zero_grad()
            with autocast(enabled=device.type == "cuda"):
                recon, quant_loss = model(images)
                l1_loss = F.l1_loss(recon, images)
                loss_g = l1_loss + quant_loss

                if epoch >= args.adv_start:
                    logits_fake = disc(recon.contiguous())
                    loss_adv = F.binary_cross_entropy_with_logits(logits_fake, torch.ones_like(logits_fake))
                    loss_g = loss_g + float(cfg["stage1"].get("adv_weight", 0.005)) * loss_adv

            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            # Discriminator step
            if epoch >= args.adv_start:
                opt_d.zero_grad()
                with autocast(enabled=device.type == "cuda"):
                    logits_real = disc(images.contiguous())
                    logits_fake = disc(recon.detach().contiguous())
                    d_loss_real = F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))
                    d_loss_fake = F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))
                    loss_d = (d_loss_real + d_loss_fake) * 0.5

                scaler_d.scale(loss_d).backward()
                scaler_d.step(opt_d)
                scaler_d.update()

            pbar.set_postfix({"l1": f"{l1_loss.item():.4f}", "quant": f"{quant_loss.item():.4f}"})

        if epoch % args.eval_freq == 0:
            torch.save(model.state_dict(), str(args.output_dir / f"vqvae_epoch_{epoch}.pth"))
            torch.save(model.state_dict(), str(args.output_dir / "latest_model.pth"))

    torch.save(model.state_dict(), str(args.output_dir / "final_model.pth"))
    print(f"VQ-VAE-GAN training complete! Checkpoint saved to {args.output_dir / 'final_model.pth'}")


if __name__ == "__main__":
    main()

