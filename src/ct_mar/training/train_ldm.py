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

from ct_mar.inference.models import (
    VQVAE,
    DiffusionModelUNet,
    MetadataConditioningEncoder,
    MetadataConditionedDiffusionModel,
    DDPMScheduler,
)
from ct_mar.training.dataset import get_mar_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D Latent Diffusion Model for CT Metal Artifact Suppression.")
    parser.add_argument("--model", choices=["anatomy", "anatomy_metadata"], default="anatomy", help="Model variant to train.")
    parser.add_argument("--config_file", type=Path, required=True, help="Path to LDM training YAML config.")
    parser.add_argument("--config_vqvae", type=Path, required=True, help="Path to VQVAE YAML config.")
    parser.add_argument("--vqvae_ckpt", type=Path, required=True, help="Pretrained VQVAE checkpoint.")
    parser.add_argument("--train_ids", type=Path, required=True, help="CSV manifest for training pairs.")
    parser.add_argument("--val_ids", type=Path, default=None, help="CSV manifest for validation pairs.")
    parser.add_argument("--output_dir", type=Path, default=Path("runs/ldm"), help="Directory where checkpoints will be saved.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per training step.")
    parser.add_argument("--n_epochs", type=int, default=250, help="Total training epochs.")
    parser.add_argument("--eval_freq", type=int, default=5, help="Epoch frequency for validation and checkpointing.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--scale_factor", type=float, default=1.0, help="Latent scaling factor.")
    parser.add_argument("--roi", type=int, nargs=3, default=[448, 448, 256], help="3D crop spatial dimensions.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load VQ-VAE (frozen)
    with open(args.config_vqvae, "r", encoding="utf-8") as f:
        vq_cfg = yaml.safe_load(f)
    stage1 = VQVAE(**vq_cfg["stage1"]["params"]).to(device)
    ckpt = torch.load(args.vqvae_ckpt, map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    stage1.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
    stage1.eval()
    for param in stage1.parameters():
        param.requires_grad = False

    # 2. Load LDM architecture
    with open(args.config_file, "r", encoding="utf-8") as f:
        ldm_cfg = yaml.safe_load(f)
    ldm_params = ldm_cfg["ldm"]["params"]

    if args.model == "anatomy_metadata":
        backbone = DiffusionModelUNet(**ldm_params)
        meta_cfg = ldm_cfg["ldm"].get("metadata_conditioning", {})
        ctx_dim = int(meta_cfg.get("context_dim", ldm_params.get("cross_attention_dim", 128)))
        encoder = MetadataConditioningEncoder(
            context_dim=ctx_dim,
            region_vocab=meta_cfg.get("region_vocab"),
            side_vocab=meta_cfg.get("side_vocab"),
            metal_name_vocab=meta_cfg.get("metal_name_vocab"),
        )
        model = MetadataConditionedDiffusionModel(diffusion_model=backbone, metadata_encoder=encoder).to(device)
        is_metadata = True
    else:
        model = DiffusionModelUNet(**ldm_params).to(device)
        encoder = None
        is_metadata = False

    # 3. Scheduler & Optimizer
    sched_cfg = ldm_cfg["ldm"].get("scheduler", {})
    scheduler = DDPMScheduler(**sched_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr or ldm_cfg["ldm"].get("base_lr", 2e-5), weight_decay=1e-4)
    scaler = GradScaler(enabled=device.type == "cuda")

    # 4. Data Loaders
    roi_tuple = tuple(args.roi)
    train_loader = get_mar_dataloader(args.train_ids, batch_size=args.batch_size, image_roi=roi_tuple, train=True, num_workers=args.num_workers)
    val_loader = None
    if args.val_ids and args.val_ids.exists():
        val_loader = get_mar_dataloader(args.val_ids, batch_size=args.batch_size, image_roi=roi_tuple, train=False, num_workers=args.num_workers)

    best_val_loss = float("inf")
    print(f"Starting LDM training: variant={args.model}, epochs={args.n_epochs}, device={device}")

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.n_epochs}")

        for batch in pbar:
            cond_images = batch["synthetic_image"]
            if hasattr(cond_images, "as_tensor"):
                cond_images = cond_images.as_tensor()
            cond_images = cond_images.to(device)

            target_images = batch["implant_only_image"]
            if hasattr(target_images, "as_tensor"):
                target_images = target_images.as_tensor()
            target_images = target_images.to(device)

            with torch.no_grad():
                target_latent = stage1.encode_stage_2_inputs(target_images) * args.scale_factor
                cond_latent = stage1.encode_stage_2_inputs(cond_images) * args.scale_factor

            # Sample random timesteps
            b = target_latent.shape[0]
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (b,), device=device).long()
            noise = torch.randn_like(target_latent)
            noisy_target = scheduler.add_noise(target_latent, noise, timesteps)

            # Condition input
            model_input = torch.cat([noisy_target, cond_latent], dim=1)

            optimizer.zero_grad()
            with autocast(enabled=device.type == "cuda"):
                if is_metadata:
                    tokens = encoder.encode_tokens_from_strings(
                        regions=batch["region"],
                        sides=batch["side"],
                        metal_names=batch["metal_name"],
                        device=device,
                    )
                    pred = model(x=model_input, timesteps=timesteps, context=tokens)
                else:
                    pred = model(x=model_input, timesteps=timesteps)

                target = scheduler.get_velocity(target_latent, noise, timesteps)
                loss = F.smooth_l1_loss(pred.float(), target.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch} finished. Avg train loss: {avg_train_loss:.5f}")

        # Validation & checkpointing
        if epoch % args.eval_freq == 0:
            torch.save(model.state_dict(), str(args.output_dir / "latest_model.pth"))
            if val_loader:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for v_batch in val_loader:
                        c_img = v_batch["synthetic_image"]
                        if hasattr(c_img, "as_tensor"):
                            c_img = c_img.as_tensor()
                        c_img = c_img.to(device)

                        t_img = v_batch["implant_only_image"]
                        if hasattr(t_img, "as_tensor"):
                            t_img = t_img.as_tensor()
                        t_img = t_img.to(device)
                        t_lat = stage1.encode_stage_2_inputs(t_img) * args.scale_factor
                        c_lat = stage1.encode_stage_2_inputs(c_img) * args.scale_factor
                        ts = torch.randint(0, scheduler.num_train_timesteps, (t_lat.shape[0],), device=device).long()
                        ns = torch.randn_like(t_lat)
                        n_tgt = scheduler.add_noise(t_lat, ns, ts)
                        m_inp = torch.cat([n_tgt, c_lat], dim=1)
                        if is_metadata:
                            toks = encoder.encode_tokens_from_strings(v_batch["region"], v_batch["side"], v_batch["metal_name"], device)
                            v_pred = model(m_inp, timesteps=ts, context=toks)
                        else:
                            v_pred = model(m_inp, timesteps=ts)
                        v_target = scheduler.get_velocity(t_lat, ns, ts)
                        val_loss += F.smooth_l1_loss(v_pred.float(), v_target.float()).item()
                avg_val = val_loss / len(val_loader)
                print(f"Validation loss: {avg_val:.5f}")
                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    torch.save(model.state_dict(), str(args.output_dir / "best_model.pth"))
                    print(f"Saved new best model to {args.output_dir / 'best_model.pth'}")

    torch.save(model.state_dict(), str(args.output_dir / "final_model.pth"))
    print("Training complete!")


if __name__ == "__main__":
    main()

