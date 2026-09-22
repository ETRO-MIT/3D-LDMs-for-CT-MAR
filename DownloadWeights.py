#!/usr/bin/env python3
"""
Download Pretrained Model Weights for 3D Latent Diffusion Models for CT Metal Artifact Suppression.

This script fetches the official pretrained model checkpoints from the Hugging Face Hub
and places them into the local checkpoints directory (default: `checkpoints/`).

Models available:
- Stage 1 VQ-VAE (ds4): vqvae_checkpoint.pth
- Model 2.1 Anatomy-Conditioned 3D LDM: anatomy_ldm_checkpoint.pth
- Model 2.2 Anatomy + Metadata-Conditioned 3D LDM: anatomy_metadata_ldm_checkpoint.pth

Usage:
    # Download all models
    python DownloadWeights.py

    # Download only a specific model
    python DownloadWeights.py --model anatomy
    python DownloadWeights.py --model anatomy_metadata
    python DownloadWeights.py --model vqvae

    # Specify custom destination directory
    python DownloadWeights.py --output_dir /path/to/checkpoints
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

# Official Hugging Face repository
DEFAULT_REPO_ID = "ETRO-MIT/3D-LDMs-for-CT-MAR"

# Checkpoint definitions
CHECKPOINT_FILES = {
    "vqvae": {
        "filename": "vqvae_checkpoint.pth",
        "description": "Stage 1 3D VQ-VAE (4x spatial downsampling)",
    },
    "anatomy": {
        "filename": "anatomy_ldm_checkpoint.pth",
        "description": "Model 2.1: Anatomy-Conditioned 3D LDM (image concat)",
    },
    "anatomy_metadata": {
        "filename": "anatomy_metadata_ldm_checkpoint.pth",
        "description": "Model 2.2: Anatomy + Metadata-Conditioned 3D LDM (concat + cross-attention)",
    },
}


def _format_size(size_bytes: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def _download_via_hf_hub(repo_id: str, filename: str, output_path: Path, token: str | None = None) -> bool:
    """Attempt download using huggingface_hub library if available."""
    try:
        from huggingface_hub import hf_hub_download

        print(f"Downloading {filename} via huggingface_hub...")
        cached_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Copy or symlink to target location
        import shutil
        shutil.copy2(cached_file, output_path)
        return True
    except ImportError:
        return False
    except Exception as exc:
        print(f"huggingface_hub download failed: {exc}. Falling back to direct URL download...")
        return False


def _download_via_urllib(url: str, output_path: Path, token: str | None = None) -> None:
    """Download file using urllib with visual progress reporting."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tmp")

    try:
        with urllib.request.urlopen(req) as response, open(temp_path, "wb") as out_file:
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1 MB blocks
            start_time = time.time()

            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                out_file.write(buffer)
                downloaded += len(buffer)

                elapsed = max(time.time() - start_time, 1e-5)
                speed = downloaded / elapsed

                if total_size > 0:
                    percent = min(100.0, downloaded * 100.0 / total_size)
                    status = (
                        f"\r  [{percent:5.1f}%] {_format_size(downloaded)} / {_format_size(total_size)} "
                        f"({_format_size(speed)}/s)"
                    )
                else:
                    status = f"\r  {_format_size(downloaded)} downloaded ({_format_size(speed)}/s)"

                sys.stdout.write(status)
                sys.stdout.flush()

            sys.stdout.write("\n")

        temp_path.replace(output_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def download_checkpoint(
    key: str,
    repo_id: str,
    output_dir: Path,
    token: str | None = None,
    force: bool = False,
) -> bool:
    """Download a single model checkpoint."""
    info = CHECKPOINT_FILES[key]
    filename = info["filename"]
    description = info["description"]
    target_path = output_dir / filename

    print(f"\n[{key.upper()}] {description}")
    print(f"Target file: {target_path}")

    if target_path.exists() and not force:
        size = target_path.stat().st_size
        print(f"  Checkpoint already exists ({_format_size(size)}). Use --force to re-download.")
        return True

    # 1. Try huggingface_hub library
    if _download_via_hf_hub(repo_id=repo_id, filename=filename, output_path=target_path, token=token):
        print(f"  Successfully downloaded to {target_path}")
        return True

    # 2. Direct HTTPS fallback
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    print(f"  Fetching from: {url}")
    try:
        _download_via_urllib(url=url, output_path=target_path, token=token)
        print(f"  Successfully downloaded to {target_path}")
        return True
    except Exception as exc:
        print(f"  Download error: {exc}")
        print(
            f"  Note: If the repository '{repo_id}' is private, please pass your Hugging Face "
            "token via --token <HF_TOKEN> or set the HF_TOKEN environment variable."
        )
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download pretrained model weights for CT Metal Artifact Suppression (3D-LDMs-for-CT-MAR).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        "-m",
        choices=["all", "vqvae", "anatomy", "anatomy_metadata"],
        default="all",
        help="Model checkpoint to download ('all', 'vqvae', 'anatomy', 'anatomy_metadata').",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=Path,
        default=Path("checkpoints"),
        help="Local directory to store downloaded checkpoint files.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=DEFAULT_REPO_ID,
        help="Hugging Face repository ID.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face access token (optional, or read from $HF_TOKEN).",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download even if destination file already exists.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    models_to_download = (
        list(CHECKPOINT_FILES.keys()) if args.model == "all" else [args.model]
    )

    print("=" * 70)
    print("3D-LDMs-for-CT-MAR: Pretrained Weight Downloader")
    print(f"Hugging Face Repository: https://huggingface.co/{args.repo_id}")
    print(f"Destination Directory  : {args.output_dir.resolve()}")
    print("=" * 70)

    success_count = 0
    for key in models_to_download:
        if download_checkpoint(
            key=key,
            repo_id=args.repo_id,
            output_dir=args.output_dir,
            token=args.token,
            force=args.force,
        ):
            success_count += 1

    print("\n" + "=" * 70)
    print(f"Completed: {success_count}/{len(models_to_download)} checkpoints ready.")
    print("=" * 70)

    return 0 if success_count == len(models_to_download) else 1


if __name__ == "__main__":
    sys.exit(main())

