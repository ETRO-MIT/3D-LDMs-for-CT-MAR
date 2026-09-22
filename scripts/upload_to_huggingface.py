#!/usr/bin/env python3
"""
Upload Pretrained Model Weights to Hugging Face Hub.

Usage:
    # 1. Login with your Hugging Face write token:
    #    huggingface-cli login
    #    OR pass token via --token / $HF_TOKEN

    # 2. Upload checkpoints:
    python scripts/upload_to_huggingface.py \
        --repo_id ETRO-MIT/3D-LDMs-for-CT-MAR \
        --vqvae_path /path/to/vqvae_checkpoint.pth \
        --anatomy_path /path/to/anatomy_ldm_checkpoint.pth \
        --metadata_path /path/to/anatomy_metadata_ldm_checkpoint.pth
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload 3D-LDMs-for-CT-MAR weights to Hugging Face Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="ETRO-MIT/3D-LDMs-for-CT-MAR",
        help="Target Hugging Face model repository ID.",
    )
    parser.add_argument(
        "--vqvae_path",
        type=Path,
        default=None,
        help="Local path to Stage 1 VQ-VAE checkpoint (.pth).",
    )
    parser.add_argument(
        "--anatomy_path",
        type=Path,
        default=None,
        help="Local path to Model 2.1 Anatomy LDM checkpoint (.pth).",
    )
    parser.add_argument(
        "--metadata_path",
        type=Path,
        default=None,
        help="Local path to Model 2.2 Anatomy+Metadata LDM checkpoint (.pth).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face write token (optional if logged in via huggingface-cli).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repository as private if it does not exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Error: 'huggingface_hub' package is required. Install it via:")
        print("    pip install huggingface_hub")
        return 1

    api = HfApi(token=args.token)

    # 1. Ensure repository exists
    print(f"Checking repository: {args.repo_id}...")
    try:
        create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True, token=args.token)
        print(f"Repository ready: https://huggingface.co/{args.repo_id}")
    except Exception as exc:
        print(f"Note on repository check: {exc}")

    # 2. File map
    files_to_upload = {}
    if args.vqvae_path and args.vqvae_path.exists():
        files_to_upload["vqvae_checkpoint.pth"] = args.vqvae_path
    if args.anatomy_path and args.anatomy_path.exists():
        files_to_upload["anatomy_ldm_checkpoint.pth"] = args.anatomy_path
    if args.metadata_path and args.metadata_path.exists():
        files_to_upload["anatomy_metadata_ldm_checkpoint.pth"] = args.metadata_path

    if not files_to_upload:
        print("\nNo valid checkpoint paths were provided or files were not found.")
        print("Please provide at least one valid path via:")
        print("  --vqvae_path /path/to/checkpoint.pth")
        print("  --anatomy_path /path/to/checkpoint.pth")
        print("  --metadata_path /path/to/checkpoint.pth")
        return 1

    print(f"\nUploading {len(files_to_upload)} checkpoint(s) to https://huggingface.co/{args.repo_id}:")
    for repo_name, local_path in files_to_upload.items():
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"\n-> Uploading '{local_path.name}' as '{repo_name}' ({size_mb:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_name,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Upload {repo_name}",
        )
        print(f"   Done: {repo_name}")

    print("\n" + "=" * 60)
    print(f"All files successfully uploaded!")
    print(f"View on Hugging Face: https://huggingface.co/{args.repo_id}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
