#!/usr/bin/env python3
"""
Main execution script for 3D Latent Diffusion Models for CT Metal Artifact Suppression (3D-LDMs-for-CT-MAR).

Supports two primary tasks:
1. 'suppress': Metal artifact suppression using 3D conditioned Latent Diffusion Models.
2. 'generate': Synthetic metal artifact simulation on clean 3D CT volumes using ASTRA CUDA.
Supports:
1. 'generate': Synthetic metal artifact simulation on clean 3D CT volumes using ASTRA CUDA.
2. 'suppress': Metal artifact suppression using 3D conditioned Latent Diffusion Models (2.1 Anatomy, 2.2 Anatomy+Metadata).
3. 'download': Fetch pretrained model weights from Hugging Face Hub into local checkpoints.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python Main.py",
        description="3D Latent Diffusion Models for CT Metal Artifact Suppression (DGM4MICCAI 2026)",
    )
    subparsers = parser.add_subparsers(dest="task", help="Select task: 'suppress' or 'generate'")
    subparsers = parser.add_subparsers(dest="task", help="Select task: 'generate', 'suppress', or 'download'")

    # --- Task 1: Artifact Suppression ---
    # --- 1. Synthetic Artifact Generation ---
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate synthetic metal artifacts from clean CT volumes using ASTRA CUDA simulation.",
    )
    generate_parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to clean 3D CT volume (.nii or .nii.gz).",
    )
    generate_parser.add_argument(
        "--output_dir",
        "-o",
        type=Path,
        required=True,
        help="Directory to save simulated artifacted CT, target, and metadata.",
    )
    generate_parser.add_argument(
        "--implant_mask",
        type=Path,
        default=None,
        help="Path to implant segmentation mask NIfTI.",
    )
    generate_parser.add_argument(
        "--implant_name",
        type=str,
        default="bipolar_hip_implant",
        help="Implant name from implant library (e.g. 'bipolar_hip_implant').",
    )
    generate_parser.add_argument(
        "--metal",
        choices=["titanium", "iron"],
        default="titanium",
        help="Implant material (titanium=3000 HU, iron=4000 HU).",
    )
    generate_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for placement and noise generation.",
    )

    # --- 2. Metal Artifact Suppression (2.1 Anatomy, 2.2 Anatomy+Metadata) ---
    suppress_parser = subparsers.add_parser(
        "suppress",
        help="Suppress metal artifacts in a corrupted 3D CT volume using conditional LDMs.",
    )
    suppress_parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to artifact-corrupted NIfTI image (.nii or .nii.gz).",
    )
    suppress_parser.add_argument(
        "--output_dir",
        "-o",
        type=Path,
        required=True,
        help="Directory to save restored CT volume and preview.",
    )
    suppress_parser.add_argument(
        "--model",
        "-m",
        choices=["anatomy", "anatomy_metadata"],
        default="anatomy",
        help="Model conditioning: 'anatomy' (image-only) or 'anatomy_metadata' (image + metadata).",
    )
    suppress_parser.add_argument(
        "--config_dir",
        type=Path,
        default=Path("configs/inference"),
        help="Directory with inference YAML configs (default: configs/inference).",
    )
    suppress_parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing model checkpoints (default: checkpoints).",
    )
    suppress_parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of diffusion sampling steps (default: 500).",
    )
    suppress_parser.add_argument(
        "--region",
        choices=["unknown", "spine", "hip", "knee", "shoulder"],
        default=None,
        help="Anatomical region of implant (for anatomy_metadata model).",
    )
    suppress_parser.add_argument(
        "--side",
        choices=["unknown", "left", "right", "midline", "bilateral"],
        default=None,
        help="Implant laterality: 'left', 'right', 'midline', 'bilateral', 'unknown'.",
    )
    suppress_parser.add_argument(
        "--metal",
        choices=["unknown", "iron", "titanium"],
        default=None,
        help="Implant metal material: 'titanium', 'iron', 'unknown'.",
    )
    suppress_parser.add_argument(
        "--metadata_json",
        type=Path,
        default=None,
        help="Optional sidecar JSON metadata file.",
    )
    suppress_parser.add_argument(
        "--device",
        default=None,
        help="Device to run inference on (e.g. 'cuda:0', 'cpu').",
    )
    suppress_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling.",
    )
    suppress_parser.add_argument(
        "--no_preview",
        action="store_true",
        help="Disable automatic side-by-side PNG slice comparison export.",
    )

    # --- Task 2: Synthetic Artifact Generation ---
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate synthetic metal artifacts from clean CT volumes using ASTRA CUDA simulation.",
    # --- 3. Pretrained Weights Download ---
    download_parser = subparsers.add_parser(
        "download",
        help="Download pretrained model checkpoints from Hugging Face Hub.",
    )
    generate_parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to clean 3D CT volume (.nii or .nii.gz).",
    download_parser.add_argument(
        "--model",
        "-m",
        choices=["all", "vqvae", "anatomy", "anatomy_metadata"],
        default="all",
        help="Model checkpoint to download ('all', 'vqvae', 'anatomy', 'anatomy_metadata').",
    )
    generate_parser.add_argument(
    download_parser.add_argument(
        "--output_dir",
        "-o",
        type=Path,
        required=True,
        help="Directory to save simulated artifacted CT, target, and metadata.",
        default=Path("checkpoints"),
        help="Directory to store downloaded weights (default: checkpoints).",
    )
    generate_parser.add_argument(
        "--implant_mask",
        type=Path,
        default=None,
        help="Path to implant segmentation mask NIfTI.",
    download_parser.add_argument(
        "--repo_id",
        type=str,
        default="ETRO-MIT/3D-LDMs-for-CT-MAR",
        help="Hugging Face repository ID.",
    )
    generate_parser.add_argument(
        "--implant_name",
    download_parser.add_argument(
        "--token",
        type=str,
        default="bipolar_hip_implant",
        help="Implant name from implant library (e.g. 'bipolar_hip_implant').",
        default=None,
        help="Hugging Face access token (optional).",
    )
    generate_parser.add_argument(
        "--metal",
        choices=["titanium", "iron"],
        default="titanium",
        help="Implant material (titanium=3000 HU, iron=4000 HU).",
    download_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download even if files already exist.",
    )
    generate_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for placement and noise generation.",
    )

    return parser


def run_suppression(args: argparse.Namespace) -> int:
    from ct_mar.inference.pipeline import MARPipeline

    print(f"Loading MAR Pipeline: model={args.model}, checkpoints={args.checkpoint_dir}")
    pipeline = MARPipeline.from_pretrained(
        model_type=args.model,
        checkpoint_dir=args.checkpoint_dir,
        config_dir=args.config_dir,
        device=args.device,
    )

    out_file = pipeline.suppress(
        image_path=args.input,
        output_dir=args.output_dir,
        num_inference_steps=args.steps,
        region=args.region,
        side=args.side,
        metal_name=args.metal,
        metadata_json=args.metadata_json,
        restore_original_shape=True,
        save_preview=not args.no_preview,
        seed=args.seed,
        verbose=True,
    )

    print(f"Artifact suppression complete. Saved restored CT to: {out_file}")
    return 0


def run_generation(args: argparse.Namespace) -> int:
    from ct_mar.synthesis.generate import main as generate_main

    cmd_args = [
        "--ct_path", str(args.input),
        "--output_dir", str(args.output_dir),
        "--metal_name", str(args.metal),
        "--seed", str(args.seed),
    ]
    if args.implant_mask:
        cmd_args.extend(["--implant_mask_path", str(args.implant_mask)])
    else:
        cmd_args.extend(["--implant_name", str(args.implant_name)])

    sys.argv = ["ct-mar-generate"] + cmd_args
    return generate_main() or 0


def run_download(args: argparse.Namespace) -> int:
    from DownloadWeights import main as download_main

    cmd_args = [
        "--model", args.model,
        "--output_dir", str(args.output_dir),
        "--repo_id", args.repo_id,
    ]
    if args.token:
        cmd_args.extend(["--token", args.token])
    if args.force:
        cmd_args.append("--force")

    sys.argv = ["DownloadWeights.py"] + cmd_args
    return download_main() or 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.task:
        parser.print_help()
        return 1

    if args.task == "suppress":
    if args.task == "generate":
        return run_generation(args)
    elif args.task == "suppress":
        return run_suppression(args)
    elif args.task == "generate":
        return run_generation(args)
    elif args.task == "download":
        return run_download(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())


