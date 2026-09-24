#!/usr/bin/env python3
"""
Main execution script for 3D Latent Diffusion Models for CT Metal Artifact Suppression (3D-LDMs-for-CT-MAR).

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
    subparsers = parser.add_subparsers(dest="task", help="Select task: 'generate', 'suppress', 'download', 'train-vqgan', or 'train-ldm'")

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
        "--anatomy_dir",
        type=Path,
        default=None,
        help="Directory containing TotalSegmentator anatomy masks for intelligent placement. If omitted, TotalSegmentator runs automatically in a temporary directory.",
    )
    generate_parser.add_argument(
        "--keep_anatomy",
        action="store_true",
        help="Keep generated TotalSegmentator anatomy masks in output_dir/anatomy instead of auto-deleting them.",
    )
    generate_parser.add_argument(
        "--implant_library",
        type=Path,
        default=Path("data/implant_library"),
        help="Path to implant library directory (default: data/implant_library).",
    )
    generate_parser.add_argument(
        "--implant_mask",
        type=Path,
        default=None,
        help="Optional pre-aligned binary implant mask NIfTI (bypasses library selection).",
    )
    generate_parser.add_argument(
        "--implant_id",
        type=str,
        default=None,
        help="Specific implant id from the library metadata.",
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
    generate_parser.add_argument(
        "--detector_spacing",
        type=float,
        default=0.5,
        help="Detector pixel spacing in cm (default: 0.5). A value of 0.5 cm covers a 64 cm FOV, ensuring implants are not truncated.",
    )
    generate_parser.add_argument(
        "--detector_pixels",
        type=int,
        default=256,
        help="Detector grid size in pixels (default: 256).",
    )
    generate_parser.add_argument(
        "--angle_num",
        type=int,
        default=360,
        help="Number of projection angles over 360 degrees (default: 360).",
    )
    generate_parser.add_argument(
        "--photon_scale",
        type=float,
        default=1.0,
        help="Multiplier for incident photon count (default: 1.0).",
    )
    generate_parser.add_argument(
        "--sod_cm",
        type=float,
        default=30.0,
        help="Source-to-origin distance in cm (default: 30.0).",
    )
    generate_parser.add_argument(
        "--sdd_cm",
        type=float,
        default=60.0,
        help="Source-to-detector distance in cm (default: 60.0).",
    )
    generate_parser.add_argument(
        "--fdk_filter",
        type=str,
        default="ram-lak",
        help="FDK reconstruction filter (default: ram-lak).",
    )
    generate_parser.add_argument(
        "--implant_category",
        type=str,
        default=None,
        help="Implant category folder under data/implant_library (e.g. hip_implants, spine_screws).",
    )
    generate_parser.add_argument(
        "--implant_source",
        choices=["library", "primitive"],
        default="library",
        help="Source for implant: 'library' or 'primitive'.",
    )
    generate_parser.add_argument(
        "--primitive_shape",
        choices=["sphere", "cylinder"],
        default="sphere",
        help="Primitive shape if implant_source is primitive.",
    )
    generate_parser.add_argument(
        "--primitive_length",
        type=int,
        default=None,
        help="Length of cylinder in voxels if primitive_shape is cylinder.",
    )
    generate_parser.add_argument(
        "--clip",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Clip output HU to [MIN, MAX] before saving.",
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

    # --- 3. Pretrained Weights Download ---
    download_parser = subparsers.add_parser(
        "download",
        help="Download pretrained model checkpoints from Hugging Face Hub.",
    )
    download_parser.add_argument(
        "--model",
        "-m",
        choices=["all", "vqvae", "anatomy", "anatomy_metadata"],
        default="all",
        help="Model checkpoint to download ('all', 'vqvae', 'anatomy', 'anatomy_metadata').",
    )
    download_parser.add_argument(
        "--output_dir",
        "-o",
        type=Path,
        default=Path("checkpoints"),
        help="Directory to store downloaded weights (default: checkpoints).",
    )
    download_parser.add_argument(
        "--repo_id",
        type=str,
        default="xabimoreno/3D-LDMs-for-CT-MAR",
        help="Hugging Face repository ID.",
    )
    download_parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face access token (optional).",
    )
    download_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download even if files already exist.",
    )

    # --- 4. Training Pipelines ---
    train_vqgan_p = subparsers.add_parser(
        "train-vqgan",
        help="Train Stage 1 VQ-VAE-GAN model on clean CT volumes.",
        add_help=False,
    )
    train_vqgan_p.add_argument("args", nargs=argparse.REMAINDER)

    train_ldm_p = subparsers.add_parser(
        "train-ldm",
        help="Train Stage 2 3D Latent Diffusion Model.",
        add_help=False,
    )
    train_ldm_p.add_argument("args", nargs=argparse.REMAINDER)

    return parser


def run_generation(args: argparse.Namespace) -> int:
    from ct_mar.synthesis.generate import main as generate_main

    cmd_args = [
        "--image", str(args.input),
        "--output-dir", str(args.output_dir),
        "--seed", str(args.seed),
        "--metal", args.metal,
        "--metal-hu", "3000.0" if args.metal == "titanium" else "4000.0",
        "--detector-spacing", str(args.detector_spacing),
        "--detector-pixels", str(args.detector_pixels),
        "--angle-num", str(args.angle_num),
        "--photon-scale", str(args.photon_scale),
        "--sod-cm", str(args.sod_cm),
        "--sdd-cm", str(args.sdd_cm),
        "--fdk-filter", str(args.fdk_filter),
        "--implant-source", str(args.implant_source),
        "--primitive-shape", str(args.primitive_shape),
    ]
    if args.primitive_length is not None:
        cmd_args.extend(["--primitive-length", str(args.primitive_length)])
    if args.clip is not None:
        cmd_args.extend(["--clip", str(args.clip[0]), str(args.clip[1])])
    if args.implant_mask:
        cmd_args.extend(["--mask", str(args.implant_mask)])
    else:
        cmd_args.extend([
            "--implant-library", str(args.implant_library),
            "--implant-random",
        ])
        if args.anatomy_dir:
            cmd_args.extend(["--anatomy-dir", str(args.anatomy_dir)])
        if args.implant_category:
            cmd_args.extend(["--implant-category", str(args.implant_category)])
        if args.implant_id:
            cmd_args.extend(["--implant-id", str(args.implant_id)])
        if args.keep_anatomy:
            cmd_args.append("--keep-anatomy")

    sys.argv = ["ct-mar-generate"] + cmd_args
    return generate_main() or 0


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
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] == "train-vqgan":
        from ct_mar.training.train_vqgan import main as train_vqgan_main
        sys.argv = ["ct-mar-train-vqgan"] + raw_args[1:]
        return train_vqgan_main() or 0
    if raw_args and raw_args[0] == "train-ldm":
        from ct_mar.training.train_ldm import main as train_ldm_main
        sys.argv = ["ct-mar-train-ldm"] + raw_args[1:]
        return train_ldm_main() or 0

    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.task:
        parser.print_help()
        return 1

    if args.task == "generate":
        return run_generation(args)
    elif args.task == "suppress":
        return run_suppression(args)
    elif args.task == "download":
        return run_download(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
