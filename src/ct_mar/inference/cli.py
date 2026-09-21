from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .pipeline import MARPipeline


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Suppress CT metal artifacts using large-volume 3D Latent Diffusion Models."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to artifact-corrupted NIfTI image (.nii or .nii.gz).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Directory where restored NIfTI outputs and previews will be saved.",
    )
    parser.add_argument(
        "--model",
        "-m",
        choices=["anatomy", "anatomy_metadata"],
        default="anatomy",
        help="Model conditioning variant: 'anatomy' (image only) or 'anatomy_metadata' (image + metadata).",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/inference"),
        help="Directory containing inference YAML configuration files.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing pretrained model checkpoints.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of diffusion sampling steps (default: 500).",
    )
    parser.add_argument(
        "--region",
        choices=["unknown", "spine", "hip", "knee", "shoulder"],
        default=None,
        help="Anatomical region of implant for anatomy_metadata model.",
    )
    parser.add_argument(
        "--side",
        choices=["unknown", "left", "right", "midline", "bilateral"],
        default=None,
        help="Laterality of implant for anatomy_metadata model.",
    )
    parser.add_argument(
        "--metal-name",
        choices=["unknown", "iron", "titanium"],
        default=None,
        help="Implant metal material for anatomy_metadata model.",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Optional path to metadata JSON file with placement and metal info.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run inference on (e.g. 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible latent sampling.",
    )
    parser.add_argument(
        "--no-restore-shape",
        action="store_true",
        help="Output raw (448, 448, 256) volume instead of restoring to original input dimensions.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable automatic side-by-side PNG slice comparison export.",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)

    pipeline = MARPipeline.from_pretrained(
        model_type=parsed.model,
        checkpoint_dir=parsed.checkpoint_dir,
        config_dir=parsed.config_dir,
        device=parsed.device,
    )

    out_file = pipeline.suppress(
        image_path=parsed.input,
        output_dir=parsed.output_dir,
        num_inference_steps=parsed.steps,
        region=parsed.region,
        side=parsed.side,
        metal_name=parsed.metal_name,
        metadata_json=parsed.metadata_json,
        restore_original_shape=not parsed.no_restore_shape,
        save_preview=not parsed.no_preview,
        seed=parsed.seed,
        verbose=True,
    )

    print(f"Artifact suppression complete. Saved restored CT to: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

