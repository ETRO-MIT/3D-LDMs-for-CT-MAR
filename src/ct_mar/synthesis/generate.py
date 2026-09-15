from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from ct_mar.synthesis.config import set_config_for_artifact_simulation
from ct_mar.synthesis.anatomy import anatomy_aware_metal_mask, map_labels_to_regions
from ct_mar.synthesis.volume import calibrate_water_correction, metal_artifact_simulation_volume
from ct_mar.synthesis.geometry_astra import require_astra_gpu
from ct_mar.synthesis.utils.io import save_config_as_json


def load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray]:
    nii = nib.load(path)
    data = nii.get_fdata().astype(np.float32)
    affine = nii.affine
    if data.ndim != 3 or not np.isfinite(data).all():
        raise ValueError(f"Expected a finite 3D NIfTI volume: {path}")
    return data, affine


def save_nifti(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, path)


def voxel_spacing(affine: np.ndarray) -> np.ndarray:
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


def _strip_nii_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def load_totalseg_labels(labels_dir: Path | None, shape=None, affine=None) -> dict[str, np.ndarray] | None:
    if labels_dir is None:
        return None
    labels = {}
    for path in sorted(labels_dir.glob("*.nii*")):
        data, label_affine = load_nifti(path)
        if shape is not None and (data.shape != shape or not np.allclose(label_affine, affine)):
            raise ValueError(f"Anatomy mask must match the CT shape and affine: {path}")
        labels[_strip_nii_suffix(path)] = (data > 0).astype(np.uint8)
    if not labels:
        raise ValueError(f"No anatomy masks found in {labels_dir}")
    return labels if labels else None


def main():
    parser = argparse.ArgumentParser(description="Full 3D metal artifact simulation (cone-beam style).")
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to input CT NIfTI (HU). Use a cropped volume for speed.",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Optional binary implant mask aligned with the CT; otherwise use library or primitive placement.",
    )
    parser.add_argument(
        "--anatomy-dir",
        type=Path,
        default=None,
        help="Optional TotalSegmentator labels directory (one NIfTI per structure).",
    )
    parser.add_argument(
        "--implant-library",
        type=Path,
        default=Path("data/implant_library"),
        help="Implant library root (contains category folders with metadata.json).",
    )
    parser.add_argument(
        "--implant-category",
        type=str,
        default=None,
        help="Implant category folder to use (e.g., pelvic_screws). Defaults to region mapping.",
    )
    parser.add_argument(
        "--implant-id",
        type=str,
        default=None,
        help="Specific implant id from the library metadata.",
    )
    parser.add_argument(
        "--implant-random",
        action="store_true",
        help="Select a random implant from the chosen category.",
    )
    parser.add_argument(
        "--implant-source",
        type=str,
        default="library",
        choices=("library", "primitive"),
        help="Use implant library or a primitive mask (sphere/cylinder).",
    )
    parser.add_argument(
        "--primitive-shape",
        type=str,
        default="sphere",
        choices=("sphere", "cylinder"),
        help="Primitive shape if --implant-source primitive.",
    )
    parser.add_argument(
        "--primitive-length",
        type=int,
        default=None,
        help="Cylinder length in voxels (defaults to 2*metal-radius).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to store outputs.",
    )
    parser.add_argument(
        "--metal-hu",
        type=float,
        default=3000.0,
        help="HU value to assign inside the metal mask before simulation.",
    )
    parser.add_argument(
        "--metal-radius",
        type=int,
        default=10,
        help="Radius (voxels) of spherical metal mask if generated.",
    )
    parser.add_argument(
        "--photon-scale",
        type=float,
        default=1.0,
        help="Multiplier for expected photon counts (higher -> lower relative noise).",
    )
    parser.add_argument(
        "--angle-num",
        type=int,
        default=180,
        help="Number of gantry angles around z.",
    )
    parser.add_argument(
        "--detector-pixels",
        type=int,
        default=256,
        help="Detector grid size (square).",
    )
    parser.add_argument(
        "--detector-spacing",
        type=float,
        default=0.1,
        help="Detector pixel size (cm).",
    )
    parser.add_argument(
        "--sod-cm",
        type=float,
        default=30.0,
        help="Source-to-origin distance (cm).",
    )
    parser.add_argument(
        "--sdd-cm",
        type=float,
        default=60.0,
        help="Source-to-detector distance (cm).",
    )
    parser.add_argument(
        "--fdk-filter",
        type=str,
        default="ram-lak",
        help="FDK filter for ASTRA (e.g., ram-lak, hann, hamming).",
    )
    parser.add_argument(
        "--clip",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Clip output HU to [MIN, MAX] before saving.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for placement and projection noise.")
    args = parser.parse_args()
    np.random.seed(args.seed)
    if args.angle_num < 1 or args.detector_pixels < 2 or args.photon_scale <= 0:
        parser.error("Angles and photon scale must be positive; detector pixels must be at least 2.")
    if args.detector_spacing <= 0 or not 0 < args.sod_cm < args.sdd_cm:
        parser.error("Require positive detector spacing and 0 < sod-cm < sdd-cm.")
    if args.mask is None and args.implant_source == "library" and not args.implant_library.is_dir():
        parser.error(f"Implant library not found: {args.implant_library}")

    volume_hu, affine = load_nifti(args.image)
    spacing = voxel_spacing(affine)
    if not np.allclose(spacing, 1.0, atol=1e-3) or nib.aff2axcodes(affine) != ('R', 'A', 'S') or not np.allclose(affine[:3, :3], np.eye(3), atol=1e-3):
        parser.error("Prepare the CT in axis-aligned RAS orientation with 1 mm isotropic spacing before simulation.")
    voxel_size_cm = float(np.mean(spacing) / 10.0)  # mm -> cm

    config = set_config_for_artifact_simulation(voxel_size_cm)
    config.angle_num = args.angle_num
    config.detector_pixels = args.detector_pixels
    config.detector_spacing = args.detector_spacing
    config.SOD_cm = args.sod_cm
    config.SDD_cm = args.sdd_cm
    config.photon_scale = args.photon_scale
    config.fdk_filter = args.fdk_filter.lower()
    placement_metadata = config.placement_metadata.copy() if config.placement_metadata else {}

    if args.mask is not None:
        metal_mask, mask_affine = load_nifti(args.mask)
        if metal_mask.shape != volume_hu.shape or not np.allclose(mask_affine, affine):
            parser.error("The implant mask must match the CT shape and affine.")
        metal_mask = (metal_mask > 0).astype(np.uint8)
        placement_metadata.update(
            {
                "region": "unknown",
                "implant_source": "provided_mask",
                "selection_mode": "provided_mask",
                "mask_path": str(args.mask),
                "metal_name": config.metal_name,
                "metal_hu": float(args.metal_hu),
            }
        )
    else:
        totalseg_labels = load_totalseg_labels(args.anatomy_dir, volume_hu.shape, affine)
        region_masks = map_labels_to_regions(totalseg_labels) if totalseg_labels else None
        metal_mask, args.metal_hu, config.metal_name, placement_metadata = anatomy_aware_metal_mask(
            volume_hu,
            affine,
            metal_radius=args.metal_radius,
            rng=np.random.default_rng(args.seed),
            totalseg_labels=totalseg_labels,
            region_masks=region_masks,
            implant_library=args.implant_library,
            implant_category=args.implant_category,
            implant_id=args.implant_id,
            implant_random=args.implant_random,
            implant_source=args.implant_source,
            primitive_shape=args.primitive_shape,
            primitive_length=args.primitive_length,
        )
    config.metal_hu = float(args.metal_hu)
    if not np.any(metal_mask):
        raise ValueError("The selected implant mask is empty after placement.")
    placement_metadata["seed"] = args.seed
    require_astra_gpu()
    calibrate_water_correction(config, phantom_size=min(64, volume_hu.shape[0]), phantom_radius=min(20, volume_hu.shape[0] // 3))

    simulated_hu = metal_artifact_simulation_volume(
        volume_hu,
        config,
        metal_mask=metal_mask,
        metal_hu=args.metal_hu,
        metal_radius=args.metal_radius,
        photon_scale=args.photon_scale,
        progress=True,
    )
    if args.clip is not None:
        clip_min, clip_max = args.clip
        simulated_hu = np.clip(simulated_hu, a_min=clip_min, a_max=clip_max)

    case_id = _strip_nii_suffix(args.image)
    synth_stem = f"synth_{case_id}"
    implant_only_stem = f"implant_only_{case_id}"
    output_image_path = args.output_dir / f"{synth_stem}.nii.gz"
    output_implant_only_path = args.output_dir / f"{implant_only_stem}.nii.gz"
    output_mask_path = args.output_dir / f"{synth_stem}_metal_mask.nii.gz"
    output_json_path = args.output_dir / f"{synth_stem}.json"
    implant_only_hu = volume_hu.copy()
    implant_only_hu[metal_mask > 0] = float(args.metal_hu)
    placement_metadata.update(
        {
            "case_id": case_id,
            "input_image_path": str(args.image),
            "output_image_path": str(output_image_path),
            "implant_only_image_path": str(output_implant_only_path),
        }
    )
    config.placement_metadata = placement_metadata

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_nifti(output_implant_only_path, implant_only_hu.astype(np.float32), affine)
    save_nifti(output_image_path, simulated_hu.astype(np.float32), affine)
    save_nifti(output_mask_path, metal_mask.astype(np.uint8), affine)
    save_config_as_json(output_json_path, config)
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
