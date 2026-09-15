"""
build_implant_library.py

Create implant masks from CT volumes and store them in the implant library.
This extracts high-HU metal, optionally cleans it, crops to the tight bounding
box, and writes metadata for later selection.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import nibabel as nib
import numpy as np
from nibabel.affines import voxel_sizes
from scipy import ndimage


def load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray]:
    nii = nib.load(path)
    data = nii.get_fdata().astype(np.float32)
    return data, nii.affine


def save_nifti(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), path)


def _parse_csv_list(text: str | None) -> list[str] | None:
    if text is None:
        return None
    items = [item.strip() for item in text.split(",") if item.strip()]
    return items or None


def _parse_range(text: str | None) -> tuple[float, float] | None:
    if text is None:
        return None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Range must be two comma-separated values, e.g. 10,60")
    return float(parts[0]), float(parts[1])


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = counts.argmax()
    return (labeled == keep).astype(np.uint8)


def _filter_small_components(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    counts = np.bincount(labeled.ravel())
    keep = (labeled > 0) & np.isin(labeled, np.where(counts >= min_voxels)[0])
    return keep.astype(np.uint8)


def _bounding_box(mask: np.ndarray) -> tuple[slice, slice, slice]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        raise ValueError("No metal voxels found after filtering.")
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return slice(mins[0], maxs[0]), slice(mins[1], maxs[1]), slice(mins[2], maxs[2])


def _expand_slices(slices: tuple[slice, slice, slice], shape: tuple[int, int, int], margin: int) -> tuple[slice, slice, slice]:
    expanded = []
    for slc, dim in zip(slices, shape):
        start = max(0, slc.start - margin)
        stop = min(dim, slc.stop + margin)
        expanded.append(slice(start, stop))
    return tuple(expanded)  # type: ignore[return-value]


def _local_affine(spacing: Tuple[float, float, float]) -> np.ndarray:
    affine = np.eye(4, dtype=np.float32)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]
    return affine


def _resample_isotropic(
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    target_mm: float,
) -> tuple[np.ndarray, Tuple[float, float, float]]:
    zoom_factors = np.array(spacing, dtype=np.float32) / float(target_mm)
    resampled = ndimage.zoom(mask.astype(np.float32), zoom_factors, order=0)
    resampled = (resampled > 0.5).astype(np.uint8)
    return resampled, (float(target_mm), float(target_mm), float(target_mm))


def _pad_to_cube(mask: np.ndarray, pad_value: int = 0) -> np.ndarray:
    max_dim = int(max(mask.shape))
    pad_width = []
    for dim in mask.shape:
        total = max_dim - dim
        before = total // 2
        after = total - before
        pad_width.append((before, after))
    return np.pad(mask, pad_width, mode="constant", constant_values=pad_value)


def _fill_holes(mask: np.ndarray, closing_iter: int = 2) -> np.ndarray:
    if closing_iter > 0:
        mask = ndimage.binary_closing(mask, iterations=closing_iter)
    return ndimage.binary_fill_holes(mask).astype(np.uint8)


def _rotation_to_z(axis: np.ndarray) -> np.ndarray:
    axis = axis.astype(np.float64)
    norm = np.linalg.norm(axis)
    if norm == 0:
        return np.eye(3, dtype=np.float64)
    v = axis / norm
    target = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if np.allclose(v, target):
        return np.eye(3, dtype=np.float64)
    if np.allclose(v, -target):
        # 180 degree rotation around X axis
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    k = np.cross(v, target)
    k_norm = np.linalg.norm(k)
    if k_norm == 0:
        return np.eye(3, dtype=np.float64)
    k = k / k_norm
    cos_t = np.dot(v, target)
    sin_t = k_norm
    K = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
        dtype=np.float64,
    )
    R = np.eye(3, dtype=np.float64) + sin_t * K + (1.0 - cos_t) * (K @ K)
    return R


def _pca_align_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(mask > 0)
    if coords.shape[0] < 10:
        return mask, np.array([0.0, 0.0, 1.0], dtype=np.float64)
    center = coords.mean(axis=0)
    centered = coords - center
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_axis = eigvecs[:, np.argmax(eigvals)]
    R = _rotation_to_z(principal_axis)
    Rt = R.T
    offset = center - Rt @ center
    rotated = ndimage.affine_transform(
        mask.astype(np.float32),
        Rt,
        offset=offset,
        order=0,
        mode="constant",
        cval=0.0,
    )
    rotated = (rotated > 0.5).astype(np.uint8)
    return rotated, principal_axis


def _update_metadata(
    metadata_path: Path,
    implant_type: str,
    item: Dict[str, Any],
    typical_regions: list[str] | None,
    materials: list[str] | None,
    preferred_axis: str | None,
    scale_mm_range: tuple[float, float] | None,
    notes: str | None,
) -> None:
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    else:
        metadata = {}

    metadata.setdefault("implant_type", implant_type)
    if typical_regions and "typical_regions" not in metadata:
        metadata["typical_regions"] = typical_regions
    if materials and "materials" not in metadata:
        metadata["materials"] = materials
    if preferred_axis and "preferred_axis" not in metadata:
        metadata["preferred_axis"] = preferred_axis
    if scale_mm_range and "scale_mm_range" not in metadata:
        metadata["scale_mm_range"] = [float(scale_mm_range[0]), float(scale_mm_range[1])]
    if notes and "notes" not in metadata:
        metadata["notes"] = notes

    items: List[Dict[str, Any]] = metadata.get("items", [])
    items.append(item)
    metadata["items"] = items

    metadata_path.write_text(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and store implant masks in the implant library.")
    parser.add_argument("--image", type=Path, required=True, help="Input CT image (.nii or .nii.gz).")
    parser.add_argument("--implant-library", type=Path, default=Path("data/implant_library"))
    parser.add_argument("--category", required=True, help="Folder name for the implant category (e.g. screws).")
    parser.add_argument("--implant-id", required=True, help="Base name for the implant mask file.")
    parser.add_argument("--implant-type", default=None, help="Human-readable implant type (defaults to category).")
    parser.add_argument("--hu-min", type=float, default=2500.0, help="Minimum HU threshold for metal.")
    parser.add_argument("--hu-max", type=float, default=None, help="Maximum HU threshold for metal.")
    parser.add_argument("--min-voxels", type=int, default=100, help="Minimum voxels to keep a component.")
    parser.add_argument("--keep-largest", action="store_true", help="Keep only the largest component.")
    parser.add_argument("--opening-iter", type=int, default=0, help="Binary opening iterations.")
    parser.add_argument("--closing-iter", type=int, default=0, help="Binary closing iterations.")
    parser.add_argument("--resample-iso", action="store_true", help="Resample to isotropic spacing.")
    parser.add_argument("--iso-mm", type=float, default=1.0, help="Target isotropic spacing in mm.")
    parser.add_argument("--align-pca", action="store_true", help="Align implant to canonical axis using PCA.")
    parser.add_argument("--crop", action="store_true", help="Crop to tight bounding box.")
    parser.add_argument("--margin", type=int, default=2, help="Margin (voxels) to add around crop.")
    parser.add_argument("--keep-affine", action="store_true", help="Preserve original affine (no local coords).")
    parser.add_argument("--typical-regions", default=None, help="Comma-separated typical regions.")
    parser.add_argument("--materials", default=None, help="Comma-separated implant materials.")
    parser.add_argument("--preferred-axis", default=None, help="Preferred axis (e.g. long_axis).")
    parser.add_argument("--scale-mm-range", default=None, help="Scale range in mm, e.g. 10,60.")
    parser.add_argument("--notes", default=None, help="Notes to store in metadata.")
    args = parser.parse_args()

    volume_hu, affine = load_nifti(args.image)
    mask = volume_hu >= args.hu_min
    if args.hu_max is not None:
        mask = np.logical_and(mask, volume_hu <= args.hu_max)

    if args.opening_iter > 0:
        mask = ndimage.binary_opening(mask, iterations=args.opening_iter)
    if args.closing_iter > 0:
        mask = ndimage.binary_closing(mask, iterations=args.closing_iter)

    mask = mask.astype(np.uint8)
    if args.keep_largest:
        mask = _largest_component(mask)
    else:
        mask = _filter_small_components(mask, args.min_voxels)

    spacing = tuple(float(v) for v in voxel_sizes(affine))
    original_spacing = spacing
    if args.resample_iso:
        mask, spacing = _resample_isotropic(mask, spacing, args.iso_mm)

    pca_axis = None
    if args.align_pca:
        mask = _pad_to_cube(mask, pad_value=0)
        mask, pca_axis = _pca_align_mask(mask)
        mask = _fill_holes(mask, closing_iter=2)

    bbox = _bounding_box(mask)
    crop_slices = _expand_slices(bbox, mask.shape, args.margin) if args.crop else None

    if crop_slices:
        mask = mask[crop_slices]
    mask = mask.astype(np.uint8)

    if args.keep_affine:
        out_affine = affine
    else:
        out_affine = _local_affine(spacing)

    category_dir = args.implant_library / args.category
    category_dir.mkdir(parents=True, exist_ok=True)
    implant_name = args.implant_id
    if not implant_name.endswith(".nii.gz"):
        implant_name = f"{implant_name}.nii.gz"
    mask_path = category_dir / implant_name
    save_nifti(mask_path, mask, out_affine)

    zyx_bbox = [
        [int(bbox[0].start), int(bbox[0].stop)],
        [int(bbox[1].start), int(bbox[1].stop)],
        [int(bbox[2].start), int(bbox[2].stop)],
    ]
    item = {
        "id": args.implant_id,
        "mask_file": implant_name,
        "source_image": str(args.image),
        "hu_threshold": [float(args.hu_min), float(args.hu_max) if args.hu_max is not None else None],
        "voxel_spacing_mm": [float(spacing[0]), float(spacing[1]), float(spacing[2])],
        "original_spacing_mm": [
            float(original_spacing[0]),
            float(original_spacing[1]),
            float(original_spacing[2]),
        ],
        "resampled_to_isotropic_mm": float(args.iso_mm) if args.resample_iso else None,
        "cropped_shape": [int(mask.shape[0]), int(mask.shape[1]), int(mask.shape[2])],
        "bbox_zyx": zyx_bbox,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "keep_affine": bool(args.keep_affine),
        "pca_aligned": bool(args.align_pca),
        "pca_principal_axis": (
            [float(pca_axis[0]), float(pca_axis[1]), float(pca_axis[2])] if pca_axis is not None else None
        ),
    }

    metadata_path = category_dir / "metadata.json"
    _update_metadata(
        metadata_path,
        args.implant_type or args.category,
        item,
        _parse_csv_list(args.typical_regions),
        _parse_csv_list(args.materials),
        args.preferred_axis,
        _parse_range(args.scale_mm_range),
        args.notes,
    )

    print(f"[build_implant_library] Saved mask to: {mask_path}")
    print(f"[build_implant_library] Updated metadata: {metadata_path}")


if __name__ == "__main__":
    main()
