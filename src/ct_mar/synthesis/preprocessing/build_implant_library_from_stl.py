"""
build_implant_library_from_stl.py

Voxelize STL implant meshes and store them as binary masks in the implant library.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import nibabel as nib
import numpy as np
import trimesh
from scipy import ndimage


def save_nifti(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), path)


def _local_affine(spacing: Tuple[float, float, float]) -> np.ndarray:
    affine = np.eye(4, dtype=np.float32)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]
    return affine


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


def _pca_align_mesh(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, np.ndarray]:
    verts = mesh.vertices
    if verts.shape[0] < 10:
        return mesh, np.array([0.0, 0.0, 1.0], dtype=np.float64)
    center = verts.mean(axis=0)
    centered = verts - center
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_axis = eigvecs[:, np.argmax(eigvals)]
    R = _rotation_to_z(principal_axis)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = center - R @ center
    aligned = mesh.copy()
    aligned.apply_transform(T)
    return aligned, principal_axis


def _bounding_box(mask: np.ndarray) -> tuple[slice, slice, slice]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        raise ValueError("No voxels found after voxelization.")
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


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "implant"


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


def _ensure_unique(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = "".join(path.suffixes)
    parent = path.parent
    idx = 1
    while True:
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    else:
        keep = mesh.nondegenerate_faces()
        if keep is not None:
            mesh.update_faces(keep)
    if not mesh.is_watertight:
        # Some trimesh versions return a bool for fill_holes()
        _ = mesh.fill_holes()
    return mesh


def _voxelize_mesh(mesh: trimesh.Trimesh, voxel_mm: float, fill: bool) -> np.ndarray:
    voxelized = mesh.voxelized(pitch=voxel_mm)
    if fill:
        voxelized = voxelized.fill()
    matrix = voxelized.matrix.astype(np.uint8)
    return np.ascontiguousarray(matrix)


def _process_mesh(
    mesh_path: Path,
    output_dir: Path,
    implant_id: str | None,
    voxel_mm: float,
    align_pca: bool,
    fill: bool,
    closing_iter: int,
    crop: bool,
    margin: int,
    implant_type: str,
    typical_regions: list[str] | None,
    materials: list[str] | None,
    preferred_axis: str | None,
    scale_mm_range: tuple[float, float] | None,
    notes: str | None,
) -> None:
    mesh = _load_mesh(mesh_path)
    pca_axis = None
    if align_pca:
        mesh, pca_axis = _pca_align_mesh(mesh)

    mask = _voxelize_mesh(mesh, voxel_mm, fill=fill)
    if closing_iter > 0:
        mask = ndimage.binary_closing(mask, iterations=closing_iter).astype(np.uint8)
    if fill:
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

    bbox = _bounding_box(mask)
    crop_slices = _expand_slices(bbox, mask.shape, margin) if crop else None
    if crop_slices:
        mask = mask[crop_slices]

    out_affine = _local_affine((voxel_mm, voxel_mm, voxel_mm))
    base_id = implant_id or _slugify(mesh_path.stem)
    out_name = f"{base_id}.nii.gz"
    out_path = _ensure_unique(output_dir / out_name)
    save_nifti(out_path, mask.astype(np.uint8), out_affine)

    zyx_bbox = [
        [int(bbox[0].start), int(bbox[0].stop)],
        [int(bbox[1].start), int(bbox[1].stop)],
        [int(bbox[2].start), int(bbox[2].stop)],
    ]
    item = {
        "id": out_path.stem,
        "mask_file": out_path.name,
        "source_mesh": str(mesh_path),
        "voxel_spacing_mm": [float(voxel_mm), float(voxel_mm), float(voxel_mm)],
        "cropped_shape": [int(mask.shape[0]), int(mask.shape[1]), int(mask.shape[2])],
        "bbox_zyx": zyx_bbox,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pca_aligned": bool(align_pca),
        "pca_principal_axis": (
            [float(pca_axis[0]), float(pca_axis[1]), float(pca_axis[2])] if pca_axis is not None else None
        ),
    }

    metadata_path = output_dir / "metadata.json"
    _update_metadata(
        metadata_path,
        implant_type,
        item,
        typical_regions,
        materials,
        preferred_axis,
        scale_mm_range,
        notes,
    )

    print(f"[build_implant_library_from_stl] Saved mask to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Voxelize STL implants into the implant library.")
    parser.add_argument("--mesh", type=Path, default=None, help="Single STL mesh to process.")
    parser.add_argument("--mesh-dir", type=Path, default=None, help="Directory with STL meshes.")
    parser.add_argument("--implant-library", type=Path, default=Path("data/implant_library"))
    parser.add_argument("--category", default="hip_implants", help="Output implant category folder.")
    parser.add_argument("--implant-id", default=None, help="Override implant id (single mesh only).")
    parser.add_argument("--voxel-mm", type=float, default=1.0, help="Voxel size in mm for voxelization.")
    parser.add_argument("--fill", action="store_true", help="Fill interior after voxelization.")
    parser.add_argument("--closing-iter", type=int, default=0, help="Binary closing iterations.")
    parser.add_argument("--align-pca", action="store_true", help="Align mesh to canonical axis using PCA.")
    parser.add_argument("--crop", action="store_true", help="Crop to tight bounding box.")
    parser.add_argument("--margin", type=int, default=2, help="Margin (voxels) to add around crop.")
    parser.add_argument("--implant-type", default=None, help="Human-readable implant type (defaults to category).")
    parser.add_argument("--typical-regions", default=None, help="Comma-separated typical regions.")
    parser.add_argument("--materials", default=None, help="Comma-separated implant materials.")
    parser.add_argument("--preferred-axis", default=None, help="Preferred axis (e.g. long_axis).")
    parser.add_argument("--scale-mm-range", default=None, help="Scale range in mm, e.g. 10,60.")
    parser.add_argument("--notes", default=None, help="Notes to store in metadata.")
    args = parser.parse_args()

    if args.mesh is None and args.mesh_dir is None:
        raise SystemExit("Provide --mesh or --mesh-dir.")

    output_dir = args.implant_library / args.category
    output_dir.mkdir(parents=True, exist_ok=True)

    typical_regions = _parse_csv_list(args.typical_regions)
    materials = _parse_csv_list(args.materials)
    scale_mm_range = _parse_range(args.scale_mm_range)
    implant_type = args.implant_type or args.category

    mesh_paths: List[Path] = []
    if args.mesh is not None:
        mesh_paths.append(args.mesh)
    if args.mesh_dir is not None:
        mesh_paths.extend(sorted(args.mesh_dir.glob("*.stl")))

    if not mesh_paths:
        raise SystemExit("No STL meshes found.")

    for mesh_path in mesh_paths:
        override_id = args.implant_id if (args.mesh is not None and len(mesh_paths) == 1) else None
        _process_mesh(
            mesh_path=mesh_path,
            output_dir=output_dir,
            implant_id=override_id,
            voxel_mm=float(args.voxel_mm),
            align_pca=bool(args.align_pca),
            fill=bool(args.fill),
            closing_iter=int(args.closing_iter),
            crop=bool(args.crop),
            margin=int(args.margin),
            implant_type=implant_type,
            typical_regions=typical_regions,
            materials=materials,
            preferred_axis=args.preferred_axis,
            scale_mm_range=scale_mm_range,
            notes=args.notes,
        )


if __name__ == "__main__":
    main()
