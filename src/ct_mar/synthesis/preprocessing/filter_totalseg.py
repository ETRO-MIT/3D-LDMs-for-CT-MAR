"""
filter_totalseg.py

Post-process TotalSegmentator outputs by removing empty/near-empty masks.
This is a preprocessing step that keeps only structures actually present.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def _strip_nii_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray]:
    nii = nib.load(path)
    data = nii.get_fdata().astype(np.float32)
    return data, nii.affine


def save_nifti(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), path)


def mask_is_present(mask: np.ndarray, min_voxels: int = 100) -> bool:
    return int(np.count_nonzero(mask)) >= int(min_voxels)


def load_totalseg_labels(labels_dir: Path) -> dict[str, np.ndarray]:
    labels = {}
    for path in sorted(labels_dir.glob("*.nii*")):
        data, _ = load_nifti(path)
        labels[_strip_nii_suffix(path)] = (data > 0).astype(np.uint8)
    return labels


def filter_totalseg_labels(
    totalseg_labels: dict[str, np.ndarray],
    min_voxels: int = 100,
) -> dict[str, np.ndarray]:
    return {
        name: mask
        for name, mask in totalseg_labels.items()
        if mask_is_present(mask, min_voxels)
    }


def save_totalseg_labels(
    labels: dict[str, np.ndarray],
    reference_affine: np.ndarray,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, mask in labels.items():
        save_nifti(output_dir / f"{name}.nii.gz", mask.astype(np.uint8), reference_affine)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter TotalSegmentator outputs by non-empty masks.")
    parser.add_argument(
        "--labels-dir",
        type=Path,
        required=True,
        help="Input directory containing TotalSegmentator label NIfTIs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write filtered labels. Defaults to <labels-dir>/filtered.",
    )
    parser.add_argument(
        "--min-voxels",
        type=int,
        default=100,
        help="Minimum non-zero voxel count to keep a label.",
    )
    args = parser.parse_args()

    labels_dir = args.labels_dir
    output_dir = args.output_dir or (labels_dir / "filtered")

    all_labels = load_totalseg_labels(labels_dir)
    if not all_labels:
        raise FileNotFoundError(f"No NIfTI labels found in {labels_dir}")

    filtered = filter_totalseg_labels(all_labels, min_voxels=args.min_voxels)
    first_path = next(labels_dir.glob("*.nii*"))
    _, affine = load_nifti(first_path)
    save_totalseg_labels(filtered, affine, output_dir)

    print(f"[filter_totalseg] Kept {len(filtered)}/{len(all_labels)} labels.")
    print(f"[filter_totalseg] Kept labels: {sorted(list(filtered.keys()))}")
    print(f"[filter_totalseg] Saved to: {output_dir}")


if __name__ == "__main__":
    main()
