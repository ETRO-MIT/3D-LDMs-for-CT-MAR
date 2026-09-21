from __future__ import annotations

from pathlib import Path
import numpy as np

try:
    import nibabel as nib
except ImportError:
    nib = None


def load_nifti(path: str | Path) -> tuple[np.ndarray, np.ndarray, any]:
    """
    Load a NIfTI volume.
    Returns (volume_data as float32 np.ndarray, affine 4x4 matrix, nibabel header).
    """
    if nib is None:
        raise ImportError("nibabel is required to load NIfTI images. Install via `pip install nibabel`.")
    nii = nib.load(str(path))
    data = np.asarray(nii.get_fdata(), dtype=np.float32)
    return data, nii.affine, nii.header


def save_nifti(
    path: str | Path,
    data: np.ndarray,
    affine: np.ndarray | None = None,
    header: any = None,
) -> Path:
    """
    Save 3D data array as NIfTI file with affine matrix and header.
    """
    if nib is None:
        raise ImportError("nibabel is required to save NIfTI images. Install via `pip install nibabel`.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if affine is None:
        affine = np.eye(4, dtype=np.float32)
    img = nib.Nifti1Image(data.astype(np.float32), affine, header=header)
    nib.save(img, str(target))
    return target


def export_slice_comparison_png(
    output_path: str | Path,
    original_hu: np.ndarray,
    restored_hu: np.ndarray,
    slice_idx: int | None = None,
    axis: int = 2,
    window_min: float = -200.0,
    window_max: float = 800.0,
) -> Path:
    """
    Export a 2D side-by-side PNG comparing artifacted input and restored CT at a given slice.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return Path(output_path)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if slice_idx is None:
        slice_idx = original_hu.shape[axis] // 2

    if axis == 0:
        orig_slice = original_hu[slice_idx, :, :]
        rest_slice = restored_hu[slice_idx, :, :]
    elif axis == 1:
        orig_slice = original_hu[:, slice_idx, :]
        rest_slice = restored_hu[:, slice_idx, :]
    else:
        orig_slice = original_hu[:, :, slice_idx]
        rest_slice = restored_hu[:, :, slice_idx]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(np.rot90(orig_slice), cmap="gray", vmin=window_min, vmax=window_max)
    axes[0].set_title("Input (Artifacted)")
    axes[0].axis("off")

    axes[1].imshow(np.rot90(rest_slice), cmap="gray", vmin=window_min, vmax=window_max)
    axes[1].set_title("Restored (3D LDM)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out

