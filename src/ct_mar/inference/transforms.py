from __future__ import annotations

import numpy as np
import torch

CT_HU_MIN: float = -1000.0
CT_HU_MAX_METAL: float = 4000.0
DEFAULT_TARGET_SIZE: tuple[int, int, int] = (448, 448, 256)


def normalize_ct_hu(volume: np.ndarray | torch.Tensor) -> np.ndarray:
    """
    Clamps HU values to [-1000, 4000] and scales linearly to [-1.0, 1.0].
    Handles NaNs and Infs.
    """
    if isinstance(volume, torch.Tensor):
        vol = volume.detach().cpu().numpy()
    else:
        vol = np.asarray(volume, dtype=np.float32)

    vol = np.nan_to_num(vol, nan=0.0, posinf=CT_HU_MAX_METAL, neginf=CT_HU_MIN)
    vol = np.clip(vol, CT_HU_MIN, CT_HU_MAX_METAL)
    norm = ((vol - CT_HU_MIN) / (CT_HU_MAX_METAL - CT_HU_MIN)) * 2.0 - 1.0
    return norm.astype(np.float32)


def denormalize_to_hu(norm_volume: np.ndarray | torch.Tensor) -> np.ndarray:
    """
    Converts normalized [-1.0, 1.0] volume back to CT Hounsfield Units [-1000, 4000].
    """
    if isinstance(norm_volume, torch.Tensor):
        vol = norm_volume.detach().cpu().numpy()
    else:
        vol = np.asarray(norm_volume, dtype=np.float32)

    vol = np.clip(vol, -1.0, 1.0)
    hu = ((vol + 1.0) * 0.5) * (CT_HU_MAX_METAL - CT_HU_MIN) + CT_HU_MIN
    return hu.astype(np.float32)


def pad_or_crop_3d(
    volume: np.ndarray,
    target_size: tuple[int, int, int] = DEFAULT_TARGET_SIZE,
    pad_value: float = -1.0,
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """
    Center crop or symmetrically pad a 3D volume [H, W, D] to target_size.
    Returns the processed volume and coordinate slices for optional unpadding/uncropping.
    """
    in_shape = volume.shape
    out = np.full(target_size, pad_value, dtype=volume.dtype)

    slices_in = []
    slices_out = []
    for cur_len, tgt_len in zip(in_shape, target_size):
        if cur_len >= tgt_len:
            # Crop
            start_in = (cur_len - tgt_len) // 2
            slices_in.append(slice(start_in, start_in + tgt_len))
            slices_out.append(slice(0, tgt_len))
        else:
            # Pad
            start_out = (tgt_len - cur_len) // 2
            slices_in.append(slice(0, cur_len))
            slices_out.append(slice(start_out, start_out + cur_len))

    out[tuple(slices_out)] = volume[tuple(slices_in)]
    meta = {
        "original_shape": in_shape,
        "slices_in": tuple((s.start, s.stop) for s in slices_in),
        "slices_out": tuple((s.start, s.stop) for s in slices_out),
    }
    return out, meta


def restore_to_original_shape(
    processed_volume: np.ndarray,
    original_shape: tuple[int, ...],
    meta: dict[str, tuple[int, int]],
    fill_from: np.ndarray | None = None,
) -> np.ndarray:
    """
    Places the processed (448, 448, 256) volume back into the original input volume dimensions.
    Unmodified regions are filled from fill_from (e.g. the original artifacted volume) if provided.
    """
    if fill_from is not None:
        restored = np.copy(fill_from)
    else:
        restored = np.full(original_shape, CT_HU_MIN, dtype=processed_volume.dtype)

    slices_in = tuple(slice(start, stop) for start, stop in meta["slices_in"])
    slices_out = tuple(slice(start, stop) for start, stop in meta["slices_out"])

    restored[slices_in] = processed_volume[slices_out]
    return restored

