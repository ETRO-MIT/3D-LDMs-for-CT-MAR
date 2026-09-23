from __future__ import annotations

import numpy as np
from scipy import ndimage

from .convert import hu2mu, mu2hu
from .simulation import metal_artifact_simulation_full3d


def create_spherical_mask(shape: tuple[int, int, int], radius: int = 10) -> np.ndarray:
    h, w, d = shape
    zc, yc, xc = (h - 1) / 2.0, (w - 1) / 2.0, (d - 1) / 2.0
    zz, yy, xx = np.ogrid[:h, :w, :d]
    mask = (zz - zc) ** 2 + (yy - yc) ** 2 + (xx - xc) ** 2 <= radius**2
    return mask.astype(np.uint8)


def calibrate_water_correction(config, phantom_size: int = 64, phantom_radius: int = 20):
    phantom = np.zeros((phantom_size, phantom_size, phantom_size), dtype=np.float32)
    zc = yc = xc = (phantom_size - 1) / 2.0
    zz, yy, xx = np.ogrid[:phantom_size, :phantom_size, :phantom_size]
    phantom[(zz - zc) ** 2 + (yy - yc) ** 2 + (xx - xc) ** 2 <= phantom_radius**2] = config.mu_water
    # Use identity correction to avoid unstable polyfit on noisy projections
    config.correction_coeff = np.array([1.0, 0.0], dtype=np.float32)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, num = ndimage.label(mask)
    if num <= 1:
        return mask.astype(bool)
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, num + 1))
    largest = int(np.argmax(sizes)) + 1
    return labeled == largest


def _body_mask_from_clean_hu(volume_hu: np.ndarray, threshold_hu: float = -900.0) -> np.ndarray:
    # Build a conservative patient support/body mask to suppress reconstruction support artifacts.
    body = volume_hu > threshold_hu
    if not np.any(body):
        return np.ones_like(volume_hu, dtype=bool)
    body = _largest_component(body)
    body = ndimage.binary_closing(body, iterations=2)
    body = ndimage.binary_fill_holes(body)
    body = ndimage.binary_dilation(body, iterations=2)
    return body.astype(bool)


def metal_artifact_simulation_volume(
    volume_hu: np.ndarray,
    config,
    metal_mask: np.ndarray | None = None,
    metal_hu: float = 3000.0,
    progress: bool = True,
    metal_radius: int = 10,
    photon_scale: float = 1.0,
) -> np.ndarray:
    if volume_hu.ndim != 3:
        raise ValueError("volume_hu must be 3D")
    h, w, d = volume_hu.shape
    clean_hu = volume_hu.astype(np.float32).copy()
    # Only clamp extreme air/outside values; preserve in-body contrast
    clean_hu[clean_hu < -1024] = -1000
    body_mask = _body_mask_from_clean_hu(clean_hu)

    if metal_mask is None:
        metal_mask = create_spherical_mask((h, w, d), radius=metal_radius)
    else:
        metal_mask = metal_mask.astype(np.uint8)
    metal_hu_volume = clean_hu.copy()
    metal_hu_volume[metal_mask > 0] = metal_hu

    clean_mu = hu2mu(clean_hu, config.mu_water, config.mu_air)
    metal_mu = hu2mu(metal_hu_volume, config.mu_water, config.mu_air)
    empty_metal_mask = np.zeros_like(metal_mask, dtype=np.uint8)

    clean_recon_mu = metal_artifact_simulation_full3d(
        clean_mu,
        empty_metal_mask,
        config,
        photon_scale=photon_scale,
        progress=progress,
    )
    metal_recon_mu = metal_artifact_simulation_full3d(
        metal_mu,
        metal_mask,
        config,
        photon_scale=photon_scale,
        progress=progress,
    )

    clean_recon_hu = mu2hu(clean_recon_mu, config.mu_water, config.mu_air)
    metal_recon_hu = mu2hu(metal_recon_mu, config.mu_water, config.mu_air)
    artifact_delta_hu = metal_recon_hu - clean_recon_hu
    print(
        f"[synthesis] Artifact delta HU: min={artifact_delta_hu.min():.1f}, "
        f"max={artifact_delta_hu.max():.1f}, mean={artifact_delta_hu.mean():.1f}"
    )
    if float(np.abs(artifact_delta_hu).max()) < 10.0:
        print(
            "[synthesis] WARNING: Artifact delta HU is near zero! Check detector geometry "
            "(e.g., --detector-spacing) to ensure metal projections are not truncated."
        )

    final_hu = clean_hu + artifact_delta_hu
    final_hu[~body_mask] = clean_hu[~body_mask]
    return final_hu
