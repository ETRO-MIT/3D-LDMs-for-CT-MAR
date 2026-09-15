from __future__ import annotations

import numpy as np

try:
    import astra
except ImportError:  # pragma: no cover
    astra = None


def astra_available() -> bool:
    """Whether ASTRA is installed with CUDA support (not a device allocation test)."""
    return astra is not None and astra.use_cuda()


def require_astra_gpu() -> None:
    if not astra_available():
        raise RuntimeError(
            "3D artifact generation requires CUDA-enabled astra-toolbox and an NVIDIA GPU. "
            "Install ASTRA on the GPU machine; no CPU simulation fallback is provided. "
            "CT preprocessing and implant-library preparation can run separately."
        )


def _vol_geom(volume: np.ndarray):
    nx, ny, nz = volume.shape
    return astra.create_vol_geom(nx, ny, nz)

def _proj_geom(config, angles: np.ndarray | None = None):
    if angles is None:
        angles = np.linspace(0.0, np.pi, config.angle_num, endpoint=False, dtype=np.float32)
    scale = 1.0 / config.voxel_size_cm
    det_spacing = config.detector_spacing * scale
    source_origin = config.SOD_cm * scale
    origin_det = (config.SDD_cm - config.SOD_cm) * scale
    return astra.create_proj_geom(
        "cone",
        det_spacing,
        det_spacing,
        config.detector_pixels,
        config.detector_pixels,
        angles,
        source_origin,
        origin_det,
    )


def cone_beam_project(volume: np.ndarray, config, angles_rad: np.ndarray | None = None, **kwargs) -> np.ndarray:
    require_astra_gpu()
    vol_geom = _vol_geom(volume)
    proj_geom = _proj_geom(config, angles_rad)
    print("Volume shape before ASTRA:", volume.shape)
    #sino_id, sino = astra.create_sino3d_gpu(volume, proj_geom, vol_geom)
    #Astra expects volume ordering (Z,Y,X)
    #Input volume is (X,Y,Z), so we transpose
    # ASTRA integrates voxel values assuming unit voxel length; therefore attenuation coefficients expressed in 1/cm must be converted to per-voxel units
    # Scale mu (1/cm) to per-voxel units to keep line integrals consistent.
    volume_scaled = volume * config.voxel_size_cm
    sino_id, sino = astra.create_sino3d_gpu(
        volume_scaled.transpose(2, 1, 0),  # (X,Y,Z) -> (Z,Y,X)
        proj_geom,
        vol_geom
        )
    astra.data3d.delete(sino_id)
    return sino


def cone_beam_backproject(
    sino: np.ndarray,
    config,
    vol_shape: tuple[int, int, int],
    angles_rad: np.ndarray | None = None,
    **kwargs,
) -> np.ndarray:
    require_astra_gpu()

    vol_geom = _vol_geom(np.zeros(vol_shape, dtype=np.float32))
    proj_geom = _proj_geom(config, angles_rad)

    proj_id = astra.data3d.create("-sino", proj_geom, sino)
    rec_id = astra.data3d.create("-vol", vol_geom)

    cfg = astra.astra_dict("FDK_CUDA")
    cfg["ProjectionDataId"] = proj_id
    cfg["ReconstructionDataId"] = rec_id
    cfg["FilterType"] = getattr(config, "fdk_filter", "ram-lak")

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    rec = astra.data3d.get(rec_id)

    # IMPORTANT:
    # ASTRA returns volumes in (Z,Y,X) order for 3D data.
    # Convert back to the pipeline convention: (X,Y,Z).
    rec = rec.transpose(2, 1, 0)

    astra.algorithm.delete(alg_id)
    astra.data3d.delete(proj_id)
    astra.data3d.delete(rec_id)

    return rec
