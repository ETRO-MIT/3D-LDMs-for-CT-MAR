from __future__ import annotations

import numpy as np

from .convert import hu2mu
from .threshold import threshold_based_weighting
from .geometry_astra import cone_beam_backproject, cone_beam_project, require_astra_gpu


def _lookup(config, energy: int, material: str) -> float:
    return float(config.data_table.loc[int(energy), material])


def metal_artifact_simulation_full3d(
    volume_mu: np.ndarray,
    metal_mask: np.ndarray,
    config,
    photon_scale: float = 1.0,
    progress: bool = True,
) -> np.ndarray:
    """
    Full 3D cone-beam style metal artifact simulation.

    Projection and FDK reconstruction require CUDA-enabled ASTRA.
    """
    require_astra_gpu()
    if getattr(config, "use_astra", True) is False:
        raise ValueError("CPU simulation is not supported; use CUDA-enabled ASTRA.")
    angles_rad = np.linspace(0.0, 2.0 * np.pi, config.angle_num, endpoint=False, dtype=np.float32)
    energy_composition = config.energy_composition
    E0 = config.E0
    mu_air = config.mu_air
    metal_name = config.metal_name
    metal_density = config.metal_density
    T1 = config.T1
    T2 = config.T2
    correction_coeff = config.correction_coeff

    m0_water = _lookup(config, E0, "Water")
    m0_bone = _lookup(config, E0, "Bone")
    m0_metal = _lookup(config, E0, metal_name)
    mu_metal0 = m0_metal * metal_density

    T1_mu = hu2mu(T1, m0_water, mu_air)
    T2_mu = hu2mu(T2, m0_water, mu_air)

    x_water, x_bone = threshold_based_weighting(volume_mu, T1_mu, T2_mu)
    metal_mask = (metal_mask > 0).astype(np.float32)
    x_water = x_water * (1.0 - metal_mask)
    x_bone = x_bone * (1.0 - metal_mask)
    x_metal = metal_mask * mu_metal0

    # Forward projections

    d_water = cone_beam_project(
        x_water,
        config,
        angles_rad=angles_rad,
        progress=progress,
        label="water",
    )
    d_bone = cone_beam_project(
        x_bone,
        config,
        angles_rad=angles_rad,
        progress=progress,
        label="bone",
    )
    d_metal = cone_beam_project(
        x_metal,
        config,
        angles_rad=angles_rad,
        progress=progress,
        label="metal",
    )

    total_intensity = 0.0
    poly_y = np.zeros((*d_water.shape, len(energy_composition)), dtype=np.float32)

    for ii, energy in enumerate(energy_composition):
        m_water = _lookup(config, energy, "Water")
        m_bone = _lookup(config, energy, "Bone")
        m_metal = _lookup(config, energy, metal_name)
        intensity = _lookup(config, energy, "Intensity")

        d_water_tmp = d_water * (m_water / m0_water)
        d_bone_tmp = d_bone * (m_bone / m0_bone)
        d_metal_tmp = d_metal * (m_metal / m0_metal)
        DRR = d_water_tmp + d_bone_tmp + d_metal_tmp
        DRR = np.clip(DRR, 0.0, 50.0)  # avoid overflow in exp for long paths

        poly_y[:, :, :, ii] = intensity * np.exp(-DRR)
        total_intensity += intensity

    poly_y = np.sum(poly_y, axis=3)
    expected_counts = np.clip(poly_y * photon_scale, 0.0, 1e6)  # cap to keep Poisson stable
    noisy_y = np.random.poisson(expected_counts).astype(np.float32)
    ratio = np.clip(noisy_y / max(total_intensity * photon_scale, 1e-12), 1e-12, None)
    p = -np.log(ratio)

    # Simulate detector gain variations (breaks ring artifacts)
    '''gain_noise = 1.0 + 0.002 * np.random.randn(*p.shape)
    p = p * gain_noise'''

    if correction_coeff is not None:
        p = np.polyval(correction_coeff, p)

    # ASTRA FDK includes reconstruction filtering.
    sim = cone_beam_backproject(
        p,
        config,
        angles_rad=angles_rad,
        vol_shape=volume_mu.shape,
        progress=progress,
        label="backproj",
    )
    print("sim min/max:", sim.min(), sim.max(), "mean:", sim.mean())
    sim = np.clip(sim, a_min=0.0, a_max=None)
    return sim / config.voxel_size_cm
    #return sim 
