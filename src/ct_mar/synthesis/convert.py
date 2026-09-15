from __future__ import annotations

import numpy as np


def hu2mu(hu: np.ndarray, mu_water: float, mu_air: float) -> np.ndarray:
    """Convert Hounsfield Units to linear attenuation coefficients."""
    return np.asarray(hu, dtype=np.float32) / 1000.0 * (mu_water - mu_air) + mu_water


def mu2hu(mu: np.ndarray, mu_water: float, mu_air: float) -> np.ndarray:
    """Convert linear attenuation coefficients to Hounsfield Units."""
    return 1000.0 * (np.asarray(mu, dtype=np.float32) - mu_water) / (mu_water - mu_air)
