from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class SimulationConfig:
    data_table: pd.DataFrame
    data_path: Path
    E0: int = 40
    metal_name: str = "Titanium"
    metal_density: float = 6.0
    metal_hu: float = 3000.0
    noise_scale: float = 12.0
    mu_water: float = 0.0
    mu_air: float = 0.0
    T1: float = 100.0
    T2: float = 1500.0
    energy_composition: np.ndarray = field(
        default_factory=lambda: np.linspace(1, 120, 120, dtype=np.uint8)
    )
    polynomial_order_for_correction: int = 3
    voxel_size_cm: float = 0.1
    angle_num: int = 180
    detector_pixels: int = 256
    detector_spacing: float = 0.1  # cm
    SOD_cm: float = 30.0
    SDD_cm: float = 60.0
    photon_scale: float = 1.0
    correction_coeff: Optional[np.ndarray] = None
    use_astra: bool = True
    fdk_filter: str = "ram-lak"
    placement_metadata: Optional[dict[str, Any]] = None

    def to_serializable(self) -> dict:
        payload = asdict(self)
        payload.pop("data_table", None)
        payload["data_path"] = str(self.data_path)
        if isinstance(payload.get("energy_composition"), np.ndarray):
            payload["energy_composition"] = payload["energy_composition"].tolist()
        if self.correction_coeff is not None:
            payload["correction_coeff"] = self.correction_coeff.tolist()
        if self.use_astra is not None:
            payload["use_astra"] = self.use_astra
        if self.fdk_filter:
            payload["fdk_filter"] = self.fdk_filter
        return payload


def load_xray_table(data_path: Path | str | None = None) -> pd.DataFrame:
    if data_path is None:
        data_path = Path(__file__).resolve().parent / "resources" / "xray_characteristic_data.csv"
    data_path = Path(data_path)
    df = pd.read_csv(data_path)
    if "Energy" not in df.columns:
        raise ValueError("xray_characteristic_data.csv must include an 'Energy' column")
    return df.set_index("Energy")


def set_config_for_artifact_simulation(voxel_size_cm: float, data_path: Path | str | None = None) -> SimulationConfig:
    data_table = load_xray_table(data_path)
    E0 = 40
    mu_water = float(data_table.loc[E0, "Water"])
    mu_air = 0.0
    cfg = SimulationConfig(
        data_table=data_table,
        data_path=Path(data_path) if data_path else Path(__file__).resolve().parent / "resources" / "xray_characteristic_data.csv",
        E0=E0,
        mu_water=mu_water,
        mu_air=mu_air,
        voxel_size_cm=float(voxel_size_cm),
    )
    return cfg
