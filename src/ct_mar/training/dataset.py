from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    RandFlipd,
    RandRotated,
    RandSpatialCropd,
    ScaleIntensityRanged,
    SpatialPadd,
    ThresholdIntensityd,
    CenterSpatialCropd,
)
from monai.data import DataLoader, Dataset

from ct_mar.inference.transforms import CT_HU_MAX_METAL, CT_HU_MIN
from ct_mar.inference.models.metadata import (
    normalize_metal_name,
    normalize_region,
    normalize_side,
    parse_metadata_json,
)


def get_mar_datalist(ids_path: str | Path) -> list[dict[str, str]]:
    """
    Build paired MAR samples from a CSV manifest.
    Expected columns:
      - synthetic_image_path: artifacted CT
      - implant_only_image_path: paired non-artifacted target with implant
      - Optional: metadata_path, region, side, metal_name, case_id
    """
    df = pd.read_csv(ids_path)
    data_list = []
    for _, row in df.iterrows():
        cond_path = str(row.get("synthetic_image_path") or row.get("condition") or "").strip()
        target_path = str(row.get("implant_only_image_path") or row.get("target") or "").strip()
        if not cond_path or not target_path:
            continue

        meta_path = str(row.get("metadata_path") or "").strip()
        if not meta_path and cond_path.endswith(".nii.gz"):
            meta_path = cond_path[:-7] + ".json"
        elif not meta_path and cond_path.endswith(".nii"):
            meta_path = cond_path[:-4] + ".json"

        extracted = parse_metadata_json(meta_path) if meta_path else {}
        region = normalize_region(str(row.get("region") or extracted.get("region", "unknown")))
        side = normalize_side(str(row.get("side") or extracted.get("side", "unknown")))
        metal_name = normalize_metal_name(str(row.get("metal_name") or extracted.get("metal_name", "unknown")))

        data_list.append(
            {
                "synthetic_image": cond_path,
                "implant_only_image": target_path,
                "metadata_path": meta_path,
                "region": region,
                "side": side,
                "metal_name": metal_name,
                "case_id": str(row.get("case_id", Path(cond_path).stem)),
            }
        )
    return data_list


def build_mar_transforms(image_roi: tuple[int, int, int] = (448, 448, 256), train: bool = True):
    """Paired transforms ensuring identical spatial crops and HU windowing."""
    keys = ["synthetic_image", "implant_only_image"]
    transforms = [
        LoadImaged(keys=keys, image_only=True),
        EnsureChannelFirstd(keys=keys),
        Lambdad(
            keys=keys,
            func=lambda x: np.nan_to_num(x, nan=0.0, posinf=CT_HU_MAX_METAL, neginf=CT_HU_MIN),
        ),
        ThresholdIntensityd(keys=keys, threshold=CT_HU_MAX_METAL, above=False, cval=CT_HU_MAX_METAL),
        ThresholdIntensityd(keys=keys, threshold=CT_HU_MIN, above=True, cval=CT_HU_MIN),
        ScaleIntensityRanged(
            keys=keys,
            a_min=CT_HU_MIN,
            a_max=CT_HU_MAX_METAL,
            b_min=-1.0,
            b_max=1.0,
            clip=True,
        ),
        SpatialPadd(keys=keys, spatial_size=image_roi, mode="constant", constant_values=-1.0),
    ]
    if train:
        transforms.extend(
            [
                RandRotated(keys=keys, range_x=0.08, range_y=0.08, range_z=0.08, prob=0.2),
                RandFlipd(keys=keys, spatial_axis=1, prob=0.5),
                RandSpatialCropd(keys=keys, roi_size=image_roi, random_size=False),
            ]
        )
    else:
        transforms.append(CenterSpatialCropd(keys=keys, roi_size=image_roi))

    transforms.extend(
        [
            EnsureTyped(keys=keys, data_type="tensor", track_meta=False),
            Lambdad(keys=keys, func=lambda x: x.as_tensor() if hasattr(x, "as_tensor") else torch.as_tensor(x)),
        ]
    )
    return Compose(transforms)


def get_mar_dataloader(
    ids_path: str | Path,
    batch_size: int = 2,
    image_roi: tuple[int, int, int] = (448, 448, 256),
    train: bool = True,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    data_list = get_mar_datalist(ids_path)
    transform = build_mar_transforms(image_roi=image_roi, train=train)
    dataset = Dataset(data=data_list, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

