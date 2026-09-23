from __future__ import annotations

from enum import Enum
import itertools
from pathlib import Path
from typing import Dict, Iterable, Tuple

import json
import numpy as np
import nibabel as nib
from scipy import ndimage

class Region(str, Enum):
    UNKNOWN = "unknown"
    SPINE = "spine"
    HIP = "hip"
    KNEE = "knee"
    SHOULDER = "shoulder"


class Metal(str, Enum):
    TITANIUM = "Titanium"
    IRON = "Iron"


METAL_POLICY: Dict[Region, Tuple[Tuple[Metal, float], ...]] = {
    Region.SPINE: ((Metal.TITANIUM, 0.7), (Metal.IRON, 0.3)),
    Region.HIP: ((Metal.TITANIUM, 0.6), (Metal.IRON, 0.4)),
    Region.KNEE: ((Metal.TITANIUM, 0.8), (Metal.IRON, 0.2)),
    Region.SHOULDER: ((Metal.TITANIUM, 0.7), (Metal.IRON, 0.3)),
    Region.UNKNOWN: ((Metal.TITANIUM, 0.5), (Metal.IRON, 0.5)),
}

METAL_HU: Dict[Metal, float] = {
    Metal.TITANIUM: 3000.0,
    Metal.IRON: 4000.0,
}

REGION_TO_IMPLANT_CATEGORY: Dict[Region, str] = {
    Region.SPINE: "spine_screws",
    #Region.SPINE: "spine_implants",
    #Region.SPINE: "plate_implants",
    #Region.HIP: "pelvic_screws",
    Region.HIP: "hip_implants",
}

_LABEL_KEYWORDS: Dict[Region, Tuple[str, ...]] = {
    Region.SPINE: ("vertebra", "spinal_cord", "spine"),
    Region.HIP: ("pelvis", "ilium", "ischium", "pubis", "acetabul", "hip", "femur_head", "femur"),
    Region.KNEE: ("knee", "patella", "tibia", "fibula"),
    Region.SHOULDER: ("scapula", "clavicula", "clavicle", "humerus"),
}


def _region_from_label_name(name: str) -> Region | None:
    label = name.lower()
    for region, keywords in _LABEL_KEYWORDS.items():
        if any(keyword in label for keyword in keywords):
            return region
    return None


def map_labels_to_regions(totalseg_labels: Dict[str, np.ndarray]) -> Dict[Region, np.ndarray]:
    region_masks: Dict[Region, np.ndarray] = {}
    for name, mask in totalseg_labels.items():
        region = _region_from_label_name(name)
        if region is None:
            continue
        if region not in region_masks:
            region_masks[region] = (mask > 0).astype(np.uint8)
        else:
            region_masks[region] = np.logical_or(region_masks[region], mask > 0).astype(np.uint8)
    return region_masks


def detect_region(
    volume_hu: np.ndarray,
    affine: np.ndarray | None = None,
    totalseg_labels: Dict[str, np.ndarray] | None = None,
    region_masks: Dict[Region, np.ndarray] | None = None,
    label_map: np.ndarray | None = None,
    label_names: Dict[int, str] | None = None,
) -> Region:
    """
    Region selector based on TotalSegmentator outputs.

    Accepts either:
    - totalseg_labels: dict of {label_name: binary_mask}
    - label_map + label_names: integer label map with id->name mapping
    """
    _ = volume_hu, affine
    region_scores = {region: 0 for region in Region}

    if totalseg_labels:
        for name, mask in totalseg_labels.items():
            region = _region_from_label_name(name)
            if region is None:
                continue
            region_scores[region] += int(np.count_nonzero(mask))

    if region_masks:
        for region, mask in region_masks.items():
            region_scores[region] += int(np.count_nonzero(mask))

    if label_map is not None and label_names:
        for label_id, name in label_names.items():
            region = _region_from_label_name(name)
            if region is None:
                continue
            region_scores[region] += int(np.count_nonzero(label_map == label_id))

    best_region = max(
        (region for region in Region if region is not Region.UNKNOWN),
        key=lambda region: region_scores[region],
    )
    if region_scores[best_region] == 0:
        return Region.UNKNOWN
    return best_region


def _choose_weighted(options: Iterable[Tuple[Metal, float]], rng: np.random.Generator) -> Metal:
    metals, weights = zip(*options)
    probs = np.array(weights, dtype=np.float64)
    probs = probs / probs.sum()
    idx = int(rng.choice(len(metals), p=probs))
    return metals[idx]


def choose_metal_for_region(region: Region, rng: np.random.Generator) -> Metal:
    return _choose_weighted(METAL_POLICY.get(region, METAL_POLICY[Region.UNKNOWN]), rng)


def choose_implant_mask(
    volume_hu: np.ndarray,
    region: Region,
    metal_radius: int,
    rng: np.random.Generator,
    primitive_shape: str = "sphere",
    primitive_length: int | None = None,
) -> np.ndarray:
    """
    Primitive placement: sphere or cylinder with optional region-aware jitter.
    """
    h, w, d = volume_hu.shape
    center = np.array([(h - 1) / 2.0, (w - 1) / 2.0, (d - 1) / 2.0], dtype=np.float32)
    jitter = rng.uniform(-0.1, 0.1, size=3) * np.array([h, w, d], dtype=np.float32)
    if region == Region.SPINE:
        jitter[1] -= 0.1 * w
    elif region == Region.HIP:
        jitter[0] += 0.1 * h
    elif region == Region.KNEE:
        jitter[0] += 0.2 * h
    elif region == Region.SHOULDER:
        jitter[0] -= 0.2 * h
    center = center + jitter

    zz, yy, xx = np.ogrid[:h, :w, :d]
    if primitive_shape == "cylinder":
        length = int(primitive_length) if primitive_length is not None else int(metal_radius * 2)
        half = max(1, length // 2)
        radial = (yy - center[1]) ** 2 + (xx - center[2]) ** 2 <= metal_radius**2
        axial = np.abs(zz - center[0]) <= half
        mask = np.logical_and(radial, axial)
    else:
        mask = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2 <= metal_radius**2
    return mask.astype(np.uint8)


def _load_implant_metadata(category_dir: Path) -> dict:
    metadata_path = category_dir / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())
    items = []
    for path in sorted(category_dir.glob("*.nii*")):
        items.append({"id": path.stem, "mask_file": path.name})
    return {"items": items}


def _select_implant_item(
    items: list[dict],
    implant_id: str | None,
    implant_random: bool,
    rng: np.random.Generator,
) -> dict | None:
    if not items:
        return None
    if implant_id:
        for item in items:
            if item.get("id") == implant_id:
                return item
        return None
    if implant_random or not implant_id:
        idx = int(rng.integers(0, len(items)))
        return items[idx]
    return None


def _load_mask(path: Path) -> np.ndarray:
    nii = nib.load(path)
    return (nii.get_fdata() > 0).astype(np.uint8)


def _best_fit_implant(
    mask: np.ndarray, target_shape: tuple[int, int, int]
) -> tuple[np.ndarray, float, tuple[int, int, int]]:
    best_scale = -1.0
    best_perm = (0, 1, 2)
    best_mask = mask
    for perm in itertools.permutations((0, 1, 2), 3):
        permuted = np.transpose(mask, perm)
        scales = []
        for m, t in zip(permuted.shape, target_shape):
            if t <= 2:
                scales.append(1.0)
            else:
                scales.append((t - 2) / max(m, 1))
        scale = float(min(scales))
        if scale > best_scale:
            best_scale = scale
            best_perm = perm
            best_mask = permuted
    return best_mask, best_scale, best_perm


def _scale_only_implant(
    mask: np.ndarray, target_shape: tuple[int, int, int]
) -> tuple[np.ndarray, float]:
    scales = []
    for m, t in zip(mask.shape, target_shape):
        if t <= 2:
            scales.append(1.0)
        else:
            scales.append((t - 2) / max(m, 1))
    scale = float(min(scales))
    return mask, scale


def _resize_mask_with_scale(mask: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        return mask
    if np.isclose(scale, 1.0):
        return mask
    resized = ndimage.zoom(mask.astype(np.float32), zoom=(scale, scale, scale), order=0)
    return (resized > 0.5).astype(np.uint8)


def _choose_center(
    volume_shape: tuple[int, int, int],
    region: Region,
    rng: np.random.Generator,
    region_masks: Dict[Region, np.ndarray] | None = None,
    implant_mask: np.ndarray | None = None,
) -> np.ndarray:
    if region_masks and region in region_masks:
        region_mask = region_masks[region]
        if region_mask.shape != volume_shape:
            coords = np.argwhere(region_mask > 0)
            if coords.shape[0] > 0:
                idx = int(rng.integers(0, coords.shape[0]))
                center = coords[idx].astype(np.float32)
                scale = (np.array(volume_shape, dtype=np.float32) - 1.0) / (
                    np.array(region_mask.shape, dtype=np.float32) - 1.0
                )
                return center * scale
        coords = np.argwhere(region_mask > 0)
        if coords.shape[0] > 0:
            if implant_mask is None:
                idx = int(rng.integers(0, coords.shape[0]))
                return coords[idx].astype(np.float32)
            # Try to pick a center where most of the implant overlaps the region mask.
            imp_shape = np.array(implant_mask.shape, dtype=int)
            max_candidates = min(200, coords.shape[0])
            best_center = None
            best_overlap = -1.0
            for _ in range(max_candidates):
                idx = int(rng.integers(0, coords.shape[0]))
                center = coords[idx].astype(int)
                start = center - imp_shape // 2
                end = start + imp_shape

                vol_slices = []
                imp_slices = []
                valid = True
                for axis in range(3):
                    v_start = max(start[axis], 0)
                    v_end = min(end[axis], volume_shape[axis])
                    i_start = max(0, -start[axis])
                    i_end = i_start + (v_end - v_start)
                    if v_end <= v_start:
                        valid = False
                        break
                    vol_slices.append(slice(v_start, v_end))
                    imp_slices.append(slice(i_start, i_end))
                if not valid:
                    continue

                imp_part = implant_mask[imp_slices[0], imp_slices[1], imp_slices[2]] > 0
                if imp_part.size == 0:
                    continue
                reg_part = region_mask[vol_slices[0], vol_slices[1], vol_slices[2]] > 0
                overlap = float(np.count_nonzero(imp_part & reg_part)) / float(np.count_nonzero(imp_part))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_center = center.astype(np.float32)
                    if best_overlap >= 0.9:
                        break
            if best_center is not None:
                return best_center
    h, w, d = volume_shape
    center = np.array([(h - 1) / 2.0, (w - 1) / 2.0, (d - 1) / 2.0], dtype=np.float32)
    jitter = rng.uniform(-0.1, 0.1, size=3) * np.array([h, w, d], dtype=np.float32)
    if region == Region.SPINE:
        jitter[1] -= 0.1 * w
    elif region == Region.HIP:
        jitter[0] += 0.1 * h
    elif region == Region.KNEE:
        jitter[0] += 0.2 * h
    elif region == Region.SHOULDER:
        jitter[0] -= 0.2 * h
    return center + jitter


def _paste_mask(
    volume_shape: tuple[int, int, int],
    implant_mask: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    full = np.zeros(volume_shape, dtype=np.uint8)
    imp_shape = np.array(implant_mask.shape, dtype=int)
    center = np.round(center).astype(int)
    start = center - imp_shape // 2
    end = start + imp_shape

    for axis in range(3):
        if start[axis] < 0:
            shift = -start[axis]
            start[axis] += shift
            end[axis] += shift
        if end[axis] > volume_shape[axis]:
            shift = end[axis] - volume_shape[axis]
            start[axis] -= shift
            end[axis] -= shift
        start[axis] = max(start[axis], 0)
        end[axis] = min(end[axis], volume_shape[axis])

    vol_slices = []
    imp_slices = []
    for axis in range(3):
        v_start = max(start[axis], 0)
        v_end = min(end[axis], volume_shape[axis])
        i_start = max(0, -start[axis])
        i_end = i_start + (v_end - v_start)
        if v_end <= v_start:
            return full
        vol_slices.append(slice(v_start, v_end))
        imp_slices.append(slice(i_start, i_end))

    full[vol_slices[0], vol_slices[1], vol_slices[2]] = implant_mask[
        imp_slices[0], imp_slices[1], imp_slices[2]
    ]
    return full


def load_implant_from_library(
    volume_shape: tuple[int, int, int],
    region: Region,
    rng: np.random.Generator,
    implant_library: Path,
    implant_category: str | None = None,
    implant_id: str | None = None,
    implant_random: bool = False,
    region_masks: Dict[Region, np.ndarray] | None = None,
) -> tuple[np.ndarray | None, dict | None]:
    category = implant_category or REGION_TO_IMPLANT_CATEGORY.get(region)
    if category is None and implant_library.exists():
        available_cats = sorted([d.name for d in implant_library.iterdir() if d.is_dir() and (d / "metadata.json").exists()])
        if available_cats:
            category = str(rng.choice(available_cats))
            print(f"[anatomy] Region is unknown; randomly selected library category: '{category}'")
    if category is None:
        return None, None
    category_dir = implant_library / category
    if not category_dir.exists():
        print(f"[anatomy] WARNING: Category folder not found: {category_dir}")
        return None, None
    metadata = _load_implant_metadata(category_dir)
    items = metadata.get("items", [])
    if not items:
        print(f"[anatomy] WARNING: No implants found in {category_dir}")
    item = _select_implant_item(items, implant_id, implant_random, rng)
    if item is None:
        return None, {
            "implant_category": category,
            "implant_type": metadata.get("implant_type"),
            "library_materials": metadata.get("materials"),
            "library_typical_regions": metadata.get("typical_regions"),
        }
    mask_file = item.get("mask_file")
    if not mask_file:
        return None, {
            "implant_category": category,
            "implant_type": metadata.get("implant_type"),
            "library_materials": metadata.get("materials"),
            "library_typical_regions": metadata.get("typical_regions"),
        }
    implant_mask = _load_mask(category_dir / mask_file)
    if region in {Region.HIP, Region.SPINE}:
        implant_mask, scale = _scale_only_implant(implant_mask, volume_shape)
        print(f"[anatomy] Preserving library orientation for {region.value} implant placement.")
    else:
        implant_mask, scale, _ = _best_fit_implant(implant_mask, volume_shape)
    scale = min(scale, 1.0)
    implant_mask = _resize_mask_with_scale(implant_mask, scale)
    center = None
    if region in {Region.HIP, Region.SPINE}:
        center = _choose_center(volume_shape, region, rng, region_masks=region_masks, implant_mask=implant_mask)
    if center is None:
        center = _choose_center(volume_shape, region, rng, region_masks=region_masks, implant_mask=implant_mask)
    full_mask = _paste_mask(volume_shape, implant_mask, center)
    selection_metadata = {
        "implant_category": category,
        "implant_type": metadata.get("implant_type"),
        "implant_id_selected": item.get("id") or Path(mask_file).stem,
        "implant_mask_file": mask_file,
        "implant_source_mesh": item.get("source_mesh"),
        "library_materials": metadata.get("materials"),
        "library_typical_regions": metadata.get("typical_regions"),
        "pca_aligned": item.get("pca_aligned"),
        "pca_principal_axis": item.get("pca_principal_axis"),
    }
    return full_mask, selection_metadata


def _best_label_for_region(totalseg_labels: Dict[str, np.ndarray], region: Region) -> tuple[str | None, int]:
    best_label = None
    best_count = 0
    for name, mask in totalseg_labels.items():
        if _region_from_label_name(name) != region:
            continue
        count = int(np.count_nonzero(mask))
        if count > best_count:
            best_label = name
            best_count = count
    return best_label, best_count


def anatomy_aware_metal_mask(
    volume_hu: np.ndarray,
    affine: np.ndarray | None = None,
    metal_radius: int = 10,
    rng: np.random.Generator | None = None,
    totalseg_labels: Dict[str, np.ndarray] | None = None,
    region_masks: Dict[Region, np.ndarray] | None = None,
    label_map: np.ndarray | None = None,
    label_names: Dict[int, str] | None = None,
    implant_library: Path | None = None,
    implant_category: str | None = None,
    implant_id: str | None = None,
    implant_random: bool = False,
    implant_source: str = "library",
    primitive_shape: str = "sphere",
    primitive_length: int | None = None,
) -> tuple[np.ndarray, float, str, dict]:
    rng = rng or np.random.default_rng()
    region = detect_region(
        volume_hu,
        affine,
        totalseg_labels=totalseg_labels,
        region_masks=region_masks,
        label_map=label_map,
        label_names=label_names,
    )
    metal = choose_metal_for_region(region, rng)
    metal_mask = None
    implant_metadata = None
    if implant_source == "library" and implant_library is not None:
        metal_mask, implant_metadata = load_implant_from_library(
            volume_hu.shape,
            region,
            rng,
            implant_library,
            implant_category=implant_category,
            implant_id=implant_id,
            implant_random=implant_random,
            region_masks=region_masks,
        )
    used_primitive_fallback = False
    if metal_mask is None and implant_source == "library":
        raise ValueError(
            "Could not select a library implant. Check the category, implant ID, and metadata. "
            "Provide anatomy masks for automatic category selection, or set --implant-category. "
            "Use --implant-source primitive to explicitly request a primitive shape."
        )
    if metal_mask is None:
        metal_mask = choose_implant_mask(
            volume_hu,
            region,
            metal_radius,
            rng,
            primitive_shape=primitive_shape,
            primitive_length=primitive_length,
        )
        used_primitive_fallback = True
    metal_hu = METAL_HU.get(metal, METAL_HU[Metal.TITANIUM])
    metal_name = metal.value if isinstance(metal, Metal) else str(metal)
    label_name = None
    label_count = 0
    if totalseg_labels:
        label_name, label_count = _best_label_for_region(totalseg_labels, region)
        if label_name:
            print(
                f"[anatomy_aware_metal_mask] Selected label: {label_name} "
                f"(region={region.value}, voxels={label_count})"
            )
        else:
            print(f"[anatomy_aware_metal_mask] Selected region: {region.value} (no matching label)")
    else:
        print(f"[anatomy_aware_metal_mask] Selected region: {region.value}")
    if implant_source == "library":
        if implant_library is not None:
            print(f"[anatomy_aware_metal_mask] Implant library: {implant_library}")
        if implant_metadata and implant_metadata.get("implant_category") is not None:
            print(f"[anatomy_aware_metal_mask] Implant category: {implant_metadata['implant_category']}")
        if implant_metadata and implant_metadata.get("implant_id_selected"):
            print(f"[anatomy_aware_metal_mask] Selected implant: {implant_metadata['implant_id_selected']}")
        elif implant_id:
            print(f"[anatomy_aware_metal_mask] Requested implant not found: {implant_id}")
    else:
        if primitive_shape == "cylinder":
            length = primitive_length if primitive_length is not None else int(metal_radius * 2)
            print(
                f"[anatomy_aware_metal_mask] Selected primitive: cylinder (radius={metal_radius}, length={length})"
            )
        else:
            print(f"[anatomy_aware_metal_mask] Selected primitive: sphere (radius={metal_radius})")
    print(f"[anatomy_aware_metal_mask] Selected metal: {metal_name} (type={type(metal).__name__})")
    placement_metadata = {
        "region": region.value,
        "main_label": label_name,
        "main_label_voxels": label_count,
        "implant_source": "primitive" if used_primitive_fallback else implant_source,
        "selection_mode": (
            "requested"
            if implant_source == "library" and implant_id and not used_primitive_fallback
            else "random"
            if implant_source == "library" and not used_primitive_fallback
            else "primitive_fallback"
            if used_primitive_fallback and implant_source == "library"
            else "primitive"
        ),
        "implant_id_requested": implant_id,
        "metal_name": metal_name,
        "metal_hu": metal_hu,
    }
    if implant_library is not None:
        placement_metadata["implant_library"] = str(implant_library)
    if implant_metadata:
        placement_metadata.update(implant_metadata)
    if used_primitive_fallback or implant_source == "primitive":
        placement_metadata["primitive_shape"] = primitive_shape
        placement_metadata["metal_radius"] = metal_radius
        if primitive_shape == "cylinder":
            placement_metadata["primitive_length"] = (
                primitive_length if primitive_length is not None else int(metal_radius * 2)
            )
    return metal_mask, metal_hu, metal_name, placement_metadata
