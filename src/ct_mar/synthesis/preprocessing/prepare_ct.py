"""Optional CT preparation adapted from Preprocessing_Check_Nii_Metadata.ipynb.

Run before anatomy segmentation for unprepared data. Already-prepared HPC volumes
can go directly to generation; --check-only inspects them without writing files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_to_output
import numpy as np

SUFFIX = "_iso1mm_RAS_origin0_int16"


def nifti_stem(path: Path) -> str:
    return path.name.removesuffix(".gz").removesuffix(".nii")


def find_images(root: Path, dataset: str) -> list[Path]:
    """Mirror the notebook's CT selection without including dataset label maps."""
    if dataset == "kits19":
        return sorted(set(root.rglob("imaging.nii")) | set(root.rglob("imaging.nii.gz")))
    if dataset in {"msd-liver", "msd-colon"}:
        pattern = "liver_*" if dataset == "msd-liver" else "*"
        return sorted({p for subset in ("imagesTr", "imagesTs")
                       for ext in (".nii", ".nii.gz")
                       for p in (root / subset).glob(pattern + ext) if p.is_file()})
    return sorted(p for p in root.rglob("*") if p.is_file() and p.name.endswith((".nii", ".nii.gz")))


def output_path(path: Path, root: Path, output: Path, dataset: str) -> Path:
    stem = nifti_stem(path)
    if dataset == "kits19":
        return output / f"Kits19_{path.parent.name}_{stem}{SUFFIX}.nii.gz"
    if dataset in {"msd-liver", "msd-colon"}:
        name = "MSD_Task03_Liver" if dataset == "msd-liver" else "MSD_Task10_Colon"
        return output / f"{name}_{path.parent.name}_{stem}{SUFFIX}.nii.gz"
    return output / path.relative_to(root).parent / f"{stem}{SUFFIX}.nii.gz"


def describe(image) -> dict:
    return {
        "shape": list(image.shape),
        "spacing_mm": nib.affines.voxel_sizes(image.affine).tolist(),
        "orientation": "".join(nib.aff2axcodes(image.affine)),
        "origin": image.affine[:3, 3].tolist(),
        "affine": image.affine.tolist(),
        "dtype": str(image.get_data_dtype()),
        "spatial_units": image.header.get_xyzt_units()[0],
    }


def load_ct(path: Path):
    image = nib.load(path)
    if len(image.shape) != 3 or not np.isfinite(image.affine).all():
        raise ValueError(f"Expected a 3D CT with a finite affine: {path}")
    units = image.header.get_xyzt_units()[0]
    if units not in {"mm", "unknown"}:
        raise ValueError(f"CT affine must use millimetres, got {units}: {path}")
    if not np.isfinite(image.get_fdata()).all():
        raise ValueError(f"CT contains non-finite intensities: {path}")
    return image


def check_prepared(path: Path) -> dict:
    image = load_ct(path)
    report = describe(image)
    # Unknown units are accepted as millimetres, matching the original notebook.
    report["prepared"] = bool(np.allclose(image.affine, np.eye(4), atol=1e-3)
                              and image.get_data_dtype() == np.dtype("int16")
                              and np.array_equal(image.get_fdata(), np.rint(image.get_fdata())))
    return report


def prepare_one(source: Path, destination: Path) -> dict:
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError(f"Refusing to overwrite an existing file: {destination}")
    image = load_ct(source)
    canonical = nib.as_closest_canonical(image)
    # Preserve the notebook's linear interpolation and zero-valued boundary fill.
    resampled = resample_to_output(canonical, voxel_sizes=(1, 1, 1), order=1,
                                   mode="constant", cval=0.0)
    rounded = np.rint(resampled.get_fdata())
    bounds = np.iinfo(np.int16)
    if not np.isfinite(rounded).all() or rounded.min() < bounds.min or rounded.max() > bounds.max:
        raise ValueError(f"CT cannot be represented as int16 without overflow: {source}")
    output = nib.Nifti1Image(rounded.astype(np.int16), np.eye(4))
    output.header.set_xyzt_units("mm")
    output.set_sform(np.eye(4), code=1)
    output.set_qform(np.eye(4), code=1)
    output.header.set_data_dtype(np.int16)
    report = {"input": str(source), "output": str(destination),
              "before": describe(image), "resampled_affine_before_origin_reset": resampled.affine.tolist(),
              "after": describe(output), "interpolation_order": 1, "boundary_fill_hu": 0.0}
    sidecar = destination.with_name(nifti_stem(destination) + ".json")
    if sidecar.exists():
        raise ValueError(f"Refusing to overwrite preprocessing report: {sidecar}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output, destination)
    sidecar.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path, help="One CT NIfTI volume.")
    inputs.add_argument("--input-dir", type=Path, help="Dataset root, or directory containing only clean CTs.")
    parser.add_argument("--output-dir", type=Path, help="Separate output directory; required unless --check-only.")
    parser.add_argument("--dataset", choices=("generic", "kits19", "msd-liver", "msd-colon"), default="generic")
    parser.add_argument("--check-only", action="store_true", help="Inspect notebook preparation conventions without modifying files.")
    args = parser.parse_args()
    if args.image and args.dataset != "generic":
        parser.error("Use --dataset with --input-dir; a single --image uses its own filename.")
    root = args.input_dir.resolve() if args.input_dir else args.image.resolve().parent
    paths = find_images(root, args.dataset) if args.input_dir else [args.image.resolve()]
    if not paths:
        parser.error(f"No matching NIfTI images found in {root}")
    if args.check_only:
        valid = True
        for path in paths:
            report = check_prepared(path)
            print(json.dumps({"file": str(path), **report}))
            valid = valid and report["prepared"]
        if not valid:
            raise SystemExit(1)
        return
    if args.output_dir is None:
        parser.error("--output-dir is required for preparation.")
    output = args.output_dir.resolve()
    if output == root or root in output.parents:
        parser.error("Use a separate output directory outside the input directory.")
    destinations = [output_path(p, root, output, args.dataset) for p in paths]
    if len(set(destinations)) != len(destinations):
        parser.error("Input filenames would produce duplicate outputs.")
    for path, destination in zip(paths, destinations):
        report = prepare_one(path, destination)
        print(f"{path.name}: {report['before']['shape']} -> {report['after']['shape']} | {destination}")


if __name__ == "__main__":
    main()
