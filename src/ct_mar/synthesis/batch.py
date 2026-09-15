"""Generate artifacts for a directory of prepared clean CT volumes."""

import argparse
from pathlib import Path
import subprocess
import sys

from .geometry_astra import require_astra_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--implant-library", type=Path, default=Path("data/implant_library"))
    parser.add_argument("--seed", type=int, default=0)
    args, simulation_args = parser.parse_known_args()
    source, output = args.input_dir.resolve(), args.output_dir.resolve()
    if output == source or source in output.parents:
        parser.error("Use an output directory outside the input directory.")
    if not args.implant_library.is_dir():
        parser.error(f"Implant library not found: {args.implant_library}")
    images = sorted(p for p in source.rglob("*") if p.is_file() and p.name.endswith((".nii", ".nii.gz")))
    if not images:
        parser.error("No NIfTI inputs found. Supply a directory containing only prepared clean CTs.")
    reserved = {"--image", "--output-dir", "--anatomy-dir", "--mask"}
    if any(arg.split("=", 1)[0] in reserved for arg in simulation_args):
        parser.error("Image, output, anatomy, and mask paths are managed per case in batch mode.")
    case_dirs = [output / p.relative_to(source).parent / p.name.removesuffix(".gz").removesuffix(".nii") for p in images]
    if len(set(case_dirs)) != len(case_dirs):
        parser.error("Input names produce duplicate output case directories.")
    require_astra_gpu()
    for idx, (image, case_dir) in enumerate(zip(images, case_dirs)):
        if case_dir.exists():
            parser.error(f"Case output already exists; use a new output directory: {case_dir}")
        anatomy = case_dir / "anatomy"
        subprocess.run([sys.executable, "-m", "ct_mar.synthesis.preprocessing.run_totalseg",
                        "--image", str(image), "--output-dir", str(anatomy), "--fast"], check=True)
        subprocess.run([sys.executable, "-m", "ct_mar.synthesis.generate",
                        "--image", str(image), "--anatomy-dir", str(anatomy),
                        "--output-dir", str(case_dir), "--implant-library", str(args.implant_library.resolve()),
                        "--seed", str(args.seed + idx), *simulation_args], check=True)


if __name__ == "__main__":
    main()
