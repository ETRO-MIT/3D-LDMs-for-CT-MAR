"""
run_totalseg.py

Helper script to run TotalSegmentator as a preprocessing step.
This script should be executed BEFORE running the metal artifact simulation.

It performs anatomical segmentation and stores the results on disk
to be reused by the simulation pipeline.
"""

import argparse
import subprocess
from pathlib import Path

import nibabel as nib


def _remove_corrupted_nifti_files(output_dir: Path) -> int:
    """
    Remove unreadable/corrupted NIfTI files from the output directory.

    Returns
    -------
    int
        Number of removed files.
    """
    removed = 0
    for file_path in output_dir.glob("*.nii*"):
        try:
            nib.load(str(file_path)).get_fdata(dtype="float32")
        except (nib.filebasedimages.ImageFileError, EOFError, OSError, ValueError):
            file_path.unlink(missing_ok=True)
            removed += 1
    return removed


def run_totalseg(input_ct: Path, output_dir: Path, fast: bool = False, roi_subset: list = None, body_seg: bool = False):
    """
    Run TotalSegmentator on a CT volume.

    Parameters
    ----------
    input_ct : Path
        Path to input CT volume (.nii or .nii.gz).
    output_dir : Path
        Directory where segmentation masks will be stored.
    fast : bool
        Whether to use the fast TotalSegmentator mode.
    roi_subset : list, optional
        List of specific ROIs to segment.
    body_seg : bool
        Whether to crop to body region first (saves memory).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "TotalSegmentator",
        "-i", str(input_ct),
        "-o", str(output_dir),
    ]

    if fast:
        cmd.append("--fast")

    if body_seg:
        cmd.append("--body_seg")

    # Force single threading for resampling/saving to reduce memory spikes
    cmd.extend(["--nr_thr_resamp", "1", "--nr_thr_saving", "1"])

    if roi_subset:
        # Handle "pelvis" alias which is not a valid TotalSegmentator class
        # Expand it to hip_left, hip_right, sacrum
        final_rois = []
        for roi in roi_subset:
            if roi == "pelvis":
                final_rois.extend(["hip_left", "hip_right", "sacrum"])
            else:
                final_rois.append(roi)
        # Deduplicate
        final_rois = sorted(list(set(final_rois)))

        print(f"[run_totalseg] Selected ROIs: {final_rois}")
        cmd.append("--roi_subset")
        cmd.extend(final_rois)
    else:
        print("[run_totalseg] No ROI subset specified. Segmenting all classes.")

    print("[run_totalseg] Running TotalSegmentator")
    print("[run_totalseg] Command:", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[run_totalseg] Interrupted by user. Cleaning corrupted outputs...")
        removed = _remove_corrupted_nifti_files(output_dir)
        print(f"[run_totalseg] Removed {removed} corrupted file(s).")
        raise
    except subprocess.CalledProcessError as e:
        if e.returncode == -9:
            print("\n[run_totalseg] ERROR: Process was killed (SIGKILL).")
            print("[run_totalseg] This almost always means the job ran out of memory (RAM).")
            print("[run_totalseg] Solution: Request more memory (e.g. --mem=64G) or use --body-seg.")
        print("[run_totalseg] TotalSegmentator failed. Cleaning corrupted outputs...")
        removed = _remove_corrupted_nifti_files(output_dir)
        print(f"[run_totalseg] Removed {removed} corrupted file(s).")
        raise

    print("[run_totalseg] Segmentation completed.")
    print("[run_totalseg] Results saved to:", output_dir)
    
    created_labels = sorted([f.stem for f in output_dir.glob("*.nii*")])
    print(f"[run_totalseg] Output labels: {created_labels}")


def main():
    parser = argparse.ArgumentParser(description="Run TotalSegmentator preprocessing.")
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Path to input CT image (.nii or .nii.gz).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to store TotalSegmentator outputs.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast TotalSegmentator mode (lower accuracy, faster).",
    )
    parser.add_argument(
        "--roi-subset",
        nargs="+",
        help="List of specific ROIs to segment (e.g. liver spleen femur_left).",
    )
    parser.add_argument(
        "--body-seg",
        action="store_true",
        help="Crop to body region before segmentation to reduce memory usage.",
    )

    args = parser.parse_args()
    run_totalseg(args.image, args.output_dir, args.fast, args.roi_subset, args.body_seg)


if __name__ == "__main__":
    main()
