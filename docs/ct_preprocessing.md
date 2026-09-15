# Optional CT preparation

This command extracts the imaging preparation from the thesis notebook
`MASynthesisFull3D/Preprocessing_Check_Nii_Metadata.ipynb`. Run it only when the
input data needs preparation. The HPC datasets already processed with that
notebook can skip this command entirely.

## What it does

1. Loads a 3D CT NIfTI in HU and reorients it to RAS.
2. Resamples to 1 × 1 × 1 mm using linear interpolation, following the notebook's
   `nibabel.processing.resample_to_output` call (including zero-valued boundary fill).
3. Rounds intensities and saves `int16`, without applying a HU clipping window.
4. Sets the output origin to `(0, 0, 0)`, the spatial units to mm, and both NIfTI
   spatial-form codes to 1.
5. Saves a JSON sidecar describing the original geometry, the resampled geometry
   before origin reset, and the final geometry.

Resetting the origin deliberately changes the world-coordinate reference, as in
the notebook. Keep the original CT and the JSON report. Run anatomy segmentation
on the prepared CT; old masks in the original image grid cannot be used directly.

## One CT

```bash
ct-mar-prepare --image data/raw/scan.nii.gz --output-dir data/prepared
```

Output: `data/prepared/scan_iso1mm_RAS_origin0_int16.nii.gz` plus its JSON report.
Use `python -m ct_mar.synthesis.preprocessing.prepare_ct` if running without the
console command after an editable installation.

## A directory

```bash
ct-mar-prepare --input-dir data/raw_cts --output-dir data/prepared
```

Generic mode preserves relative subdirectories and processes every NIfTI in the
input directory. Supply only clean CT images, not segmentation masks.

For the original datasets, choose a file-selection mode:

| `--dataset` | Selected files |
| --- | --- |
| `generic` (default; also suitable for CLINIC) | All `.nii` and `.nii.gz` recursively |
| `kits19` | Only `imaging.nii` and `imaging.nii.gz` recursively |
| `msd-liver` | `liver_*.nii[.gz]` inside `imagesTr` and `imagesTs` |
| `msd-colon` | NIfTI files inside `imagesTr` and `imagesTs` |

```bash
ct-mar-prepare --input-dir data/kits19/data \
  --output-dir data/kits19_prepared --dataset kits19
```

KiTS/MSD outputs include case or subset names, following the notebook's naming
intent. Repeated `.nii` suffixes produced by the original notebook are removed.
Use generic mode when checking already-prepared files with renamed filenames.

## Check existing HPC data without changing it

```bash
ct-mar-prepare --input-dir data/prepared --check-only
```

This prints shape, spacing, orientation, origin, dtype, and a `prepared` boolean
for each CT. It checks for the notebook's RAS/1 mm/origin-zero/int16 convention
and integer intensities. A nonzero exit status means a file fails the checks.
The check does not prove that the data contains HU or that its anatomy is correct.

The generation command only requires aligned RAS/1 mm inputs; it also accepts a
nonzero origin and floating-point HU data. Failure of the stricter notebook check
does not always mean generation is impossible.

## Differences from the notebook

- User-supplied paths replace hard-coded local dataset paths.
- Only 3D CT inputs are accepted, matching the simulation. No automatic 4D frame
  selection or label-map interpolation is performed.
- Non-finite intensities and int16 overflow raise errors; the notebook could wrap
  out-of-range values during casting. Valid in-range inputs use the same steps.
- Unknown spatial units are assumed to be mm, as in the notebook. Explicit units
  other than mm are rejected rather than silently misinterpreted.
- Existing outputs are not overwritten. Use a separate output folder.
- Training/validation split creation is not included; it is not part of preparing
  a CT for artifact generation.

NiBabel's resampling API is documented [here](https://nipy.org/nibabel/reference/nibabel.processing.html#resample-to-output).
