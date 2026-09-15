# Synthetic generation migration

Source: `m-sc-thesis-xabier-25-26/MASynthesisFull3D`, thesis commit `5df91ee`.

| Source | Destination |
| --- | --- |
| `masynthesis3dfull/*.py` except CPU `geometry.py`, `utils/io.py` | `src/ct_mar/synthesis/` |
| `simulation_demo_full3d.py` | `src/ct_mar/synthesis/generate.py` |
| Four `preprocessing/*.py` scripts | `src/ct_mar/synthesis/preprocessing/` |
| `data/xray_characteristic_data.csv` | Packaged `synthesis/resources/` |
| `data/implant_library/` | Local, Git-ignored `data/implant_library/` |
| `Preprocessing_Check_Nii_Metadata.ipynb` imaging preparation | `preprocessing/prepare_ct.py` |

Changes during extraction:

- Adapted imports and packaged the X-ray table so installation is independent of
  the thesis directory.
- Added `ct-mar-generate`, `ct-mar-prepare`, and optional synthesis/STL dependencies.
- Removed the CPU projection/reconstruction implementation and backend-selection
  flags. ASTRA with CUDA is mandatory for generation. Retained the original GPU
  projection angles and reconstruction calculation path.
- Added a seed for implant placement and NumPy projection noise.
- Replaced shape-based axis guessing with validation of prepared RAS, 1 mm inputs.
  Validate mask shape and affine against the CT and reject empty implant masks.
- Fail if a requested library implant is unavailable instead of silently falling
  back to a primitive shape. Explicit primitive generation is still available.
- Set provided-mask HU consistently in saved configuration.
- Fixed the CT implant builder's small-component filter to exclude background
  label zero; the source filter could turn background voxels into implant voxels.
- Added a small batch driver using installed modules, separate case folders,
  retained segmentations, and no cluster-specific paths. It replaces the original
  dataset runner, which could share segmentation folders between different CTs.

The original ASTRA simulation/reconstruction calculations are retained. The library builders
are retained with the component-filter fix above. Cluster jobs, dataset manifests, notebooks, and plotting utilities
were not copied. The implant library is byte-identical to the inspected HPC ZIP.

## Relationship to the original entry scripts

`generate.py` is adapted from `simulation_demo_full3d.py`, with the changes above.
It is not an identical copy. `batch.py` is a new, smaller replacement for
`build_artifact_dataset.py`, not a rename. Both batch drivers run anatomy
segmentation followed by simulation for each CT. The new driver keeps each case's
anatomy in its own folder, retains those masks, and rejects existing case outputs.
It does not reproduce the original skip-segmentation, filtering, or cleanup options.

CT resampling is independent of both drivers and never happens implicitly. The
notebook's imaging preparation was extracted into an optional command; its
inspection capability is available through `--check-only`. Training/validation
CSV splits remain outside this generation workflow. See `ct_preprocessing.md` for
notebook parity and intentional differences (including int16 overflow handling).

## Validation

Validated locally on macOS ARM64 with Python 3.11.16, NumPy 2.4.6, SciPy 1.17.1,
NiBabel 5.4.2, pandas 2.3.3, and trimesh 4.12.2.

Local tests cover CT orientation/resampling, output dtype and origin, geometry
reports, check-only behavior, dataset selection, invalid inputs, missing-CUDA
errors, misaligned masks, and CT implant extraction. GPU simulation tests are
explicitly opt-in with `CT_MAR_TEST_GPU=1`; they are skipped on the local machine.
The earlier CPU simulator was tested during the initial migration but has now
been removed; those results do not validate the GPU release.

ASTRA/CUDA, TotalSegmentator execution, full-size clinical volumes, and full batch
execution remain unvalidated. Local tests do not establish paper reproduction.
