# Generate synthetic metal artifacts

## Install

From the repository root, activate a Python 3.11 environment and install:

```bash
python -m pip install -e ".[synthesis]"

# Install CUDA-enabled ASTRA Toolbox for GPU 3D simulation:
conda install -c astra-toolbox -c conda-forge astra-toolbox -y
```

For 3D projection and FDK reconstruction, an NVIDIA GPU and CUDA-enabled [ASTRA Toolbox](https://astra-toolbox.com/docs/install.html) are required. There is no CPU simulation path. Image preparation and file handling use NumPy/SciPy on the CPU.

## Inputs

- A clean 3D NIfTI CT in HU, prepared in axis-aligned RAS orientation at 1 mm
  isotropic spacing. This command validates this preparation; it does not resample.
- A prepared implant library located under `data/implant_library/` (e.g. sample hip implants).
- Anatomy masks in the same voxel grid as the CT for automatic region selection.

## Optional dataset preparation

The notebook preparation step is now available as `ct-mar-prepare`. **Skip it for
HPC volumes already prepared with the notebook.** Generation never calls it
automatically. See [CT preparation](ct_preprocessing.md) for the exact steps and
file selection options.

For unprepared data, prepare the CT first, then segment the resulting volume:

```bash
ct-mar-prepare --image data/raw/scan.nii.gz --output-dir data/prepared
```

For already-prepared HPC data, optionally inspect it without writing anything:

```bash
ct-mar-prepare --input-dir data/prepared --check-only
```

## Run one case

### Option A: All-in-One (Automated on-the-fly segmentation)
TotalSegmentator runs automatically in a temporary directory, selects and positions the implant based on bone overlap, and deletes the temporary masks when complete:

```bash
# Recommended via Main.py:
python Main.py generate \
  --input data/clean_ct.nii.gz \
  --output_dir outputs/generated \
  --metal titanium

# Or via CLI:
ct-mar-generate \
  --image data/clean_ct.nii.gz \
  --output-dir outputs/generated \
  --metal titanium \
  --seed 0
```
*(Pass `--keep-anatomy` or `--keep_anatomy` if you want to preserve the generated anatomy masks under `output-dir/anatomy`).*

### Option B: Using pre-computed anatomy masks
If you already ran TotalSegmentator or have organ masks available:

```bash
# Via Main.py:
python Main.py generate \
  --input data/clean_ct.nii.gz \
  --anatomy_dir outputs/anatomy \
  --implant_library data/implant_library \
  --output_dir outputs/generated \
  --metal titanium

# Or via CLI:
ct-mar-generate \
  --image data/clean_ct.nii.gz \
  --anatomy-dir outputs/anatomy \
  --implant-library data/implant_library \
  --output-dir outputs/generated \
  --metal titanium \
  --seed 0
```

The source mapping automatically selects `hip_implants` for hip and `spine_screws`
for spine. Other categories require `--implant-category`, for example
`--implant-category pelvic_screws`. Use `--implant-id` to choose an ID from that
category's `metadata.json`. If the requested library implant cannot be selected,
the command fails instead of substituting a primitive.

Alternatively, supply `--mask path/to/implant_mask.nii.gz` for an aligned implant
mask, or `--implant-source primitive` for a sphere/cylinder example. Neither path
requires anatomy segmentation. Material selection follows the original simulation
code: `Titanium` or `Iron`; supplied masks use the configuration's default titanium
material. The source name `Iron` is retained pending reconciliation with the paper's
steel terminology.

## Outputs

For an input named `clean_ct.nii.gz`:

| File | Contents |
| --- | --- |
| `synth_clean_ct.nii.gz` | Artifact-corrupted CT in HU |
| `implant_only_clean_ct.nii.gz` | Original CT with inserted implant, without simulated artifacts |
| `synth_clean_ct_metal_mask.nii.gz` | Binary implant mask |
| `synth_clean_ct.json` | Simulation settings, selected implant, material, and seed |

The generated volumes keep the input shape and affine. Use a new output directory
for each run; the single-case command overwrites matching output filenames.

## Batch generation

Put only prepared clean CT volumes in the input directory. This command segments
each case, generates its artifacts, and retains anatomy masks in separate case
folders. Existing case output folders are rejected to avoid accidental reuse.

```bash
python -m ct_mar.synthesis.batch \
  --input-dir data/clean_cts --output-dir outputs/dataset \
  --implant-library data/implant_library --seed 0
```

Additional simulation flags, such as `--implant-category`, are passed to each case.
The seed is incremented in sorted input order. Full batch execution and GPU
reconstruction still require validation on the target machine.

## Validation

Run the preprocessing, input-validation, and builder tests locally:

```bash
python -m unittest discover -s tests -v
```

On a machine with CUDA-enabled ASTRA and an NVIDIA GPU, also enable the small
simulation checks:

```bash
CT_MAR_TEST_GPU=1 python -m unittest discover -s tests -v
```

GPU tests are skipped by default; passing local tests does not validate GPU
reconstruction. Simulation defaults were inherited from the original single-case
script; the published dataset settings must be verified separately before claiming
exact paper reproduction.

See [optional implant builders](implant_library.md) to extend the library.
