# 3D-LDMs-CT-MAR

Companion repository for **Large-Volume Conditioned 3D Latent Diffusion Models for CT Metal Artifact Suppression**, DGM4MICCAI 2026.

## Development status

The ASTRA GPU artifact generation pipeline, optional CT preprocessing, and
implant-library builders are available. Artifact suppression is still a placeholder pending migration.
See [synthetic generation](docs/synthetic_generation.md) for the runnable workflow.

The two independent workflows are:

1. **Synthetic artifact generation:** prepare anatomy and implant masks, insert an
   implant into a clean CT, and generate an artifact-corrupted CT together with its
   implant-only reference, metal mask, and metadata.
2. **Artifact suppression:** process an artifact-corrupted CT using either the
   anatomy-conditioned or anatomy-metadata-conditioned model.

Each workflow will have its own setup instructions, configuration, and example.
Users can choose synthetic generation or artifact suppression independently.

Trained weights will be hosted externally. Both suppression models use the same
autoencoder. Inference will require that autoencoder and the selected diffusion
model, including the metadata encoder for the metadata-conditioned variant.

## Local development installation

Use Python 3.11. From the repository directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[synthesis]"
python -c "import ct_mar; print(ct_mar.__version__)"
```

An editable installation makes changes under `src/` available without reinstalling
the package. The `synthesis` extra installs the numerical and NIfTI dependencies.
CUDA-enabled ASTRA and an NVIDIA GPU are required for simulation. TotalSegmentator
is needed when producing anatomy masks; see the generation guide.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/ct_mar/synthesis/` | Synthetic artifact generation |
| `src/ct_mar/inference/` | Artifact suppression with both model variants |
| `configs/` | Portable workflow configurations |
| `examples/generate_artifacts/` | Synthetic generation example |
| `examples/suppress_artifacts/` | Artifact suppression example |
| `docs/` | Setup and workflow documentation |
| `weights/` | External checkpoint release information; no weight files |
| `third_party/` | Preserved upstream license texts |

See [the development guide](docs/development.md) for the branch and commit workflow.

## Attribution and licensing

The implementation will be extracted from the thesis codebase, including its
adapted MedLoRD and MONAI components. See [upstream acknowledgments](THIRD_PARTY_NOTICES.md)
and [license status](LICENSE.md). The license for original project code and the
terms for model weights have not yet been selected.
