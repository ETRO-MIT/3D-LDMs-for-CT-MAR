# Upstream acknowledgments

This document records migrated simulation code and model components from the thesis repository.

## Adapted MedLoRD and MONAI components

- Source: `medlord_xabier/` subtree.
- Upstream works:
  - *MedLoRD: A Medical Low-Resource Diffusion Model for High-Resolution 3D CT Image Synthesis*, https://arxiv.org/abs/2503.13211.
  - *MONAI Generative Models* / *Diffusers* (Apache License 2.0).
- Apache License 2.0 text is preserved at `third_party/Apache-2.0.txt`.
- Migrated modules under `src/ct_mar/inference/`:
  - `models/diffusion_unet.py`: 3D Diffusion UNet backbone with cross-attention and spatial attention blocks (MONAI / Diffusers).
  - `models/vqvae.py`: 3D VQ-VAE autoencoder (MONAI).
  - `models/vector_quantizer.py`: EMA and Vector Quantizer codebook modules (DeepMind / MONAI).
  - `models/schedulers.py`: DDPMScheduler supporting cosine beta schedule and v-prediction (MONAI / Diffusers).
  - `models/metadata.py`: Categorical metadata cross-attention conditioning encoder and wrapper (MedLoRD adaptation).
  - `inferer.py`: Latent diffusion reverse sampling loop (MONAI).

## Synthetic generation

The simulation code was copied from the thesis repository's `MASynthesisFull3D/`
subtree at commit `5df91ee`. See `docs/synthesis_migration.md` for file mappings and
changes. The original-code license remains undecided.
TotalSegmentator and ASTRA are planned external tools; their own licenses apply.
They are external dependencies and are not bundled.

Implant masks, STL meshes, CT examples, and pretrained model weights are separate
artifacts. Their redistribution terms must be documented individually when added
to a release; this notice does not assign them the code's license.
