# Upstream acknowledgments

This document records migrated simulation code and model components planned for
migration. Model implementations have not yet been imported.

## Adapted MedLoRD code

- Source: the thesis repository's `medlord_xabier/` subtree.
- Upstream work: *MedLoRD: A Medical Low-Resource Diffusion Model for
  High-Resolution 3D CT Image Synthesis*, https://arxiv.org/abs/2503.13211.
- That subtree includes an Apache License 2.0 text, preserved without modification
  at `third_party/Apache-2.0.txt`.
- Record exact imported files, upstream revisions where available, and local
  modifications when the implementation is extracted.

## MONAI components

The thesis copies of the VQ-VAE and diffusion inferer, among other modules, carry
`Copyright (c) MONAI Consortium` headers and Apache License 2.0 notices. Preserve
the headers in each migrated file and record changes to those files.

## Synthetic generation

The simulation code was copied from the thesis repository's `MASynthesisFull3D/`
subtree at commit `5df91ee`. See `docs/synthesis_migration.md` for file mappings and
changes. The original-code license remains undecided.
TotalSegmentator and ASTRA are planned external tools; their own licenses apply.
They are external dependencies and are not bundled.

Implant masks, STL meshes, CT examples, and pretrained model weights are separate
artifacts. Their redistribution terms must be documented individually when added
to a release; this notice does not assign them the code's license.
