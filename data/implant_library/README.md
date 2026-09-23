# 3D Implant Library

This directory provides voxelized 3D metal implant masks for synthetic CT metal artifact simulation.

## Structure

```text
implant_library/
├── hip_implants/         # 7 femoral stems, cups, and lattice hip prostheses
│   ├── *.nii.gz         # Binary 3D implant masks (1 mm isotropic, PCA-aligned)
│   └── metadata.json    # Category metadata, bounding boxes, and principal axes
└── spine_screws/         # 7 orthopedic fixation and pedicle screws
    ├── *.nii.gz         # Binary 3D screw masks (1 mm isotropic, PCA-aligned)
    └── metadata.json    # Category metadata, bounding boxes, and principal axes
```

## Usage

These implants are loaded automatically by `ct-mar-generate` based on the targeted anatomical region:
- **Hip CTs (`hip`):** Selects from `hip_implants/`.
- **Spine CTs (`spine`):** Selects from `spine_screws/`.

Custom implants can be added by placing additional 1 mm isotropic binary `.nii.gz` masks in the appropriate folder and updating `metadata.json`.

