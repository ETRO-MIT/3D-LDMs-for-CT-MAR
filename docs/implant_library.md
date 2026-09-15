# Optional implant-library builders

Use the prepared library for ordinary artifact generation. These tools are only
needed to create or add implant masks.

## Extract a mask from CT

```bash
python -m ct_mar.synthesis.preprocessing.build_implant_library \
  --image data/implant_ct.nii.gz \
  --implant-library data/custom_implant_library \
  --category pelvic_screws --implant-id custom_screw \
  --hu-min 2500 --keep-largest --resample-iso --crop
```

The script thresholds the CT and optionally filters, resamples, aligns, and crops
the mask. Inspect the extracted shape: thresholding alone may retain several
connected objects or artifact voxels. Use a new implant ID for each addition.

## Convert an STL mesh

```bash
python -m pip install -e ".[synthesis,implant-tools]"
python -m ct_mar.synthesis.preprocessing.build_implant_library_from_stl \
  --mesh data/custom_implant.stl \
  --implant-library data/custom_implant_library \
  --category hip_implants --implant-id custom_hip \
  --voxel-mm 1 --fill --crop
```

Mesh coordinates must represent millimetres. Both builders write a binary NIfTI
mask and update the category's `metadata.json`. The current placement algorithm
uses mask voxel arrays directly, so prepare masks at 1 mm spacing for 1 mm CTs.
The builders retain their original options; see each command's `--help`.

Keep source assets and generated libraries outside Git. Before distributing a
library, review source attribution, redistribution terms, and source paths in
metadata. Re-running a builder with the same ID currently appends another metadata
entry and overwrites the matching mask; use a fresh output library for experiments.
