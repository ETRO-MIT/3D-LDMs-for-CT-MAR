"""CT preparation checks using small synthetic images."""

import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np

from ct_mar.synthesis.preprocessing.prepare_ct import (
    check_prepared, find_images, output_path, prepare_one,
)


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def save(self, data, affine=None):
        path = self.root / "input.nii.gz"
        image = nib.Nifti1Image(data, np.eye(4) if affine is None else affine)
        image.header.set_xyzt_units("mm")
        nib.save(image, path)
        return path

    def test_flip_resample_round_origin_and_report(self):
        # Left-to-right reversal and 2 mm spacing: after preparation the x ramp
        # must be [20,15,10,5,0] on a 1 mm grid, regardless of original translation.
        data = np.broadcast_to(np.array([0, 10, 20], dtype=np.float32)[:, None, None], (3, 2, 2)).copy()
        affine = np.diag([-2., 1., 1., 1.])
        affine[:3, 3] = [30, 40, 50]
        source = self.save(data, affine)
        destination = self.root / "out/prepared.nii.gz"
        report = prepare_one(source, destination)
        image = nib.load(destination)
        self.assertEqual(image.shape, (5, 2, 2))
        np.testing.assert_array_equal(image.get_fdata()[:, 0, 0], [20, 15, 10, 5, 0])
        np.testing.assert_array_equal(image.affine, np.eye(4))
        self.assertEqual(image.get_data_dtype(), np.dtype("int16"))
        self.assertEqual(image.header.get_xyzt_units()[0], "mm")
        self.assertEqual(int(image.header["qform_code"]), 1)
        self.assertEqual(int(image.header["sform_code"]), 1)
        np.testing.assert_array_equal(report["before"]["affine"], affine)
        self.assertEqual(report["resampled_affine_before_origin_reset"][0][3], 26)
        self.assertEqual(json.loads((destination.parent / "prepared.json").read_text()), report)
        self.assertTrue(check_prepared(destination)["prepared"])

    def test_no_hu_clipping_and_check_does_not_write(self):
        data = np.array([-1200.4, 5000.6, -2.5, 2.5, 0, 10, 20, 30], dtype=np.float32).reshape(2, 2, 2)
        source = self.save(data)
        destination = self.root / "prepared.nii.gz"
        prepare_one(source, destination)
        np.testing.assert_array_equal(nib.load(destination).get_fdata(), np.rint(data))
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        self.assertTrue(check_prepared(destination)["prepared"])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})
        with self.assertRaisesRegex(ValueError, "overwrite"):
            prepare_one(source, destination)

    def test_reject_overflow_and_nonfinite_and_4d(self):
        for data in (np.full((2, 2, 2), 40000., dtype=np.float32),
                     np.full((2, 2, 2), np.nan, dtype=np.float32),
                     np.zeros((2, 2, 2, 2), dtype=np.float32)):
            with self.subTest(shape=data.shape, first=data.flat[0]):
                source = self.save(data)
                with self.assertRaises(ValueError):
                    prepare_one(source, self.root / "bad.nii.gz")
                self.assertFalse((self.root / "bad.nii.gz").exists())

    def test_dataset_selection_and_names(self):
        for name in ("case_00000/imaging.nii.gz", "case_00000/segmentation.nii.gz",
                     "imagesTr/liver_0.nii.gz", "imagesTs/colon_0.nii", "labelsTr/liver_0.nii.gz"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        self.assertEqual([p.name for p in find_images(self.root, "kits19")], ["imaging.nii.gz"])
        self.assertEqual([p.name for p in find_images(self.root, "msd-liver")], ["liver_0.nii.gz"])
        self.assertEqual(len(find_images(self.root, "msd-colon")), 2)
        output = output_path(self.root / "case_00000/imaging.nii.gz", self.root, self.root / "out", "kits19")
        self.assertEqual(output.name, "Kits19_case_00000_imaging_iso1mm_RAS_origin0_int16.nii.gz")


if __name__ == "__main__":
    unittest.main()
