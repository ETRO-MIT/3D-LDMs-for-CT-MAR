"""Validation tests; set CT_MAR_TEST_GPU=1 on a CUDA machine for GPU simulation."""

import json
import os
from unittest.mock import patch
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np


class SynthesisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.affine = np.eye(4)
        self.affine[:3, 3] = [12, -8, 20]
        self.ct = np.full((16, 16, 16), -1000, dtype=np.float32)
        self.ct[3:13, 3:13, 3:13] = 50
        self.mask = np.zeros_like(self.ct, dtype=np.uint8)
        self.mask[7:9, 7:9, 7:9] = 1
        self.save("ct.nii.gz", self.ct)
        self.save("mask.nii.gz", self.mask)

    def save(self, name, data, affine=None):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(data, self.affine if affine is None else affine), path)
        return path

    def run_module(self, module, *args):
        return subprocess.run([sys.executable, "-m", module, *map(str, args)],
                              capture_output=True, text=True)

    def generate(self, output, *extra):
        return self.run_module("ct_mar.synthesis.generate", "--image", self.root / "ct.nii.gz",
                               "--output-dir", self.root / output,
                               "--angle-num", 4, "--detector-pixels", 16, "--seed", 7, *extra)

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(os.environ.get("CT_MAR_TEST_GPU") == "1", "Requires ASTRA and NVIDIA GPU")
    def test_provided_mask_outputs_and_repeatability(self):
        for folder in ("first", "second"):
            self.assert_success(self.generate(folder, "--mask", self.root / "mask.nii.gz"))
        first = self.root / "first"
        for name in ("synth_ct.nii.gz", "implant_only_ct.nii.gz", "synth_ct_metal_mask.nii.gz"):
            image = nib.load(first / name)
            self.assertEqual(image.shape, self.ct.shape)
            np.testing.assert_allclose(image.affine, self.affine)
            self.assertTrue(np.isfinite(image.get_fdata()).all())
            np.testing.assert_array_equal(image.get_fdata(), nib.load(self.root / "second" / name).get_fdata())
        target = nib.load(first / "implant_only_ct.nii.gz").get_fdata()
        np.testing.assert_array_equal(target[self.mask == 0], self.ct[self.mask == 0])
        np.testing.assert_array_equal(target[self.mask > 0], 3000)
        np.testing.assert_array_equal(nib.load(first / "synth_ct_metal_mask.nii.gz").get_fdata(), self.mask)
        config = json.loads((first / "synth_ct.json").read_text())
        self.assertEqual(config["placement_metadata"]["seed"], 7)
        self.assertEqual(config["metal_hu"], 3000)

    def test_reject_misaligned_mask(self):
        wrong = self.affine.copy()
        wrong[0, 3] += 1
        self.save("mask.nii.gz", self.mask, wrong)
        result = self.generate("bad", "--mask", self.root / "mask.nii.gz")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must match the CT shape and affine", result.stderr)
        self.assertFalse((self.root / "bad").exists())

    @unittest.skipUnless(os.environ.get("CT_MAR_TEST_GPU") == "1", "Requires ASTRA and NVIDIA GPU")
    def test_library_selection_and_missing_id(self):
        category = self.root / "library/hip_implants"
        self.save("library/hip_implants/test.nii.gz", np.ones((3, 3, 3), dtype=np.uint8), np.eye(4))
        (category / "metadata.json").write_text(json.dumps({"items": [{"id": "test", "mask_file": "test.nii.gz"}]}))
        flags = ("--implant-library", category.parent, "--implant-category", "hip_implants")
        self.assert_success(self.generate("library_output", *flags, "--implant-id", "test"))
        config = json.loads((self.root / "library_output/synth_ct.json").read_text())
        self.assertEqual(config["placement_metadata"]["implant_id_selected"], "test")
        result = self.generate("missing", *flags, "--implant-id", "missing")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Could not select a library implant", result.stderr)

    def test_no_cpu_fallback(self):
        from ct_mar.synthesis.geometry_astra import require_astra_gpu
        with patch("ct_mar.synthesis.geometry_astra.astra", None):
            with self.assertRaisesRegex(RuntimeError, "no CPU simulation fallback"):
                require_astra_gpu()
        with patch("ct_mar.synthesis.geometry_astra.astra") as astra:
            astra.use_cuda.return_value = False
            with self.assertRaisesRegex(RuntimeError, "CUDA-enabled"):
                require_astra_gpu()

    def test_ct_library_builder(self):
        ct = self.ct.copy()
        ct[self.mask > 0] = 3000
        self.save("implant_ct.nii.gz", ct)
        self.assert_success(self.run_module(
            "ct_mar.synthesis.preprocessing.build_implant_library",
            "--image", self.root / "implant_ct.nii.gz", "--implant-library", self.root / "built",
            "--category", "screws", "--implant-id", "test", "--min-voxels", 1, "--crop"))
        mask = nib.load(self.root / "built/screws/test.nii.gz").get_fdata()
        self.assertEqual(int(mask.sum()), int(self.mask.sum()))
        metadata = json.loads((self.root / "built/screws/metadata.json").read_text())
        self.assertEqual(metadata["items"][0]["mask_file"], "test.nii.gz")


if __name__ == "__main__":
    unittest.main()
