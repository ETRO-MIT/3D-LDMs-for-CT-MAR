import unittest
import numpy as np

from ct_mar.inference.transforms import (
    normalize_ct_hu,
    denormalize_to_hu,
    pad_or_crop_3d,
    restore_to_original_shape,
    CT_HU_MIN,
    CT_HU_MAX_METAL,
)
from ct_mar.inference.models.metadata import (
    normalize_region,
    normalize_side,
    normalize_metal_name,
)

try:
    import torch
    import torch.nn as nn
    from ct_mar.inference.models import (
        DDPMScheduler,
        MetadataConditioningEncoder,
        MetadataConditionedDiffusionModel,
        DiffusionModelUNet,
        VQVAE,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestInferenceTransforms(unittest.TestCase):
    def test_hu_normalization_and_denormalization(self):
        test_hu = np.array([-1200.0, -1000.0, 0.0, 1500.0, 4000.0, 5000.0], dtype=np.float32)
        norm = normalize_ct_hu(test_hu)
        self.assertEqual(norm[0], -1.0)
        self.assertEqual(norm[1], -1.0)
        self.assertEqual(norm[4], 1.0)
        self.assertEqual(norm[5], 1.0)
        self.assertTrue(np.all(norm >= -1.0) and np.all(norm <= 1.0))

        recovered = denormalize_to_hu(norm)
        expected_clamped = np.clip(test_hu, CT_HU_MIN, CT_HU_MAX_METAL)
        np.testing.assert_allclose(recovered, expected_clamped, atol=1e-3)

    def test_pad_or_crop_3d_padding(self):
        small_vol = np.ones((10, 20, 30), dtype=np.float32)
        target_size = (16, 32, 40)
        padded, meta = pad_or_crop_3d(small_vol, target_size, pad_value=-1.0)
        self.assertEqual(padded.shape, target_size)
        self.assertEqual(padded[0, 0, 0], -1.0)

        restored = restore_to_original_shape(padded, (10, 20, 30), meta)
        np.testing.assert_array_equal(restored, small_vol)

    def test_pad_or_crop_3d_cropping(self):
        large_vol = np.random.randn(50, 50, 50).astype(np.float32)
        target_size = (32, 32, 32)
        cropped, meta = pad_or_crop_3d(large_vol, target_size)
        self.assertEqual(cropped.shape, target_size)

        restored = restore_to_original_shape(cropped, (50, 50, 50), meta, fill_from=large_vol)
        self.assertEqual(restored.shape, (50, 50, 50))


class TestMetadataHandling(unittest.TestCase):
    def test_vocab_normalizations(self):
        self.assertEqual(normalize_region("HIP"), "hip")
        self.assertEqual(normalize_region("brain"), "unknown")
        self.assertEqual(normalize_side("Left"), "left")
        self.assertEqual(normalize_side("coronal"), "unknown")
        self.assertEqual(normalize_metal_name("Titanium"), "titanium")
        self.assertEqual(normalize_metal_name("gold"), "unknown")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch and MONAI required for neural network tests")
class TestInferenceModels(unittest.TestCase):
    def test_metadata_encoder_output_shape(self):
        encoder = MetadataConditioningEncoder(context_dim=64)
        tokens = encoder.encode_tokens_from_strings(
            regions=["hip", "spine"],
            sides=["left", "midline"],
            metal_names=["titanium", "iron"],
            device=torch.device("cpu"),
        )
        self.assertEqual(tokens.shape, (2, 3, 64))

    def test_ddpm_scheduler(self):
        scheduler = DDPMScheduler(num_train_timesteps=50, schedule="cosine", prediction_type="v_prediction")
        scheduler.set_timesteps(5)
        self.assertEqual(len(scheduler.timesteps), 5)

        sample = torch.randn(2, 4, 8, 8, 8)
        model_out = torch.randn(2, 4, 8, 8, 8)
        t = scheduler.timesteps[0].item()
        prev_sample, pred_x0 = scheduler.step(model_out, t, sample)
        self.assertEqual(prev_sample.shape, sample.shape)
        self.assertEqual(pred_x0.shape, sample.shape)

    def test_small_vqvae_forward(self):
        model = VQVAE(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_channels=[16, 32],
            num_res_channels=[16, 32],
            num_res_layers=1,
            downsample_parameters=[[2, 4, 1, 1], [2, 4, 1, 1]],
            upsample_parameters=[[2, 4, 1, 1, 0], [2, 4, 1, 1, 0]],
            num_embeddings=64,
            embedding_dim=4,
        )
        x = torch.randn(1, 1, 16, 16, 16)
        latent = model.encode_stage_2_inputs(x)
        self.assertEqual(latent.shape, (1, 4, 4, 4, 4))
        decoded = model.decode_stage_2_outputs(latent)
        self.assertEqual(decoded.shape, (1, 1, 16, 16, 16))


if __name__ == "__main__":
    unittest.main()

