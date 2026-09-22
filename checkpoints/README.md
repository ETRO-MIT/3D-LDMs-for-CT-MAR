# Pretrained Model Checkpoints

This directory stores the pretrained model checkpoints required for metal artifact suppression (MAR) inference with 3D Latent Diffusion Models.

---

## Required Checkpoint Files

| Model | Checkpoint Filename | Architecture / Description | Parameter Size |
|---|---|---|---|
| **Stage 1 VQ-VAE** | `vqvae_checkpoint.pth` | 3D VQ-VAE autoencoder with $4\times$ spatial downsampling (`configs/inference/vqvae_ds4.yaml`) | ~33.5 MB |
| **Model 2.1: Anatomy LDM** | `anatomy_ldm_checkpoint.pth` | 3D Latent Diffusion Model conditioned on corrupted anatomy via channel concatenation (`configs/inference/anatomy_ldm.yaml`) | ~340 MB |
| **Model 2.2: Anatomy + Metadata LDM** | `anatomy_metadata_ldm_checkpoint.pth` | 3D Latent Diffusion Model conditioned on corrupted anatomy and categorical metadata (region, side, metal) via cross-attention (`configs/inference/anatomy_metadata_ldm.yaml`) | ~341 MB |

---

## Automatic Download

The easiest way to obtain the pretrained weights is using the repository download script:

```bash
# Download all model weights (recommended)
python DownloadWeights.py

# Or via Main.py
python Main.py download

# Or download only a specific model
python DownloadWeights.py --model anatomy
python DownloadWeights.py --model anatomy_metadata
python DownloadWeights.py --model vqvae
```

### Private / Gated Repository Access
If the repository is private or requires authorization, supply your Hugging Face user access token:

```bash
python DownloadWeights.py --token <YOUR_HF_TOKEN>
# Or export as an environment variable:
export HF_TOKEN=<YOUR_HF_TOKEN>
python DownloadWeights.py
```

---

## Manual Download via Hugging Face Hub CLI

You can also use the official `huggingface-cli`:

```bash
# Install huggingface_hub
pip install huggingface_hub

# Download to the checkpoints directory
huggingface-cli download ETRO-MIT/3D-LDMs-for-CT-MAR vqvae_checkpoint.pth --local-dir checkpoints/
huggingface-cli download ETRO-MIT/3D-LDMs-for-CT-MAR anatomy_ldm_checkpoint.pth --local-dir checkpoints/
huggingface-cli download ETRO-MIT/3D-LDMs-for-CT-MAR anatomy_metadata_ldm_checkpoint.pth --local-dir checkpoints/
```

---

## Expected Directory Layout

Once downloaded, the `checkpoints/` directory should look like this:

```text
checkpoints/
├── README.md
├── vqvae_checkpoint.pth
├── anatomy_ldm_checkpoint.pth
└── anatomy_metadata_ldm_checkpoint.pth
```
