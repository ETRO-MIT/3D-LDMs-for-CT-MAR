from __future__ import annotations

import json
from pathlib import Path
import torch
import torch.nn as nn

DEFAULT_REGION_VOCAB = ["unknown", "spine", "hip", "knee", "shoulder"]
DEFAULT_SIDE_VOCAB = ["unknown", "left", "right", "midline", "bilateral"]
DEFAULT_METAL_NAME_VOCAB = ["unknown", "iron", "titanium"]


def _safe_lower(value: str | None) -> str:
    return str(value or "").strip().lower()


def normalize_region(region: str | None, region_vocab: list[str] | None = None) -> str:
    vocab = set(region_vocab or DEFAULT_REGION_VOCAB)
    region_l = _safe_lower(region)
    return region_l if region_l in vocab else "unknown"


def normalize_side(side: str | None, side_vocab: list[str] | None = None) -> str:
    vocab = set(side_vocab or DEFAULT_SIDE_VOCAB)
    side_l = _safe_lower(side)
    return side_l if side_l in vocab else "unknown"


def normalize_metal_name(metal_name: str | None, metal_name_vocab: list[str] | None = None) -> str:
    vocab = set(metal_name_vocab or DEFAULT_METAL_NAME_VOCAB)
    metal_name_l = _safe_lower(metal_name)
    return metal_name_l if metal_name_l in vocab else "unknown"


def parse_metadata_json(metadata_path: str | Path) -> dict[str, str]:
    """Extract region, side, and metal material attributes from a metadata JSON file."""
    path = Path(metadata_path)
    if not path.exists():
        return {"region": "unknown", "side": "unknown", "metal_name": "unknown"}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    placement = data.get("placement_metadata", {}) or {}
    region = normalize_region(placement.get("region") or data.get("region"))

    main_label = _safe_lower(placement.get("main_label") or data.get("main_label"))
    if any(tok in main_label for tok in ["bilateral", "both"]):
        side = "bilateral"
    elif any(tok in main_label for tok in ["_left", " left", "left_", "left"]):
        side = "left"
    elif any(tok in main_label for tok in ["_right", " right", "right_", "right"]):
        side = "right"
    elif region == "spine":
        side = "midline"
    else:
        side = normalize_side(placement.get("side") or data.get("side"))

    metal_name = normalize_metal_name(data.get("metal_name") or placement.get("metal_name"))
    return {"region": region, "side": side, "metal_name": metal_name}


class MetadataConditioningEncoder(nn.Module):
    """
    Encodes categorical metadata (region, side, metal material) into cross-attention tokens.
    Produces tensor of shape [batch_size, 3, context_dim].
    """

    def __init__(
        self,
        context_dim: int = 128,
        region_vocab: list[str] | None = None,
        side_vocab: list[str] | None = None,
        metal_name_vocab: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.region_vocab = list(region_vocab or DEFAULT_REGION_VOCAB)
        self.side_vocab = list(side_vocab or DEFAULT_SIDE_VOCAB)
        self.metal_name_vocab = list(metal_name_vocab or DEFAULT_METAL_NAME_VOCAB)

        self.region_to_idx = {name: idx for idx, name in enumerate(self.region_vocab)}
        self.side_to_idx = {name: idx for idx, name in enumerate(self.side_vocab)}
        self.metal_to_idx = {name: idx for idx, name in enumerate(self.metal_name_vocab)}

        self.region_embedding = nn.Embedding(len(self.region_vocab), context_dim)
        self.side_embedding = nn.Embedding(len(self.side_vocab), context_dim)
        self.metal_name_embedding = nn.Embedding(len(self.metal_name_vocab), context_dim)
        self.field_embedding = nn.Embedding(3, context_dim)

        self.token_mlp = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
        )

    def encode_tokens_from_strings(
        self,
        regions: list[str],
        sides: list[str],
        metal_names: list[str],
        device: torch.device,
    ) -> torch.Tensor:
        ids = []
        for r, s, m in zip(regions, sides, metal_names):
            r_idx = self.region_to_idx.get(normalize_region(r, self.region_vocab), self.region_to_idx.get("unknown", 0))
            s_idx = self.side_to_idx.get(normalize_side(s, self.side_vocab), self.side_to_idx.get("unknown", 0))
            m_idx = self.metal_to_idx.get(normalize_metal_name(m, self.metal_name_vocab), self.metal_to_idx.get("unknown", 0))
            ids.append([r_idx, s_idx, m_idx])
        ids_tensor = torch.tensor(ids, dtype=torch.long, device=device)
        return self(ids_tensor)

    def forward(self, metadata_ids: torch.Tensor) -> torch.Tensor:
        if metadata_ids.ndim != 2 or metadata_ids.shape[1] != 3:
            raise ValueError(
                f"Expected metadata_ids with shape [batch, 3], got {tuple(metadata_ids.shape)}"
            )
        region_token = self.region_embedding(metadata_ids[:, 0])
        side_token = self.side_embedding(metadata_ids[:, 1])
        metal_name_token = self.metal_name_embedding(metadata_ids[:, 2])
        tokens = torch.stack([region_token, side_token, metal_name_token], dim=1)

        field_ids = torch.arange(3, device=metadata_ids.device, dtype=torch.long)
        tokens = tokens + self.field_embedding(field_ids).unsqueeze(0)
        return self.token_mlp(tokens)


class MetadataConditionedDiffusionModel(nn.Module):
    """
    Wraps the 3D DiffusionModelUNet backbone and injects metadata tokens as cross-attention context.
    The artifacted CT condition arrives through channel concatenation.
    """

    def __init__(
        self,
        diffusion_model: nn.Module,
        metadata_encoder: MetadataConditioningEncoder,
    ) -> None:
        super().__init__()
        self.diffusion_model = diffusion_model
        self.metadata_encoder = metadata_encoder

    def encode_metadata(self, metadata_ids: torch.Tensor) -> torch.Tensor:
        return self.metadata_encoder(metadata_ids)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor | None = None,
        metadata: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if metadata is not None:
            if context is not None:
                raise ValueError("Provide either metadata or context, not both.")
            context = self.encode_metadata(metadata)

        if context is not None:
            context = context.to(device=x.device, dtype=x.dtype)

        return self.diffusion_model(
            x=x,
            timesteps=timesteps,
            context=context,
            **kwargs,
        )

