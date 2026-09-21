# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence
import torch
from torch import nn

__all__ = ["VectorQuantizer", "EMAQuantizer"]


class EMAQuantizer(nn.Module):
    """
    Vector Quantization module using Exponential Moving Average (EMA) to learn codebook
    parameters (Oord et al., Neural Discrete Representation Learning).
    """

    def __init__(
        self,
        spatial_dims: int,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        embedding_init: str = "normal",
        ddp_sync: bool = True,
        restart_threshold: float = 1.0,
        max_iterations: int = 200000,
        training_step: int = 20000,
    ) -> None:
        super().__init__()
        self.spatial_dims: int = spatial_dims
        self.embedding_dim: int = embedding_dim
        self.num_embeddings: int = num_embeddings
        self.max_iterations: int = max_iterations
        self.training_step: int = training_step

        if self.spatial_dims not in [2, 3]:
            raise ValueError(f"EMAQuantizer supports 2D and 3D tensors, got {spatial_dims}.")

        self.embedding: nn.Embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        if embedding_init == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.embedding.weight.data, mode="fan_in", nonlinearity="linear")
        self.embedding.weight.requires_grad = False

        self.commitment_cost: float = commitment_cost
        self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())

        self.decay: float = decay
        self.epsilon: float = epsilon
        self.ddp_sync: bool = ddp_sync
        self.restart_threshold: float = restart_threshold

        self.flatten_permutation: Sequence[int] = [0] + list(range(2, self.spatial_dims + 2)) + [1]
        self.quantization_permutation: Sequence[int] = [0, self.spatial_dims + 1] + list(
            range(1, self.spatial_dims + 1)
        )

    def quantize(
        self, inputs: torch.Tensor, batch_size: int | None = 32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoding_indices_view = list(inputs.shape)
        del encoding_indices_view[1]

        inputs = inputs.float()
        inputs = inputs.permute(self.flatten_permutation).contiguous().view(-1, self.embedding_dim)

        total = inputs.size(0)
        if not batch_size or batch_size <= 0:
            batch_size = total

        num_batches = (total + batch_size - 1) // batch_size
        quantized_indices_batches = []
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, total)
            batch_input = inputs[start:end]

            distances = (
                (batch_input ** 2).sum(dim=1, keepdim=True)
                + (self.embedding.weight.t() ** 2).sum(dim=0, keepdim=True)
                - 2 * torch.mm(batch_input, self.embedding.weight.t())
            )
            encoding_indices = torch.max(-distances, dim=1)[1]
            quantized_indices_batches.append(encoding_indices)

        encoding_indices = torch.cat(quantized_indices_batches).view(encoding_indices_view)
        return inputs, encoding_indices

    def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(embedding_indices).permute(self.quantization_permutation).contiguous()

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_input, encoding_indices = self.quantize(inputs)
        quantized = self.embed(encoding_indices)

        loss = self.commitment_cost * nn.functional.mse_loss(quantized.detach(), inputs)
        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss, encoding_indices


class VectorQuantizer(nn.Module):
    """Wrapper around quantizer for forward and index embedding."""

    def __init__(self, quantizer: nn.Module | None = None) -> None:
        super().__init__()
        self.quantizer: nn.Module = quantizer

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        quantized, loss, _ = self.quantizer(inputs)
        return loss, quantized

    def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        return self.quantizer.embed(embedding_indices=embedding_indices)

    def quantize(self, encodings: torch.Tensor) -> torch.Tensor:
        _, _, encoding_indices = self.quantizer(encodings)
        return encoding_indices

