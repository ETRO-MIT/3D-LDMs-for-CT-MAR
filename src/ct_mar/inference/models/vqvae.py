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
import torch.nn as nn
from monai.networks.blocks import Convolution
from monai.networks.layers import Act
from monai.utils import ensure_tuple_rep

from .vector_quantizer import EMAQuantizer, VectorQuantizer

__all__ = ["VQVAE"]


class VQVAEResidualUnit(nn.Module):
    """Residual unit for VQ-VAE."""

    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        num_res_channels: int,
        act: tuple | str | None = Act.RELU,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=num_channels,
            out_channels=num_res_channels,
            adn_ordering="DA",
            act=act,
            dropout=dropout,
            bias=bias,
        )
        self.conv2 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=num_res_channels,
            out_channels=num_channels,
            bias=bias,
            conv_only=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(x + self.conv2(self.conv1(x)), inplace=True)


class Encoder(nn.Module):
    """Encoder module for VQ-VAE."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_channels: Sequence[int],
        num_res_layers: int,
        num_res_channels: Sequence[int],
        downsample_parameters: Sequence[Sequence[int]],
        dropout: float,
        act: tuple | str | None,
    ) -> None:
        super().__init__()
        blocks = []
        for i in range(len(num_channels)):
            blocks.append(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels if i == 0 else num_channels[i - 1],
                    out_channels=num_channels[i],
                    strides=downsample_parameters[i][0],
                    kernel_size=downsample_parameters[i][1],
                    adn_ordering="DA",
                    act=act,
                    dropout=None if i == 0 else dropout,
                    dropout_dim=1,
                    dilation=downsample_parameters[i][2],
                    padding=downsample_parameters[i][3],
                )
            )
            for _ in range(num_res_layers):
                blocks.append(
                    VQVAEResidualUnit(
                        spatial_dims=spatial_dims,
                        num_channels=num_channels[i],
                        num_res_channels=num_res_channels[i],
                        act=act,
                        dropout=dropout,
                    )
                )

        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=num_channels[-1],
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class Decoder(nn.Module):
    """Decoder module for VQ-VAE."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_channels: Sequence[int],
        num_res_layers: int,
        num_res_channels: Sequence[int],
        upsample_parameters: Sequence[Sequence[int]],
        dropout: float,
        act: tuple | str | None,
        output_act: tuple | str | None,
    ) -> None:
        super().__init__()
        reversed_num_channels = list(reversed(num_channels))
        reversed_num_res_channels = list(reversed(num_res_channels))

        blocks = []
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=reversed_num_channels[0],
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        for i in range(len(num_channels)):
            for _ in range(num_res_layers):
                blocks.append(
                    VQVAEResidualUnit(
                        spatial_dims=spatial_dims,
                        num_channels=reversed_num_channels[i],
                        num_res_channels=reversed_num_res_channels[i],
                        act=act,
                        dropout=dropout,
                    )
                )

            blocks.append(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=reversed_num_channels[i],
                    out_channels=out_channels if i == len(num_channels) - 1 else reversed_num_channels[i + 1],
                    strides=upsample_parameters[i][0],
                    kernel_size=upsample_parameters[i][1],
                    adn_ordering="DA",
                    act=act,
                    dropout=dropout if i != len(num_channels) - 1 else None,
                    norm=None,
                    dilation=upsample_parameters[i][2],
                    conv_only=i == len(num_channels) - 1,
                    is_transposed=True,
                    padding=upsample_parameters[i][3],
                    output_padding=upsample_parameters[i][4],
                )
            )

        if output_act:
            blocks.append(Act[output_act]())

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class VQVAE(nn.Module):
    """
    Vector-Quantised Variational Autoencoder (VQ-VAE) for 3D volumetric CT compression.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_channels: Sequence[int] | int = (96, 96, 192),
        num_res_layers: int = 3,
        num_res_channels: Sequence[int] | int = (96, 96, 192),
        downsample_parameters: Sequence[Sequence[int]] = ((2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1)),
        upsample_parameters: Sequence[Sequence[int]] = ((2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
        num_embeddings: int = 32,
        embedding_dim: int = 64,
        embedding_init: str = "normal",
        commitment_cost: float = 0.25,
        decay: float = 0.5,
        epsilon: float = 1e-5,
        dropout: float = 0.0,
        act: tuple | str | None = Act.RELU,
        output_act: tuple | str | None = None,
        ddp_sync: bool = True,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_dims = spatial_dims
        self.num_channels = num_channels
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.use_checkpointing = use_checkpointing

        if isinstance(num_res_channels, int):
            num_res_channels = ensure_tuple_rep(num_res_channels, len(num_channels))

        self.encoder = Encoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=embedding_dim,
            num_channels=num_channels,
            num_res_layers=num_res_layers,
            num_res_channels=num_res_channels,
            downsample_parameters=downsample_parameters,
            dropout=dropout,
            act=act,
        )

        self.decoder = Decoder(
            spatial_dims=spatial_dims,
            in_channels=embedding_dim,
            out_channels=out_channels,
            num_channels=num_channels,
            num_res_layers=num_res_layers,
            num_res_channels=num_res_channels,
            upsample_parameters=upsample_parameters,
            dropout=dropout,
            act=act,
            output_act=output_act,
        )

        self.quantizer = VectorQuantizer(
            quantizer=EMAQuantizer(
                spatial_dims=spatial_dims,
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost,
                decay=decay,
                epsilon=epsilon,
                embedding_init=embedding_init,
                ddp_sync=ddp_sync,
            )
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing:
            return torch.utils.checkpoint.checkpoint(self.encoder, images, use_reentrant=False)
        return self.encoder(images)

    def quantize(self, encodings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_loss, x = self.quantizer(encodings)
        return x, x_loss

    def decode(self, quantizations: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing:
            return torch.utils.checkpoint.checkpoint(self.decoder, quantizations, use_reentrant=False)
        return self.decoder(quantizations)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        quantizations, quantization_losses = self.quantize(self.encode(images))
        reconstruction = self.decode(quantizations)
        return reconstruction, quantization_losses

    def encode_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        e, _ = self.quantize(z)
        return e

    def decode_stage_2_outputs(self, z: torch.Tensor) -> torch.Tensor:
        e, _ = self.quantize(z)
        return self.decode(e)

