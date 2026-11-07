from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks import ConvNormAct
from src.models.blocks import DownsampleBlock
from shared.src.chiptune_shared.nn.blocks.upsample import UpsampleBlock
from src.models.blocks import ResidualBlock
from src.models.blocks import SpatialSelfAttention
from src.models.blocks import PatchDiscriminatorBlock

from shared.src.chiptune_shared.nn.model_utils import (
    NormType,
    PaddingMode,
    InitType,
    init_weights,
    count_parameters,
)


@dataclass
class GeneratorConfig:
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 32
    max_channels: int = 256

    num_downsamples: int = 2
    num_res_blocks: int = 6

    norm: NormType = "instance"
    padding_mode: PaddingMode = "reflect"

    residual_dropout: float = 0.0
    encoder_dropout: float = 0.0
    decoder_dropout: float = 0.0

    use_attention: bool = True
    num_att_blocks: int = 1
    attention_position: Literal["balanced", "middle", "after_resblocks", "none"] = "middle"
    attention_reduction: int = 8

    upsample_mode: Literal["nearest", "bilinear"] = "nearest"

    final_activation: Literal["tanh", "none"] = "tanh"

    init_type: InitType = "normal"
    init_gain: float = 0.02

    use_se: bool = False
    se_reduction: int = 16


@dataclass
class DiscriminatorConfig:
    in_channels: int = 1
    base_channels: int = 32
    max_channels: int = 512

    num_layers: int = 3
    norm: NormType = "instance"

    spectral_norm: bool = False
    use_sigmoid: bool = False

    init_type: InitType = "normal"
    init_gain: float = 0.02


def validate_attention_config(config: GeneratorConfig) -> None:
    if not config.use_attention:
        return

    if config.num_att_blocks < 1:
        raise ValueError(
            "num_att_blocks must be >= 1 when use_attention=True. "
            f"Got num_att_blocks={config.num_att_blocks}."
        )

    if config.num_att_blocks > config.num_res_blocks:
        raise ValueError(
            "num_att_blocks must be <= num_res_blocks. "
            f"Got num_att_blocks={config.num_att_blocks}, "
            f"num_res_blocks={config.num_res_blocks}."
        )

    valid_positions = {"middle", "after_resblocks", "balanced"}

    if config.attention_position not in valid_positions:
        raise ValueError(
            f"Unknown attention_position={config.attention_position!r}. "
            f"Expected one of {sorted(valid_positions)}."
        )


def make_residual_block(
    channels: int,
    config: GeneratorConfig,
) -> nn.Module:
    return ResidualBlock(
        channels=channels,
        padding_mode=config.padding_mode,
        norm=config.norm,
        dropout=config.residual_dropout,
        use_se=config.use_se,
        se_reduction=config.se_reduction,
    )


def make_attention_block(
    channels: int,
    config: GeneratorConfig,
) -> nn.Module:
    return SpatialSelfAttention(
        channels=channels,
        reduction=config.attention_reduction,
    )


def split_resblocks_for_balanced_attention(
    num_res_blocks: int,
    num_att_blocks: int,
) -> list:
    """
    Splits residual blocks into num_att_blocks + 1 chunks.

    Example:
        num_res_blocks=6, num_att_blocks=2 -> [2, 2, 2]
        num_res_blocks=7, num_att_blocks=2 -> [3, 2, 2]
        num_res_blocks=8, num_att_blocks=3 -> [2, 2, 2, 2]
    """
    num_chunks = num_att_blocks + 1
    base = num_res_blocks // num_chunks
    remainder = num_res_blocks % num_chunks

    chunks = []

    for chunk_idx in range(num_chunks):
        size = base + (1 if chunk_idx < remainder else 0)
        chunks.append(size)

    return chunks


def build_bottleneck_layers(
    channels: int,
    config: GeneratorConfig,
) -> list[nn.Module]:
    validate_attention_config(config)

    layers: list[nn.Module] = []

    if not config.use_attention:
        for _ in range(config.num_res_blocks):
            layers.append(make_residual_block(channels, config))
        return layers

    if config.attention_position == "after_resblocks":
        for _ in range(config.num_res_blocks):
            layers.append(make_residual_block(channels, config))

        for _ in range(config.num_att_blocks):
            layers.append(make_attention_block(channels, config))

        return layers

    if config.attention_position == "middle":
        before = config.num_res_blocks // 2
        after = config.num_res_blocks - before

        for _ in range(before):
            layers.append(make_residual_block(channels, config))

        for _ in range(config.num_att_blocks):
            layers.append(make_attention_block(channels, config))

        for _ in range(after):
            layers.append(make_residual_block(channels, config))

        return layers

    if config.attention_position == "balanced":
        chunks = split_resblocks_for_balanced_attention(
            num_res_blocks=config.num_res_blocks,
            num_att_blocks=config.num_att_blocks,
        )

        for chunk_idx, chunk_size in enumerate(chunks):
            for _ in range(chunk_size):
                layers.append(make_residual_block(channels, config))

            # Insert attention between chunks, not after the final chunk.
            if chunk_idx < config.num_att_blocks:
                layers.append(make_attention_block(channels, config))

        return layers

    raise ValueError(f"Unhandled attention_position: {config.attention_position}")


class AudioResnetGenerator(nn.Module):
    """
    ResNet-style CycleGAN generator for CQT tensors.

    Input:
    [B, 1, F, T]

    Output:
    [B, 1, F, T]

    For your current preprocessing:
    [B, 1, 96, 172] -> [B, 1, 96, 172]

    Structure:
    Initial 7x7 conv
    Downsample x N
    Residual blocks + optional bottleneck attention
    Upsample x N
    Final 7x7 conv + Tanh
    """

    def __init__(self, config: GeneratorConfig = GeneratorConfig()) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []

        current_channels = config.base_channels

        # Initial Projection
        layers.append(
            ConvNormAct(
                in_channels=config.in_channels,
                out_channels=current_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode=config.padding_mode,
                norm=config.norm,
                activation="relu",
                dropout=0.0,
            )
        )

        # Encoder/Downsampling
        for _ in range(config.num_downsamples):
            next_channels = min(current_channels * 2, config.max_channels)

            layers.append(
                DownsampleBlock(
                    in_channels=current_channels,
                    out_channels=next_channels,
                    norm=config.norm,
                    activation="relu",
                    padding_mode=config.padding_mode,
                    dropout=config.encoder_dropout,
                    spectral_norm=False,
                )
            )

            current_channels = next_channels

        # Bottleneck residual transformation
        layers.extend(
            build_bottleneck_layers(
                channels=current_channels,
                config=config,
            )
        )

        # Decoder/Upsampling
        for _ in range(config.num_downsamples):
            next_channels = max(config.base_channels, current_channels // 2)

            layers.append(
                UpsampleBlock(
                    in_channels=current_channels,
                    out_channels=next_channels,
                    norm=config.norm,
                    activation="relu",
                    padding_mode=config.padding_mode,
                    dropout=config.decoder_dropout,
                    upsample_mode=config.upsample_mode,
                )
            )

            current_channels = next_channels

        self.body = nn.Sequential(*layers)

        # Final projection
        final_layers: list[nn.Module] = [
            ConvNormAct(
                in_channels=current_channels,
                out_channels=config.out_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode=config.padding_mode,
                norm="none",
                activation="none",
                dropout=0.0,
            )
        ]

        if config.final_activation == "tanh":
            final_layers.append(nn.Tanh())

        self.final = nn.Sequential(*final_layers)

        self.apply(lambda m: init_weights(m, config.init_type, config.init_gain))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_spatial_size = x.shape[-2:]

        h = self.body(x)

        # Safety for odd dimensions or future parameter changes.
        if h.shape[-2:] != input_spatial_size:
            h = F.interpolate(
                h,
                size=input_spatial_size,
                mode="bilinear",
                align_corners=False,
            )

        y = self.final(h)

        return y


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator for CQT tensors.

    Input:
    [B, 1, F, T]

    Output:
    [B, 1, H_patch, W_patch]

    The discriminator evaluates local CQT patches rather than producing a single
    real/fake scalar.
    """

    def __init__(self, config: DiscriminatorConfig = DiscriminatorConfig()) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []

        current_channels = config.base_channels

        # First layer usually has no normalization
        layers.append(
            PatchDiscriminatorBlock(
                in_channels=config.in_channels,
                out_channels=current_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                norm=config.norm,
                spectral_norm=config.spectral_norm,
                use_norm=False,
            )
        )

        # Deeper strided layers
        for layer_idx in range(1, config.num_layers):
            next_channels = min(config.base_channels * (2 ** layer_idx), config.max_channels)

            layers.append(
                PatchDiscriminatorBlock(
                    in_channels=current_channels,
                    out_channels=next_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    norm=config.norm,
                    spectral_norm=config.spectral_norm,
                    use_norm=True,
                )
            )

            current_channels = next_channels

        # One stride-1 refinement layer
        next_channels = min(current_channels * 2, config.max_channels)

        layers.append(
            PatchDiscriminatorBlock(
                in_channels=current_channels,
                out_channels=next_channels,
                kernel_size=4,
                stride=1,
                padding=1,
                norm=config.norm,
                spectral_norm=config.spectral_norm,
                use_norm=True,
            )
        )

        current_channels = next_channels

        # Final patch prediction layer
        layers.append(
            nn.Conv2d(
                in_channels=current_channels,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=True,
            )
        )

        if config.use_sigmoid:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

        self.apply(lambda m: init_weights(m, config.init_type, config.init_gain))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiScalePatchGANDiscriminator(nn.Module):
    """
    Optional multi-scale discriminator wrapper.

    It runs several PatchGANDdiscriminators on progressively downsampled inputs.
    Useful later if one discriminator only learns very local texture and misses
    broader rhythmic/spectral structure.

    * For first baseline training, a single PatchGANDiscriminator is enough.
    """
    def __init__(
            self,
            config: DiscriminatorConfig = DiscriminatorConfig(),
            num_scales: int = 2,
    ) -> None:
        super().__init__()

        self.num_scales = num_scales

        self.discriminators = nn.ModuleList(
            [PatchGANDiscriminator(config) for _ in range(num_scales)]
        )

        self.downsample = nn.AvgPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
            count_include_pad=False,
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []

        current = x

        for discriminator in self.discriminators:
            outputs.append(discriminator(current))
            current = self.downsample(current)

        return outputs


def build_generator(
        **kwargs,
) -> AudioResnetGenerator:
    config = GeneratorConfig(**kwargs)
    return AudioResnetGenerator(config)


def build_discriminator(
        **kwargs,
) -> PatchGANDiscriminator:
    config = DiscriminatorConfig(**kwargs)
    return PatchGANDiscriminator(config)


def describe_model(model: nn.Module) -> str:
    params = count_parameters(model)
    return f"{model.__class__.__name__}(trainable_params={params})"

