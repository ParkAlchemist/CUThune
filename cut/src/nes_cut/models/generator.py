from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Literal, Iterable

from shared.src.chiptune_shared.nn.blocks.conv_norm_act import ConvNormAct
from shared.src.chiptune_shared.nn.blocks.convnext import ConvNeXtContextBlock2d
from shared.src.chiptune_shared.nn.blocks.downsample import DownsampleBlock
from shared.src.chiptune_shared.nn.blocks.upsample import UpsampleBlock


@dataclass(frozen=True)
class CUTGeneratorConfig:
    in_channels: int = 1
    out_channels: int = 1

    stem_channels: int = 64
    num_sampling_blocks: int = 2
    sampling_ratio: int = 2

    num_bottleneck_blocks: int = 3
    convnext_kernel_size: int = 7
    convnext_dilation: int = 1
    expansion_ratio: int = 4

    use_grn: bool = True
    grn_eps: float = 1e-6
    layer_scale_initial: float = 1e-6

    norm_eps: float = 1e-6

    input_freq_pad_value: float = -1.0
    feature_freq_pad_value: float = 0.0

    stem_kernel_size: int = 7
    head_kernel_size: int = 7

    output_activation: Literal["tanh"] = "tanh"


def validate_generator_config(
    config: CUTGeneratorConfig,
) -> None:
    channel_values = {
        "in_channels": config.in_channels,
        "out_channels": config.out_channels,
        "stem_channels": config.stem_channels,
        "down_1_channels": config.stem_channels * config.sampling_ratio,
        "bottleneck_channels": config.stem_channels * (config.sampling_ratio**config.num_sampling_blocks),
    }

    for name, value in channel_values.items():
        if value <= 0:
            raise ValueError(
                f"{name} must be positive, got {value}"
            )

    if config.num_bottleneck_blocks <= 0:
        raise ValueError(
            "num_bottleneck_blocks must be positive, "
            f"got {config.num_bottleneck_blocks}"
        )

    for name, value in {
        "stem_kernel_size": config.stem_kernel_size,
        "head_kernel_size": config.head_kernel_size,
        "convnext_kernel_size": config.convnext_kernel_size,
    }.items():
        if value <= 0 or value % 2 == 0:
            raise ValueError(
                f"{name} must be a positive odd integer, got {value}"
            )

    if config.expansion_ratio <= 0:
        raise ValueError(
            "expansion_ratio must be positive, "
            f"got {config.expansion_ratio}"
        )

    if config.norm_eps <= 0.0:
        raise ValueError(
            f"norm_eps must be positive, got {config.norm_eps}"
        )


FeatureMapDict = dict[str, torch.Tensor]


class CUTGenerator(nn.Module):
    def __init__(
            self,
            config: CUTGeneratorConfig,
    ) -> None:
        super().__init__()

        validate_generator_config(config)

        self.config = config

        stem_padding = config.stem_kernel_size // 2
        head_padding = config.head_kernel_size // 2

        self.stem = ConvNormAct(
            in_channels=config.in_channels,
            out_channels=config.stem_channels,
            kernel_size=config.stem_kernel_size,
            stride=1,
            padding=stem_padding,
            padding_mode="cqt",
            padding_value=config.input_freq_pad_value,
            norm="layer",
            activation="gelu",
            bias=None,
            spectral_norm=False,
            norm_affine=True,
        )

        self.downs = nn.ModuleList([])

        current_channels = config.stem_channels

        for i in range(config.num_sampling_blocks):
            self.downs.append(
                DownsampleBlock(
                    in_channels=current_channels,
                    out_channels=current_channels * config.sampling_ratio,
                    mode="pixel_unshuffle",
                    norm="layer",
                    activation="gelu",
                    padding_mode="cqt",
                    padding_val=config.feature_freq_pad_value,
                    spectral_norm=False,
                    ratio=config.sampling_ratio,
                )
            )

            current_channels = current_channels * self.config.sampling_ratio

        self.bottleneck_blocks = nn.ModuleList([])

        for i in range(config.num_bottleneck_blocks):
            self.bottleneck_blocks.append(
                ConvNeXtContextBlock2d(
                    channels=current_channels,
                    kernel_size=config.convnext_kernel_size,
                    dilation=config.convnext_dilation,
                    expansion_ratio=config.expansion_ratio,
                    layer_scale_initial=config.layer_scale_initial,
                    use_grn=config.use_grn,
                    grn_eps=config.grn_eps,
                )
            )

        self.ups = nn.ModuleList([])

        for i in range(config.num_sampling_blocks):
            self.ups.append(
                UpsampleBlock(
                    in_channels=current_channels,
                    out_channels=current_channels // config.sampling_ratio,
                    norm="layer",
                    activation="gelu",
                    padding_mode="cqt",
                    padding_val=config.feature_freq_pad_value,
                    upsample_mode="nearest",
                )
            )

            current_channels = current_channels // config.sampling_ratio

        self.head = ConvNormAct(
            in_channels=current_channels,
            out_channels=config.out_channels,
            kernel_size=config.head_kernel_size,
            stride=1,
            padding=head_padding,
            padding_mode="cqt",
            padding_value=config.feature_freq_pad_value,
            norm="none",
            activation="tanh",
            bias=True,
            spectral_norm=False,
        )

        self.available_feature_layers = tuple(
            ["stem"] +
            [f"down_{i + 1}" for i in range(config.num_sampling_blocks)] +
            [f"bottleneck_{i + 1}" for i in range(config.num_bottleneck_blocks)]
        )

        self.apply(initialize_cut_module)

    @staticmethod
    def _validate_feature_layers(
            requested: tuple[str, ...],
            available: tuple[str, ...],
    ) -> None:
        unknown = sorted(
            set(requested) - set(available)
        )

        if unknown:
            raise ValueError(
                f"Unknown feature layers: {unknown}. "
                f"Available layers: {list(available)}"
            )

        if len(set(requested)) != len(requested):
            raise ValueError(
                "feature_layers must not contain duplicates"
            )

    @staticmethod
    def _pad_time_to_multiple(
            x: torch.Tensor,
            multiple: int = 4,
    ) -> tuple[torch.Tensor, int]:
        original_frames = x.shape[-1]
        pad_frames = (multiple - original_frames % multiple) % multiple

        if pad_frames == 0:
            return x, original_frames

        if original_frames <= pad_frames:
            raise RuntimeError(
                f"Time dimension is too short for reflection padding: "
                f"T={original_frames}, P={pad_frames}"
            )

        x = F.pad(x, (0, pad_frames, 0, 0), mode="reflect")

        return x, original_frames

    @staticmethod
    def _collect_feature(
            features: FeatureMapDict,
            requested: set[str],
            name: str,
            value: torch.Tensor,
    ) -> None:
        if name in requested:
            features[name] = value

    def encode(
            self,
            x: torch.Tensor,
            *,
            feature_layers: Iterable[str] = ()
    ) -> tuple[torch.Tensor, FeatureMapDict, int]:
        if x.ndim != 4:
            raise RuntimeError(
                f"CUTGenerator only support 4D input. Got {tuple(x.shape)}"
            )

        if x.shape[1] != self.config.in_channels:
            raise RuntimeError(
                f"CUTGenerator channel mismatch: "
                f"Expected {self.config.in_channels}, got {x.shape[1]}"
            )

        requested_tuple = tuple(feature_layers)
        self._validate_feature_layers(
            requested=requested_tuple,
            available=self.available_feature_layers,
        )
        requested = set(requested_tuple)

        features: FeatureMapDict = {}

        x, original_frames = self._pad_time_to_multiple(
            x,
            multiple=self.config.sampling_ratio**self.config.num_sampling_blocks,
        )

        x = self.stem(x)
        self._collect_feature(features, requested, "stem", x)

        for idx, down in enumerate(self.downs):
            x = down(x)
            self._collect_feature(features, requested, f"down_{idx+1}", x)

        for idx, bottleneck in enumerate(self.bottleneck_blocks):
            x = bottleneck(x)
            self._collect_feature(features, requested, f"bottleneck_{idx+1}", x)

        ordered_features = {
            name: features[name] for name in requested_tuple
        }

        return x, ordered_features, original_frames

    def decode(
            self,
            latent: torch.Tensor,
            *,
            output_frames: int | None = None,
    ) -> torch.Tensor:
        if latent.ndim != 4:
            raise RuntimeError(
                f"CUTGenerator only support 4D input. Got {tuple(latent.shape)}"
            )
        if latent.shape[1] != self.config.stem_channels*(self.config.sampling_ratio**self.config.num_sampling_blocks):
            raise RuntimeError(
                f"CUTGenerator channel mismatch: "
                f"Expected {self.config.stem_channels*(self.config.sampling_ratio**self.config.num_sampling_blocks)}, "
                f"got {latent.shape[1]}"
            )

        x = latent
        for up in self.ups:
            x = up(x)

        x = self.head(x)

        if output_frames is not None:
            if output_frames <= 0:
                raise ValueError(
                    f"output frames should be greater than 0, got {output_frames}"
                )

            if output_frames > x.shape[-1]:
                raise ValueError(
                    f"output frames exceeds decoded length: "
                    f"requested length: {output_frames}, "
                    f"available length: {x.shape[-1]}"
                )

            x = x[..., :output_frames]

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent, _, original_frames = self.encode(x)
        return self.decode(latent, output_frames=original_frames)

    def forward_with_features(
            self,
            x: torch.Tensor,
            *,
            feature_layers: Iterable[str],
    ) -> tuple[torch.Tensor, FeatureMapDict]:
        latent, features, original_frames = self.encode(x, feature_layers=feature_layers)
        output = self.decode(latent, output_frames=original_frames)
        return output, features


def initialize_cut_module(
    module: nn.Module,
) -> None:
    if isinstance(module, nn.Conv2d):
        weight = getattr(
            module,
            "weight_orig",
            module.weight,
        )
        nn.init.trunc_normal_(
            weight,
            std=0.02,
        )

        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(
            module.weight,
            std=0.02,
        )

        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.LayerNorm):
        if module.elementwise_affine:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


