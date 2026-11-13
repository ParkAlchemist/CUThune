from __future__ import annotations

import torch
import torch.nn as nn

from collections.abc import Sequence

from shared.src.chiptune_shared.nn.blocks.global_response_norm import GlobalResponseNorm1d, GlobalResponseNorm2d
from shared.src.chiptune_shared.nn.blocks.temporal_attention import RoPETemporalSelfAttention1d
from vocoder.src.nes_vocoder.config.schema import ContextAttentionConfig


class ConvNeXtContextBlock1d(nn.Module):
    def __init__(
            self,
            channels: int,
            kernel_size: int = 7,
            dilation: int = 1,
            expansion_ratio: int = 2,
            layer_scale_initial: float = 1e-6,
            use_grn: bool = False,
            grn_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")

        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")

        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")

        if dilation <= 0:
            raise ValueError(f"dilation must be positive, got {dilation}")

        if expansion_ratio <= 0:
            raise ValueError(f"expansion_ratio must be positive, got {expansion_ratio}")

        if layer_scale_initial < 0.0:
            raise ValueError(
                "layer_scale_initial must be non-negative, "
                f"got {layer_scale_initial}"
            )

        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.expansion_ratio = expansion_ratio
        self.layer_scale_initial = layer_scale_initial

        padding = dilation * (kernel_size - 1) // 2

        self.depthwise = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            groups=channels,
        )

        self.norm = nn.LayerNorm(normalized_shape=channels)

        expanded_channels = channels * expansion_ratio

        self.expand = nn.Linear(
            in_features=channels,
            out_features=expanded_channels,
        )

        self.activation = nn.GELU()

        if use_grn:
            self.grn = GlobalResponseNorm1d(
                channels=expanded_channels,
                eps=grn_eps,
            )
        else:
            self.grn = nn.Identity()

        self.contract = nn.Linear(
            in_features=expanded_channels,
            out_features=channels
        )

        self.layer_scale = nn.Parameter(
            torch.full(
                (channels,),
                fill_value=layer_scale_initial,
                dtype=torch.float32,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise RuntimeError(f"ConvNeXtContextBlock1d only supports 3 dimensional tensors, "
                               f"got {tuple(x.shape)}")

        if x.shape[1] != self.channels:
            raise RuntimeError(f"ConvNeXtContextBlock1d channel mismatch: "
                               f"expected {self.channels}, got {x.shape[1]}")

        res = x

        x = self.depthwise(x)

        # [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)

        x = self.norm(x)

        # [B, T, C] -> [B, T, rC]
        x = self.expand(x)

        x = self.activation(x)

        x = self.grn(x)

        # [B, T, rC] -> [B, T, C]
        x = self.contract(x)

        x = x * self.layer_scale

        # [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)

        if x.shape != res.shape:
            raise RuntimeError(f"ConvNeXtContextBlock1d forward dim mismatch: "
                               f"Input: {tuple(res.shape)}, Transformed {tuple(x.shape)}")

        return x + res


class ConvNeXtContextBlock2d(nn.Module):
    def __init__(
            self,
            channels: int,
            kernel_size: int = 7,
            dilation: int = 1,
            expansion_ratio: int = 2,
            layer_scale_initial: float = 1e-6,
            use_grn: bool = False,
            grn_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")

        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")

        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")

        if dilation <= 0:
            raise ValueError(f"dilation must be positive, got {dilation}")

        if expansion_ratio <= 0:
            raise ValueError(f"expansion_ratio must be positive, got {expansion_ratio}")

        if layer_scale_initial < 0.0:
            raise ValueError(
                "layer_scale_initial must be non-negative, "
                f"got {layer_scale_initial}"
            )

        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.expansion_ratio = expansion_ratio
        self.layer_scale_initial = layer_scale_initial

        padding = dilation * (kernel_size - 1) // 2

        self.depthwise = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            groups=channels,
        )

        self.norm = nn.LayerNorm(normalized_shape=channels)

        expanded_channels = channels * expansion_ratio

        self.expand = nn.Linear(
            in_features=channels,
            out_features=expanded_channels,
        )

        self.activation = nn.GELU()

        if use_grn:
            self.grn = GlobalResponseNorm2d(
                channels=expanded_channels,
                eps=grn_eps,
            )
        else:
            self.grn = nn.Identity()

        self.contract = nn.Linear(
            in_features=expanded_channels,
            out_features=channels
        )

        self.layer_scale = nn.Parameter(
            torch.full(
                (channels,),
                fill_value=layer_scale_initial,
                dtype=torch.float32,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise RuntimeError(f"ConvNeXtContextBlock2d only supports 4 dimensional tensors, "
                               f"got {tuple(x.shape)}")

        if x.shape[1] != self.channels:
            raise RuntimeError(f"ConvNeXtContextBlock2d channel mismatch: "
                               f"expected {self.channels}, got {x.shape[1]}")

        res = x

        x = self.depthwise(x)

        # [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1)

        x = self.norm(x)
        x = self.expand(x)
        x = self.activation(x)
        x = self.grn(x)
        x = self.contract(x)
        x = x * self.layer_scale

        # [B, H, W, C] -> [B, C, H, W]
        x = x.permute(0, 3, 1, 2)

        if x.shape != res.shape:
            raise RuntimeError(f"ConvNeXtContextBlock2d forward dim mismatch: "
                               f"Input: {tuple(res.shape)}, Transformed {tuple(x.shape)}")

        return res + x


def _expand_per_block_setting(
    value: int | Sequence[int],
    *,
    number_of_blocks: int,
    name: str,
) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * number_of_blocks

    values = tuple(value)

    if len(values) != number_of_blocks:
        raise ValueError(
            f"{name} must contain exactly "
            f"{number_of_blocks} values, got {len(values)}"
        )

    return values


def compute_balanced_channel_splits(
        channels: int,
        number_of_branches: int,
) -> tuple[int, ...]:
    """
    Divide channels evenly across branches.

    Examples:
        channels=16, branches=4:
        (4, 4, 4, 4)

        channels=18, branches=4:
        (5, 5, 4, 4)
    """
    if channels <= 0:
        raise ValueError(f"channels must be positive, got {channels}")

    if number_of_branches <= 0:
        raise ValueError(f"number_of_branches must be positive, got {number_of_branches}")

    if number_of_branches > channels:
        raise ValueError(f"number_of_branches must be <= channels, got {number_of_branches}")

    base_size = channels // number_of_branches
    remainder = channels % number_of_branches

    return tuple(
        base_size + (1 if index < remainder else 0) for index in range(number_of_branches)
    )


def expand_branch_setting(
        value: int | Sequence[int],
        *,
        number_of_branches: int,
        name: str,
) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * number_of_branches

    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an integer or sequence of integers)")

    values = tuple(value)

    if len(values) != number_of_branches:
        raise ValueError(
            f"{name} must contain exactly "
            f"{number_of_branches} values, got {len(values)}"
        )

    if not all(isinstance(item, int) for item in values):
        raise TypeError(f"{name} must contain only integers")

    return values


class MultiKernelConvNeXtContextBlock1d(nn.Module):
    """
    ConvNeXt style temporal context block with multiple depthwise kernel sizes operating on separate channel groups.

    Input & Output:
        [B, C, T]

    Processing:
        channel split
        -> depthwise temporal convolution per split
        -> concatenation
        -> channel-last LayerNorm
        -> pointwise expansion
        -> GELU
        -> pointwise contraction
        -> layer scale
        -> residual addition
    """

    def __init__(
            self,
            channels: int,
            kernel_sizes: int | Sequence[int] = (3, 7, 15, 31),
            dilations: int | Sequence[int] = 1,
            expansion_ratio: int = 2,
            layer_scale_initial: float = 1e-6,
            use_grn: bool = False,
            grn_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")

        kernel_sizes = tuple(kernel_sizes)

        if not kernel_sizes:
            raise ValueError(f"kernel_sizes must contain at least one value")

        if not all(isinstance(kernel_size, int) for kernel_size in kernel_sizes):
            raise TypeError(f"Every kernel size must be an integer")

        for kernel_size in kernel_sizes:
            if kernel_size <= 0 or kernel_size % 2 == 0:
                raise ValueError(f"kernel_size must be positive and odd, got {kernel_size}")

        if expansion_ratio <= 0:
            raise ValueError(f"expansion_ratio must be positive, got {expansion_ratio}")

        if layer_scale_initial < 0.0:
            raise ValueError(f"layer_scale_initial must be non-negative, got {layer_scale_initial}")

        number_of_branches = len(kernel_sizes)

        branch_dilations = expand_branch_setting(
            dilations,
            number_of_branches=number_of_branches,
            name="dilations",
        )

        for dilation in branch_dilations:
            if dilation <= 0:
                raise ValueError(f"Every dilation must be positive.")

        channel_splits = compute_balanced_channel_splits(
            channels,
            number_of_branches
        )

        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.channel_splits = channel_splits
        self.expansion_ratio = expansion_ratio

        self.depthwise_branches = nn.ModuleList()

        for branch_channels, kernel_size, dilation in zip(
                channel_splits,
                kernel_sizes,
                branch_dilations,
        ):
            padding = (dilation * (kernel_size - 1)) // 2

            self.depthwise_branches.append(
                nn.Conv1d(
                    in_channels=branch_channels,
                    out_channels=branch_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    groups=branch_channels,
                )
            )

        self.norm = nn.LayerNorm(channels)

        expanded_channels = channels * expansion_ratio

        self.expand = nn.Linear(channels, expanded_channels)

        self.activation = nn.GELU()

        if use_grn:
            self.grn = GlobalResponseNorm1d(
                channels=expanded_channels,
                eps=grn_eps,
            )
        else:
            self.grn = nn.Identity()

        self.contract = nn.Linear(expanded_channels, channels)

        self.layer_scale = nn.Parameter(
            torch.full(
                (channels,),
                fill_value=layer_scale_initial,
                dtype=torch.float32,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"MultiKernelConvNeXtContext.forward expects 3D tensor, got {tuple(x.shape)}")

        if x.shape[1] != self.channels:
            raise ValueError(f"Expected input to have shape {self.channels}, got {x.shape[1]}")

        residual = x

        channel_groups = torch.split(
            x,
            split_size_or_sections=list(self.channel_splits),
            dim=1,
        )

        if len(channel_groups) != len(self.depthwise_branches):
            raise RuntimeError(f"Channel splitting produced an unexpected number of branches, got {len(channel_groups)}")

        branch_outputs: list[torch.Tensor] = []

        for channel_group, depthwise_branch in zip(channel_groups, self.depthwise_branches):
            branch_outputs.append(depthwise_branch(channel_group))

        x = torch.cat(branch_outputs, dim=1)

        if x.shape != residual.shape:
            raise RuntimeError(f"MultiKernel temporal mixer changed shape: "
                               f"input={tuple(residual.shape)}, mixed={x.shape}")

        # [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)

        x = self.norm(x)
        x = self.expand(x)
        x = self.activation(x)
        x = self.grn(x)
        x = self.contract(x)

        x = x * self.layer_scale

        # [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)

        if x.shape != residual.shape:
            raise RuntimeError(f"MultiKernel residual branch changed shape: "
                               f"input={tuple(residual.shape)}, mixed={x.shape}")

        return residual + x


class ConvNeXtContextTrunk1d(nn.Module):
    def __init__(
        self,
        channels: int,
        number_of_blocks: int,
        block_type: str = "convnext",
        kernel_size: int = 7,
        dilation: int = 1,
        multi_kernel_sizes: Sequence[int] = (
            3,
            7,
            15,
            31,
        ),
        multi_kernel_dilations: Sequence[int] = (1, 1, 1, 1),
        expansion_ratio: int = 2,
        layer_scale_initial: float = 1e-6,
        attn_config: ContextAttentionConfig = ContextAttentionConfig(),
        use_grn: bool = False,
        grn_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(
                f"channels must be positive, got {channels}"
            )

        if number_of_blocks <= 0:
            raise ValueError(
                "number_of_blocks must be positive, "
                f"got {number_of_blocks}"
            )

        if attn_config.enabled:
            if not 0 <= attn_config.placement <= number_of_blocks:
                raise ValueError(
                    f"attn_config.placement must be between 0 and number_of_blocks, got placement={attn_config.placement}"
                    f" and number_of_blocks={number_of_blocks}"
                )

        self.channels = channels
        self.number_of_blocks = number_of_blocks
        self.block_type = block_type

        self.blocks = nn.ModuleList()

        for block_idx in range(number_of_blocks):
            if block_type == "convnext":

                block = ConvNeXtContextBlock1d(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    expansion_ratio=expansion_ratio,
                    layer_scale_initial=layer_scale_initial,
                    use_grn=use_grn,
                    grn_eps=grn_eps,
                )

            elif block_type == "multi_kernel_convnext":

                block = MultiKernelConvNeXtContextBlock1d(
                    channels=channels,
                    kernel_sizes=multi_kernel_sizes,
                    dilations=multi_kernel_dilations,
                    expansion_ratio=expansion_ratio,
                    layer_scale_initial=layer_scale_initial,
                    use_grn=use_grn,
                    grn_eps=grn_eps,
                )

            else:
                raise ValueError(f"Unsupported block type: {block_type!r}")

            self.blocks.append(block)

        self.attention_enabled = attn_config.enabled
        self.attention_placement = attn_config.placement

        if attn_config.enabled:
            self.attention = RoPETemporalSelfAttention1d(
                channels=channels,
                number_of_heads=attn_config.number_of_heads,
                dropout=attn_config.dropout,
                rope_base=attn_config.rope_base,
                layer_scale_initial=attn_config.layer_scale_initial,
                qkv_bias=attn_config.qkv_bias,
                output_bias=attn_config.output_bias,
            )
        else:
            self.attention = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        for block_idx, block in enumerate(self.blocks, start=0):

            if self.attention_enabled and self.attention_placement == block_idx:
                x = self.attention(x)

            x = block(x)

        if self.attention_placement == self.number_of_blocks:
            x = self.attention(x)

        return x

