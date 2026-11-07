from __future__ import annotations
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as apply_spectral_norm


NormType = Literal["none", "batch", "instance", "group"]
ActivationType = Literal["none", "relu", "leaky_relu", "gelu", "silu", "tanh"]
PaddingMode = Literal["reflect", "replicate", "zeros", "cqt"]
InitType = Literal["normal", "xavier", "kaiming", "orthogonal"]


def get_activation(
        name: ActivationType = "relu",
        inplace: bool = True,
        negative_slope: float = 0.2,
) -> nn.Module:
    if name == "none":
        return nn.Identity()
    elif name == "relu":
        return nn.ReLU(inplace=inplace)
    elif name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
    elif name == "gelu":
        return nn.GELU()
    elif name == "silu":
        return nn.SiLU(inplace=inplace)
    elif name == "tanh":
        return nn.Tanh()

    raise ValueError(f"Unknown activation function: {name}")


def get_norm_layer(
        channels: int,
        norm: NormType = "instance",
        num_groups: int = 32,
        affine: bool = True,
) -> nn.Module:
    if norm == "none":
        return nn.Identity()
    if norm == "batch":
        return nn.BatchNorm2d(channels, affine=affine)
    if norm == "instance":
        return nn.InstanceNorm2d(channels,
                                 affine=affine,
                                 track_running_stats=False)
    if norm == "group":
        groups = min(num_groups, channels)

        while channels % groups != 0 and groups > 1:
            groups -= 1

        return nn.GroupNorm(num_groups=groups,
                            num_channels=channels,
                            affine=affine)

    raise ValueError(f"Unknown norm function: {norm}")


class CQTBufferPad(nn.Module):
    def __init__(
            self,
            pad_time_left: int,
            pad_time_right: int,
            pad_freq_low: int,
            pad_freq_high: int,
            freq_pad_value: float = 0.0,
    ) -> None:
        super().__init__()

        self.time_pad = (pad_time_left, pad_time_right, 0, 0)
        self.freq_pad = (0, 0, pad_freq_low, pad_freq_high)
        self.freq_pad_value = freq_pad_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.time_pad, mode="reflect")
        x = F.pad(x, self.freq_pad, mode="constant", value=self.freq_pad_value)
        return x


def get_padding_layer(
        padding: int | tuple[int, int, int, int],
        padding_mode: PaddingMode = "reflect",
) -> nn.Module:
    if padding == 0:
        return nn.Identity()

    if padding_mode == "reflect":
        return nn.ReflectionPad2d(padding)

    if padding_mode == "replicate":
        return nn.ReplicationPad2d(padding)

    if padding_mode == "zeros":
        return nn.ZeroPad2d(padding)

    if padding_mode == "cqt":
        if isinstance(padding, tuple):
            left, right, freq_low, freq_high = padding
            return CQTBufferPad(left, right, freq_low, freq_high)
        if isinstance(padding, int):
            return CQTBufferPad(padding, padding, padding, padding)

    raise ValueError(f"Unknown padding mode: {padding_mode}")


def maybe_spectral_norm(
        layer: nn.Module,
        use_spectral_norm: bool = False,
) -> nn.Module:
    if use_spectral_norm:
        return apply_spectral_norm(layer)
    return layer


def init_weights(
        module: nn.Module,
        init_type: InitType = "normal",
        gain: float = 0.02,
) -> None:
    """
    Initialize Conv/Linear layers in the style commonly used by GANs.

    Call with:
    model.apply(lambda m: init_weights(m, "normal", 0.02))
    """

    classname = module.__class__.__name__

    if hasattr(module, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
        if init_type == "normal":
            nn.init.normal_(module.weight.data, mean=0.0, std=gain)
        elif init_type == "xavier":
            nn.init.xavier_normal_(module.weight.data, gain=gain)
        elif init_type == "kaiming":
            nn.init.kaiming_normal_(module.weight.data, a=0, mode="fan_in")
        elif init_type == "orthogonal":
            nn.init.orthogonal_(module.weight.data, gain=gain)
        else:
            raise ValueError(f"Unknown init_type: {init_type}")

        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)

    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        if getattr(module, "weight", None) is not None:
            nn.init.normal_(module.weight.data, mean=1.0, std=gain)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())

