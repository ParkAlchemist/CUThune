from __future__ import annotations

from typing import TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import weight_norm, spectral_norm


from vocoder.src.nes_vocoder.config.schema import (
    NormType,
    PeriodDiscriminatorConfig,
    MultiPeriodDiscriminatorConfig,
    ScaleDiscriminatorConfig,
    MultiScaleDiscriminatorConfig,
    ResolutionDiscriminatorConfig,
    MultiResolutionDiscriminatorConfig,
    VocoderDiscriminatorConfig,
)


DiscriminatorFamilyOutput: TypeAlias = tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[list[torch.Tensor]],
    list[list[torch.Tensor]],
]


def apply_norm(module: nn.Module, norm: NormType) -> nn.Module:
    if norm == "weight":
        return weight_norm(module)
    if norm == "spectral":
        return spectral_norm(module)
    if norm == "none":
        return module

    raise ValueError(f"Unknown norm type: {norm}")


def try_remove_parameter_norm(module: nn.Module) -> None:
    if not parametrize.is_parametrized(module, "weight"):
        return

    parametrize.remove_parametrizations(
        module,
        "weight",
        leave_parametrized=True,
    )


class PeriodDiscriminator(nn.Module):
    """
    HiFi-GAN-style period discriminator.
    It reshapes a waveform from:
        [B, 1, T]
    into:
        [B, C, T // period, period]
    """

    def __init__(self, config: PeriodDiscriminatorConfig) -> None:
        super().__init__()
        self.config = config
        self.period = int(config.period)

        channels = config.channels

        convs: list[nn.Module] = []

        in_channels = 1

        for i, out_channels in enumerate(channels):
            stride = (config.stride, 1) if i < len(channels) - 1 else (1, 1)

            convs.append(
                apply_norm(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=(config.kernel_size, 1),
                        stride=stride,
                        padding=((config.kernel_size - 1) // 2, 0),
                    ),
                    config.norm,
                )
            )

            in_channels = out_channels

        self.convs = nn.ModuleList(convs)

        self.conv_post = apply_norm(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=(3, 1),
                stride=(1, 1),
                padding=(1, 0),
            ),
            config.norm,
        )

    def _reshape_by_period(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, time = x.shape

        if time % self.period != 0:
            pad_length = self.period - (time % self.period)
            x = F.pad(x, (0, pad_length), mode="reflect")
            time = time + pad_length

        x = x.view(batch, channels, time // self.period, self.period)
        return x


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if x.ndim != 3:
            raise ValueError(f"Expected waveform [B, 1, T], got {tuple(x.shape)}]")

        feature_maps: list[torch.Tensor] = []

        x = self._reshape_by_period(x)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, negative_slope=self.config.negative_slope)
            feature_maps.append(x)

        x = self.conv_post(x)
        feature_maps.append(x)

        # Flatten patch predictions to [B, N]
        prediction = torch.flatten(x, start_dim=1)

        return prediction, feature_maps

    def remove_norm(self) -> None:
        for conv in self.convs:
            try_remove_parameter_norm(conv)

        try_remove_parameter_norm(self.conv_post)


class MultiPeriodDiscriminator(nn.Module):
    """
    Ensemble of period discriminators.
    Default periods follow the common HiFi-GAN MPD setup:
        [2, 3, 5, 7, 11]
    """

    def __init__(self, config: MultiPeriodDiscriminatorConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = MultiPeriodDiscriminatorConfig()
        self.config = config

        self.discriminators = nn.ModuleList(
            [
                PeriodDiscriminator(
                    PeriodDiscriminatorConfig(
                        period=period,
                        channels=config.channels,
                        norm=config.norm,
                        kernel_size=config.kernel_size,
                        stride=config.stride,
                        negative_slope=config.negative_slope,
                    )
                )
                for period in self.config.periods
            ]
        )

    def forward(
            self,
            real: torch.Tensor,
            fake: torch.Tensor,
    ) -> DiscriminatorFamilyOutput:
        real_outputs: list[torch.Tensor] = []
        fake_outputs: list[torch.Tensor] = []
        real_feature_maps: list[list[torch.Tensor]] = []
        fake_feature_maps: list[list[torch.Tensor]] = []

        for discriminator in self.discriminators:
            real_pred, real_fmap = discriminator(real)
            fake_pred, fake_fmap = discriminator(fake)

            real_outputs.append(real_pred)
            fake_outputs.append(fake_pred)
            real_feature_maps.append(real_fmap)
            fake_feature_maps.append(fake_fmap)

        return real_outputs, fake_outputs, real_feature_maps, fake_feature_maps

    def remove_norm(self) -> None:
        for discriminator in self.discriminators:
            discriminator.remove_norm()


class ScaleDiscriminator(nn.Module):
    """
    HiFi-GAN-style scale discriminator.
    It operates directly on waveform tensors:
        [B, 1, T]
    """

    def __init__(self, config: ScaleDiscriminatorConfig) -> None:
        super().__init__()
        self.config = config

        if not (
            len(config.channels)
            == len(config.kernel_sizes)
            == len(config.strides)
            == len(config.groups)
        ):
            raise ValueError("channels/kernel_sizes/strides/groups must have equal length")

        convs: list[nn.Module] = []

        in_channels = 1

        for out_channels, kernel_size, stride, groups, in zip(
            config.channels,
            config.kernel_sizes,
            config.strides,
            config.groups,
        ):

            if in_channels % groups != 0:
                raise ValueError(f"in_channels={in_channels} must be divisible by groups={groups}")

            if out_channels % groups != 0:
                raise ValueError(f"out_channels={out_channels} must be divisible by groups={groups}")

            convs.append(
                apply_norm(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=(kernel_size - 1) // 2,
                        groups=groups,
                    ),
                    config.norm,
                )
            )

            in_channels = out_channels

        self.convs = nn.ModuleList(convs)

        self.conv_post = apply_norm(
            nn.Conv1d(
                in_channels,
                1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            config.norm,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if x.ndim != 3:
            raise ValueError(f"Expected waveform [B, 1, T], got {tuple(x.shape)}]")

        feature_maps: list[torch.Tensor] = []

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, negative_slope=self.config.negative_slope)
            feature_maps.append(x)

        x = self.conv_post(x)
        feature_maps.append(x)

        prediction = torch.flatten(x, start_dim=1)

        return prediction, feature_maps

    def remove_norm(self) -> None:
        for conv in self.convs:
            try_remove_parameter_norm(conv)

        try_remove_parameter_norm(self.conv_post)


class MultiScaleDiscriminator(nn.Module):
    """
    Ensemble of scale discriminators.

    Scale 0:
        raw waveform
    Scale 1:
        pooled waveform
    Scale 2:
        pooled again
    """

    def __init__(self, config: MultiScaleDiscriminatorConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = MultiScaleDiscriminatorConfig()
        self.config = config

        discriminators: list[nn.Module] = []

        for scale_idx in range(config.num_scales):
            norm = (
                config.first_discriminator_norm
                if scale_idx == 0
                else config.other_discriminator_norm
            )

            scale_config = config.discriminator

            discriminators.append(
                ScaleDiscriminator(
                    ScaleDiscriminatorConfig(
                        channels=scale_config.channels,
                        kernel_sizes=scale_config.kernel_sizes,
                        strides=scale_config.strides,
                        groups=scale_config.groups,
                        norm=norm,
                        negative_slope=scale_config.negative_slope,
                    ),
                )
            )

        self.discriminators = nn.ModuleList(discriminators)

        self.pooling = nn.AvgPool1d(
            kernel_size=config.pool_kernel_size,
            stride=config.pool_stride,
            padding=config.pool_padding,
            count_include_pad=False,
        )

    def forward(
            self,
            real: torch.Tensor,
            fake: torch.Tensor,
    ) -> DiscriminatorFamilyOutput:
        real_outputs: list[torch.Tensor] = []
        fake_outputs: list[torch.Tensor] = []
        real_feature_maps: list[list[torch.Tensor]] = []
        fake_feature_maps: list[list[torch.Tensor]] = []

        real_current = real
        fake_current = fake

        for scale_idx, discriminator in enumerate(self.discriminators):
            if scale_idx > 0:
                real_current = self.pooling(real_current)
                fake_current = self.pooling(fake_current)

            real_pred, real_fmap = discriminator(real_current)
            fake_pred, fake_fmap = discriminator(fake_current)

            real_outputs.append(real_pred)
            fake_outputs.append(fake_pred)
            real_feature_maps.append(real_fmap)
            fake_feature_maps.append(fake_fmap)

        return real_outputs, fake_outputs, real_feature_maps, fake_feature_maps

    def remove_norm(self) -> None:
        for discriminator in self.discriminators:
            discriminator.remove_norm()


class ResolutionDiscriminator(nn.Module):
    """
    Spectrogram discriminator operating at one STFT resolution.

    Input:
        waveform [B, 1, T]

    Internal Representation:
        compressed magnitude spectrogram [B, 1, F, frames]

    Output:
        flattened patch logits [B, N]
        intermediate feature maps
    """
    def __init__(
            self,
            *,
            fft_size: int,
            hop_size: int,
            win_length: int,
            config: ResolutionDiscriminatorConfig,
    ) -> None:
        super().__init__()

        if fft_size <= 0:
            raise ValueError(f"Expected fft_size > 0, got {fft_size}")

        if hop_size <= 0:
            raise ValueError(f"Expected hop_size > 0, got {hop_size}")

        if win_length <= 0:
            raise ValueError(f"Expected win_length > 0, got {win_length}")

        if win_length > fft_size:
            raise ValueError(f"Expected win_length > fft_size, got win_length={win_length} and fft_size={fft_size}")

        if not len(config.channels) == len(config.kernel_sizes) == len(config.strides):
            raise ValueError(f"MRD channels, kernel_sizes and strides are not equal")

        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.config = config

        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

        convolutions: list[nn.Module] = []
        in_channels = 1

        for out_channels, kernel_size, stride in zip(config.channels, config.kernel_sizes, config.strides):
            if len(kernel_size) != 2:
                raise ValueError(f"Each MRD kernel must contain exactly 2 elements, got {len(kernel_size)}")

            if len(stride) != 2:
                raise ValueError(f"Each MRD stride must contain exactly 2 elements, got {len(stride)}")

            kernel_frequency, kernel_time = kernel_size

            if kernel_frequency <= 0 or kernel_time <= 0:
                raise ValueError(f"MRD kernel sizes must be positive, got {kernel_frequency, kernel_time}")

            if kernel_frequency % 2 == 0:
                raise ValueError(f"MRD frequency kernel must be odd, got {kernel_frequency}")

            if kernel_time % 2 == 0:
                raise ValueError(f"MRD time kernel must be odd, got {kernel_time}")

            padding = (
                (kernel_frequency - 1) // 2,
                (kernel_time - 1) // 2,
            )

            convolutions.append(
                apply_norm(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    config.norm,
                )
            )

            in_channels = out_channels

        self.convs = nn.ModuleList(convolutions)

        self.conv_post = apply_norm(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            config.norm,
        )

    def _spectrogram(
            self,
            waveform: torch.Tensor,
    ) -> torch.Tensor:
        if waveform.ndim != 3:
            raise ValueError(f"ResolutionDiscriminator expects waveform [B, 1, T], got {tuple(waveform.shape)}]")

        if waveform.shape[1] != 1:
            raise ValueError(f"ResolutionDiscriminator currently supports mono waveform input only, "
                             f"got {waveform.shape[1]} channels")

        waveform = waveform[:, 0, :]

        original_dtype = waveform.dtype

        with torch.amp.autocast(device_type=waveform.device.type, enabled=False):

            waveform_fp32 = waveform.float()

            window = self.window.to(device=waveform.device, dtype=torch.float32)

            spectrum = torch.stft(
                waveform_fp32,
                n_fft=self.fft_size,
                hop_length=self.hop_size,
                win_length=self.win_length,
                window=window,
                center=True,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )

            magnitude = spectrum.abs()

            compression = self.config.magnitude_compression

            if compression == "log1p":
                magnitude = torch.log1p(magnitude)
            elif compression == "log":
                magnitude = torch.log(magnitude.clamp_min(self.config.eps))
            elif compression == "none":
                pass
            else:
                raise ValueError(f"Unknown compression type: {compression}")

        return magnitude.unsqueeze(1)

    def forward(
            self,
            waveform: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self._spectrogram(waveform)

        feature_maps: list[torch.Tensor] = []

        for convolution in self.convs:
            x = convolution(x)
            x = F.leaky_relu(x, negative_slope=self.config.negative_slope)
            feature_maps.append(x)

        x = self.conv_post(x)
        feature_maps.append(x)

        prediction = torch.flatten(x, start_dim=1)

        return prediction, feature_maps

    def remove_norm(self) -> None:
        for conv in self.convs:
            try_remove_parameter_norm(conv)

        try_remove_parameter_norm(self.conv_post)


class MultiResolutionDiscriminator(nn.Module):
    """
    Ensemble of spectrogram discriminators using separate STFT resolutions.
    """
    def __init__(
            self,
            config: MultiResolutionDiscriminatorConfig | None = None
    ):
        super().__init__()

        if config is None:
            config = MultiResolutionDiscriminatorConfig()

        self.config = config
        self.discriminators = nn.ModuleList()

        if not config.resolutions:
            raise ValueError(f"ResolutionDiscriminator expects at least one resolution")

        for resolution in config.resolutions:
            if len(resolution) != 3:
                raise ValueError(f"ResolutionDiscriminator expects (fft_size, hop_size, win_length), got {resolution}")

            fft_size, hop_size, win_length = resolution

            self.discriminators.append(
                ResolutionDiscriminator(
                    fft_size=fft_size,
                    hop_size=hop_size,
                    win_length=win_length,
                    config=config.discriminator,
                )
            )

    def forward(
            self,
            real: torch.Tensor,
            fake: torch.Tensor,
    ) -> DiscriminatorFamilyOutput:
        real_outputs: list[torch.Tensor] = []
        fake_outputs: list[torch.Tensor] = []

        real_feature_maps: list[list[torch.Tensor]] = []
        fake_feature_maps: list[list[torch.Tensor]] = []

        for discriminator in self.discriminators:
            real_prediction, real_maps = discriminator(real)
            fake_prediction, fake_maps = discriminator(fake)

            real_outputs.append(real_prediction)
            fake_outputs.append(fake_prediction)

            real_feature_maps.append(real_maps)
            fake_feature_maps.append(fake_maps)

        return (
            real_outputs,
            fake_outputs,
            real_feature_maps,
            fake_feature_maps,
        )

    def remove_norm(self) -> None:
        for discriminator in self.discriminators:
            discriminator.remove_norm()


class HiFiGANMultiDiscriminator(nn.Module):
    """
    Combined MPD + MSD + MRD wrapper.

    This object returns all predictions and feature maps needed for:
        - discriminator adversarial loss
        - generator adverarial loss
        - feature matching loss
    """

    def __init__(self, config: VocoderDiscriminatorConfig | None = None) -> None:
        super().__init__()

        if config is None:
            config = VocoderDiscriminatorConfig()

        self.config = config

        self.mpd = MultiPeriodDiscriminator(config.mpd) if config.use_mpd else None
        self.msd = MultiScaleDiscriminator(config.msd) if config.use_msd else None
        self.mrd = MultiResolutionDiscriminator(config.mrd) if config.use_mrd else None

        if all(discriminator is None for discriminator in (self.mpd, self.msd, self.mrd)):
            raise ValueError(f"At least one discriminator is required")

    def forward(self, real: torch.Tensor, fake: torch.Tensor) -> dict[str, list]:
        combined_real_outputs: list[torch.Tensor] = []
        combined_fake_outputs: list[torch.Tensor] = []

        combined_real_maps: list[list[torch.Tensor]] = []
        combined_fake_maps: list[list[torch.Tensor]] = []

        family_results: dict[str, list] = {}

        families = (
            ("mpd", self.mpd),
            ("msd", self.msd),
            ("mrd", self.mrd),
        )

        for family_name, discriminator in families:
            if discriminator is None:
                real_outputs: list[torch.Tensor] = []
                fake_outputs: list[torch.Tensor] = []
                real_maps: list[list[torch.Tensor]] = []
                fake_maps: list[list[torch.Tensor]] = []
            else:
                (
                    real_outputs,
                    fake_outputs,
                    real_maps,
                    fake_maps,
                ) = discriminator(real, fake)

            family_results[f"{family_name}_real_outputs"] = real_outputs
            family_results[f"{family_name}_fake_outputs"] = fake_outputs
            family_results[f"{family_name}_real_feature_maps"] = real_maps
            family_results[f"{family_name}_fake_feature_maps"] = fake_maps

            combined_real_outputs.extend(real_outputs)
            combined_fake_outputs.extend(fake_outputs)

            combined_real_maps.extend(real_maps)
            combined_fake_maps.extend(fake_maps)


        return {
            "real_outputs": combined_real_outputs,
            "fake_outputs": combined_fake_outputs,
            "real_feature_maps": combined_real_maps,
            "fake_feature_maps": combined_fake_maps,
            **family_results,
        }

    def remove_norm(self) -> None:
        for discriminator in (self.mpd, self.msd, self.mrd):
            if discriminator is not None:
                discriminator.remove_norm()


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    return sum(p.numel() for p in module.parameters())

