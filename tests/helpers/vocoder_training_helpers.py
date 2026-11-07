from __future__ import annotations

from dataclasses import dataclass
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vocoder.src.nes_vocoder.config.schema import (
    FeatureMatchingLossConfig,
    MRSTFTConfig,
    TemporalGradientLossConfig,
    VocoderLossConfig, RMSEnvelopeLossConfig, BandPowerLossConfig,
)
from src.losses.vocoder_losses import (
    VocoderLossBundle,
)
from src.training.vocoder_step import (
    VocoderModels,
    VocoderOptimizers,
)


class TinyVocoderGenerator(nn.Module):
    """
    Minimal generator for training-step tests.

    Input:
        [B, cqt_bins, time]

    Output:
        [B, 1, time]
    """

    def __init__(
        self,
        cqt_bins: int,
        hidden_channels: int = 8,
    ) -> None:
        super().__init__()

        self.conv_in = nn.Conv1d(
            in_channels=cqt_bins,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
        )

        self.conv_out = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        cqt: torch.Tensor,
    ) -> torch.Tensor:
        if cqt.ndim == 4:
            cqt = cqt[:, 0]

        if cqt.ndim != 3:
            raise ValueError(
                "TinyVocoderGenerator expects [B,F,T], "
                f"got {tuple(cqt.shape)}."
            )

        x = self.conv_in(cqt)
        x = F.leaky_relu(
            x,
            negative_slope=0.1,
        )
        x = self.conv_out(x)

        return torch.tanh(x)


class TinyVocoderDiscriminator(nn.Module):
    """
    Minimal family-aware discriminator for training-step tests.

    Only the MPD family is represented. MSD and MRD are disabled.
    """

    def __init__(
        self,
        hidden_channels: int = 8,
    ) -> None:
        super().__init__()

        self.config = SimpleNamespace(
            mpd=SimpleNamespace(
                periods=(2,),
            ),
            msd=SimpleNamespace(
                num_scales=0,
            ),
            mrd=SimpleNamespace(
                resolutions=(),
            ),
            use_mpd=True,
            use_msd=False,
            use_mrd=False,
        )

        self.conv_feature = nn.Conv1d(
            in_channels=1,
            out_channels=hidden_channels,
            kernel_size=5,
            padding=2,
        )

        self.conv_output = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=3,
            padding=1,
        )

    def _forward_one(
        self,
        waveform: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
    ]:
        if waveform.ndim != 3:
            raise ValueError(
                "TinyVocoderDiscriminator expects [B,1,T], "
                f"got {tuple(waveform.shape)}."
            )

        feature = self.conv_feature(waveform)
        feature = F.leaky_relu(
            feature,
            negative_slope=0.2,
        )

        output_map = self.conv_output(feature)

        prediction = torch.flatten(
            output_map,
            start_dim=1,
        )

        feature_maps = [
            feature,
            output_map,
        ]

        return prediction, feature_maps

    def forward(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> dict[str, list]:
        real_prediction, real_maps = (
            self._forward_one(real)
        )

        fake_prediction, fake_maps = (
            self._forward_one(fake)
        )

        real_outputs = [real_prediction]
        fake_outputs = [fake_prediction]

        real_feature_maps = [real_maps]
        fake_feature_maps = [fake_maps]

        return {
            "real_outputs": real_outputs,
            "fake_outputs": fake_outputs,
            "real_feature_maps": real_feature_maps,
            "fake_feature_maps": fake_feature_maps,

            "mpd_real_outputs": real_outputs,
            "mpd_fake_outputs": fake_outputs,
            "mpd_real_feature_maps": (
                real_feature_maps
            ),
            "mpd_fake_feature_maps": (
                fake_feature_maps
            ),

            "msd_real_outputs": [],
            "msd_fake_outputs": [],
            "msd_real_feature_maps": [],
            "msd_fake_feature_maps": [],

            "mrd_real_outputs": [],
            "mrd_fake_outputs": [],
            "mrd_real_feature_maps": [],
            "mrd_fake_feature_maps": [],
        }


@dataclass
class VocoderTrainingFixture:
    batch: dict[str, torch.Tensor]

    models: VocoderModels
    optimizers: VocoderOptimizers
    loss_bundle: VocoderLossBundle

    device: torch.device


def set_test_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_training_test_loss_config(
        *,
        feature_matching_normalization: str = "layer_mean",
) -> VocoderLossConfig:
    return VocoderLossConfig(
        lambda_adversarial=1.0,
        lambda_feature_matching=1.0,
        lambda_mrstft=1.0,
        lambda_waveform=0.5,
        lambda_temporal_gradient=0.1,
        lambda_rms_envelope=0.1,
        lambda_band_power=1.0,

        feature_matching=FeatureMatchingLossConfig(
            normalization=feature_matching_normalization,
            scale_floor=1e-2,
            eps=1e-8,
        ),

        mrstft=MRSTFTConfig(
            fft_sizes=(16, 32),
            hop_sizes=(4, 8),
            win_lengths=(16, 32),
            spectral_convergence_weight=1.0,
            log_magnitude_weight=1.0,
            eps=1e-7,
        ),

        temporal_gradient=TemporalGradientLossConfig(
            enabled=True,
            lags=(1, 2, 4),
            normalization="none",
            scale_floor=1e-3,
            eps=1e-8,
            edge_weighting=False,
            edge_weight=1.0,
            max_edge_weight=5.0,
        ),

        rms_envelope=RMSEnvelopeLossConfig(),
        band_power=BandPowerLossConfig(
            enabled=True,
            fft_sizes=(32, 64),
            hop_sizes=(8, 16),
            win_lengths=(32, 64),
            number_of_bands=6,
            fmin=50.0,
            fmax=4000.0,
            absolute_weight=1.0,
            shape_weight=1.0,
            eps=1e-8,
        )
    )


def make_training_fixture(
    *,
    seed: int = 1337,
    batch_size: int = 2,
    cqt_bins: int = 4,
    sequence_length: int = 128,
    learning_rate_generator: float = 1e-3,
    learning_rate_discriminator: float = 1e-3,
    feature_matching_normalization: str = "layer_mean",
    device: torch.device | str = "cpu",
    sample_rate: int = 8000,
) -> VocoderTrainingFixture:
    set_test_seed(seed)

    device = torch.device(device)

    generator = TinyVocoderGenerator(
        cqt_bins=cqt_bins,
        hidden_channels=8,
    ).to(device)

    discriminator = TinyVocoderDiscriminator(
        hidden_channels=8,
    ).to(device)

    models = VocoderModels(
        generator=generator,
        discriminator=discriminator,
    )

    optimizers = VocoderOptimizers(
        generator=torch.optim.AdamW(
            generator.parameters(),
            lr=learning_rate_generator,
            betas=(0.8, 0.99),
            weight_decay=0.0,
        ),
        discriminator=torch.optim.AdamW(
            discriminator.parameters(),
            lr=learning_rate_discriminator,
            betas=(0.8, 0.99),
            weight_decay=0.0,
        ),
    )

    loss_config = (
        make_training_test_loss_config(feature_matching_normalization=feature_matching_normalization)
    )

    loss_bundle = VocoderLossBundle(
        loss_config,
        sample_rate=sample_rate
    ).to(device)

    cqt = torch.randn(
        batch_size,
        cqt_bins,
        sequence_length,
        device=device,
    )

    audio = torch.randn(
        batch_size,
        1,
        sequence_length,
        device=device,
    )

    # Keep the target within the generator's tanh output range.
    audio = torch.tanh(audio)

    batch = {
        "cqt": cqt,
        "audio": audio,
    }

    return VocoderTrainingFixture(
        batch=batch,
        models=models,
        optimizers=optimizers,
        loss_bundle=loss_bundle,
        device=device,
    )


def clone_parameter_state(
    module: nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter
        in module.named_parameters()
    }


def assert_module_parameters_equal(
    first: nn.Module,
    second: nn.Module,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    first_parameters = dict(
        first.named_parameters()
    )

    second_parameters = dict(
        second.named_parameters()
    )

    assert first_parameters.keys() == (
        second_parameters.keys()
    )

    for name in first_parameters:
        assert torch.allclose(
            first_parameters[name],
            second_parameters[name],
            atol=atol,
            rtol=rtol,
        ), (
            f"Parameter {name!r} differs between modules."
        )


def assert_module_parameters_changed(
    module: nn.Module,
    before: dict[str, torch.Tensor],
) -> None:
    after = dict(
        module.named_parameters()
    )

    assert before.keys() == after.keys()

    changed = [
        name
        for name, parameter in after.items()
        if not torch.equal(
            before[name],
            parameter.detach(),
        )
    ]

    assert changed, (
        "No trainable parameters changed."
    )


