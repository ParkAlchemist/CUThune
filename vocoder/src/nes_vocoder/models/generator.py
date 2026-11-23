from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import (
    is_parametrized,
    remove_parametrizations,
)

from shared.src.chiptune_shared.nn.blocks.channel_attention import ECABlock1d
from shared.src.chiptune_shared.nn.blocks.convnext import ConvNeXtContextTrunk1d
from vocoder.src.nes_vocoder.config.schema import (
    VocoderGeneratorModelConfig,
    ActivationType, SnakeBetaConfig, AliasFreeConfig,
)

from vocoder.src.nes_vocoder.models.alias_free import (
    AliasFreeActivation1d,
)


@dataclass(frozen=True)
class RMSEnvelopeEmbeddingOutput:
    activity_mean: torch.Tensor
    silence_fraction: torch.Tensor

    embedding_mean: torch.Tensor
    embedding_rms: torch.Tensor
    embedding_max_abs: torch.Tensor

    to_cqt_ratio: torch.Tensor

    layer_scale_mean_abs: torch.Tensor
    layer_scale_max_abs: torch.Tensor


@dataclass(frozen=True)
class GeneratorForwardOutput:
    audio: torch.Tensor
    pre_tanh: torch.Tensor
    rms_embedding: RMSEnvelopeEmbeddingOutput | None = None


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


class SnakeBeta(nn.Module):
    """
    Channel-wise SnakeBeta activation for [B, C, T] tensors.

    Formula:
        y = x + sin(alpha * x)^2 / beta

    When alpha_logscale=True, the learnable parameters store:

        alpha_parameter = log(alpha_effective)
        beta_parameter = log(beta_effective)

    and the effective positive values are recovered with exp().
    """

    def __init__(
        self,
        channels: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        alpha_logscale: bool = True,
        eps: float = 1e-9,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(
                f"channels must be positive, got {channels}."
            )

        if alpha <= 0.0:
            raise ValueError(
                f"alpha must be positive, got {alpha}."
            )

        if beta <= 0.0:
            raise ValueError(
                f"beta must be positive, got {beta}."
            )

        if eps <= 0.0:
            raise ValueError(
                f"eps must be positive, got {eps}."
            )

        self.channels = channels
        self.alpha_logscale = alpha_logscale
        self.eps = eps

        if alpha_logscale:
            alpha_parameter = math.log(alpha)
            beta_parameter = math.log(beta)
        else:
            alpha_parameter = alpha
            beta_parameter = beta

        self.alpha = nn.Parameter(
            torch.full(
                (1, channels, 1),
                fill_value=alpha_parameter,
                dtype=torch.float32,
            )
        )

        self.beta = nn.Parameter(
            torch.full(
                (1, channels, 1),
                fill_value=beta_parameter,
                dtype=torch.float32,
            )
        )

    def effective_parameters(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the positive values used by the activation.
        """
        if self.alpha_logscale:
            alpha = torch.exp(self.alpha)
            beta = torch.exp(self.beta)
        else:
            alpha = self.alpha
            beta = self.beta.abs()

        return alpha, beta

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "SnakeBeta expects a [B,C,T] tensor, "
                f"got shape {tuple(x.shape)}."
            )

        if x.shape[1] != self.channels:
            raise ValueError(
                "SnakeBeta channel mismatch: "
                f"configured for {self.channels} channels, "
                f"received {x.shape[1]}."
            )

        orig_dtype = x.dtype
        x = x.float()

        alpha, beta = self.effective_parameters()

        sin_term = torch.sin(alpha * x)

        return (x + sin_term.square() / (beta + self.eps)).to(orig_dtype)


def get_activation(
    activation: ActivationType,
    channels: int,
    snake_beta_config: SnakeBetaConfig,
    alias_free_config: AliasFreeConfig,
) -> nn.Module:
    if activation == "leaky_relu":
        return nn.LeakyReLU(0.1)

    elif activation == "snake_beta":

        module = SnakeBeta(
            channels=channels,
            alpha=snake_beta_config.alpha_initial,
            beta=snake_beta_config.beta_initial,
            alpha_logscale=snake_beta_config.alpha_logscale,
        )

        if alias_free_config.enabled:
            return AliasFreeActivation1d(
                activation=module,
                upsample_ratio=alias_free_config.upsample_ratio,
                downsample_ratio=alias_free_config.downsample_ratio,
                upsample_kernel_size=alias_free_config.upsample_kernel_size,
                downsample_kernel_size=alias_free_config.downsample_kernel_size,
            )
        return module

    else:
        raise ValueError(f"Unknown activation: {activation}")


def remove_parametrized_weight_norm(module: nn.Module,) -> None:
    if is_parametrized(module,"weight"):
        remove_parametrizations(
            module,
            "weight",
            leave_parametrized=True,
        )


class ResBlock1D(nn.Module):
    """
    HiFi-GAN-style residual block.

    Each dilation stage:
        activation -> dilated Conv1d -> activation -> Conv1d -> residual add
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: tuple[int, ...],
        activation: ActivationType = "leaky_relu",
        snake_beta_config: SnakeBetaConfig = SnakeBetaConfig(),
        alias_free_config: AliasFreeConfig = AliasFreeConfig(),
    ) -> None:
        super().__init__()

        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.acts1 = nn.ModuleList()
        self.acts2 = nn.ModuleList()

        for dilation in dilations:
            self.acts1.append(get_activation(activation, channels, snake_beta_config, alias_free_config))
            self.convs1.append(
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=dilation,
                        padding=get_padding(kernel_size, dilation),
                    )
                )
            )

            self.acts2.append(get_activation(activation, channels, snake_beta_config, alias_free_config))
            self.convs2.append(
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for act1, conv1, act2, conv2 in zip(
            self.acts1,
            self.convs1,
            self.acts2,
            self.convs2,
        ):
            residual = x
            y = conv1(act1(x))
            y = conv2(act2(y))
            x = y + residual

        return x

    def remove_weight_norm(self) -> None:
        for conv in self.convs1:
            remove_parametrized_weight_norm(conv)

        for conv in self.convs2:
            remove_parametrized_weight_norm(conv)


class RMSEnvelopeEmbedding(nn.Module):
    def __init__(
        self,
        *,
        output_channels: int,
        hidden_channels: int = 64,
        kernel_size: int = 7,
        layer_scale_initial: float = 1e-3,
        remap_to_activity: bool = True,
    ) -> None:
        super().__init__()

        if output_channels <= 0:
            raise ValueError(
                "output_channels must be positive."
            )

        if hidden_channels <= 0:
            raise ValueError(
                "hidden_channels must be positive."
            )

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be positive "
                "and odd."
            )

        if layer_scale_initial < 0.0:
            raise ValueError(
                "layer_scale_initial must be "
                "non-negative."
            )

        self.output_channels = output_channels

        self.remap_to_activity = remap_to_activity

        self.input_projection = nn.Conv1d(
            in_channels=1,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

        self.activation = nn.SiLU()

        self.output_projection = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=output_channels,
            kernel_size=1,
            bias=False,
        )

        self.layer_scale = nn.Parameter(
            torch.full(
                (output_channels,),
                fill_value=layer_scale_initial,
                dtype=torch.float32,
            )
        )

        self.reset_parameters()

    def zero_biases(self) -> None:
        nn.init.zeros_(self.input_projection.bias)
        nn.init.zeros_(self.output_projection.bias)

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.input_projection.weight,
            mean=0.0,
            std=0.01,
        )

        nn.init.normal_(
            self.output_projection.weight,
            mean=0.0,
            std=0.01,
        )

    def forward(
        self,
        envelope: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if envelope.ndim != 3 or envelope.shape[1] != 1:
            raise ValueError(
                "Expected RMS envelope [B,1,T], "
                f"got {tuple(envelope.shape)}."
            )

        if self.remap_to_activity:
            activity = (envelope + 1.0) * 0.5
            activity = activity.clamp(min=0.0, max=1.0)
        else:
            activity = envelope

        embedding = self.input_projection(activity)
        embedding = self.activation(embedding)
        embedding = self.output_projection(embedding)
        embedding = embedding * self.layer_scale[None, :, None]

        return embedding, activity


class CQTUHiFiGANGenerator(nn.Module):
    """
    Lightweight CQT-conditioned HiFi-GAN-style generator.

    Input:
        [B, Bins, T]
        or [B, 1, Bins, T]

    Output:
        [B, 1, T * Hop]
    """

    def __init__(
            self,
            cqt_bins: int,
            model_config: VocoderGeneratorModelConfig,
    ) -> None:
        super().__init__()
        self.config = model_config

        if len(model_config.upsample_rates) != len(model_config.upsample_kernel_sizes):
            raise ValueError("upsample_rates and upsample_kernel_sizes must match.")

        if len(model_config.resblock_kernel_sizes) != len(model_config.resblock_dilation_sizes):
            raise ValueError(
                "resblock_kernel_sizes and resblock_dilation_sizes must match."
            )

        self.conv_pre = weight_norm(
            nn.Conv1d(
                cqt_bins,
                model_config.upsample_initial_channel,
                kernel_size=7,
                stride=1,
                padding=0,
            )
        )

        self.rms_embedding = RMSEnvelopeEmbedding(
            output_channels=model_config.upsample_initial_channel,
            hidden_channels=model_config.rms_embedding.hidden_channels,
            kernel_size=model_config.rms_embedding.kernel_size,
            layer_scale_initial=model_config.rms_embedding.layer_scale_initial,
            remap_to_activity=model_config.rms_embedding.remap_to_activity,
        ) if model_config.rms_embedding.enabled else None

        context_config = model_config.context

        if context_config.enabled:
            self.context_trunk = ConvNeXtContextTrunk1d(
                channels=model_config.upsample_initial_channel,
                number_of_blocks=context_config.number_of_blocks,
                block_type=context_config.block_type,
                kernel_size=context_config.kernel_size,
                dilation = context_config.dilation,
                multi_kernel_sizes=(
                    context_config.multi_kernel_sizes
                ),
                multi_kernel_dilations = (
                    context_config.multi_kernel_dilations
                ),
                expansion_ratio = context_config.expansion_ratio,
                layer_scale_initial = (
                    context_config.layer_scale_initial
                ),
                attn_config=context_config.attention,
                use_grn=context_config.use_grn,
                grn_eps=context_config.grn_eps,
            )
        else:
            self.context_trunk = nn.Identity()

        self.ups = nn.ModuleList()
        self.mrf_ecas = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        current_channels = model_config.upsample_initial_channel

        for upsample_rate, upsample_kernel_size in zip(
            model_config.upsample_rates,
            model_config.upsample_kernel_sizes,
        ):
            next_channels = current_channels // 2

            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        current_channels,
                        next_channels,
                        kernel_size=upsample_kernel_size,
                        stride=upsample_rate,
                        padding=(upsample_kernel_size - upsample_rate) // 2,
                    )
                )
            )

            for kernel_size, dilations in zip(
                model_config.resblock_kernel_sizes,
                model_config.resblock_dilation_sizes,
            ):
                self.resblocks.append(
                    ResBlock1D(
                        channels=next_channels,
                        kernel_size=kernel_size,
                        dilations=dilations,
                        activation=model_config.activation,
                        snake_beta_config=model_config.snake_beta,
                        alias_free_config=model_config.alias_free,
                    )
                )

            if model_config.eca.enabled:
                self.mrf_ecas.append(
                    ECABlock1d(
                        channels=next_channels,
                        kernel_size=model_config.eca.kernel_size,
                        gamma=model_config.eca.gamma,
                        beta=model_config.eca.beta,
                        minimum_kernel_size=model_config.eca.minimum_kernel_size,
                        residual=model_config.eca.residual,
                    )
                )
            else:
                self.mrf_ecas.append(nn.Identity())

            current_channels = next_channels

        self.activation_post = get_activation(model_config.activation, current_channels, model_config.snake_beta, model_config.alias_free)

        self.conv_post = weight_norm(
            nn.Conv1d(
                current_channels,
                1,
                kernel_size=7,
                stride=1,
                padding=3,
            )
        )

        self.reset_parameters()

    @property
    def total_upsample_factor(self) -> int:
        factor = 1
        for rate in self.config.upsample_rates:
            factor *= rate
        return factor

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
            self,
            cqt: torch.Tensor,
            rms_envelope: torch.Tensor,
            *,
            return_diagnostics: bool = False,
    ) -> torch.Tensor | GeneratorForwardOutput:
        if cqt.ndim == 4:
            # [B, 1, F, T] -> [B, F, T]
            cqt = cqt[:, 0, :, :]

        if cqt.ndim != 3:
            raise ValueError(
                f"Expected CQT shape [B,F,T] or [B,1,F,T], got {tuple(cqt.shape)}"
            )

        if rms_envelope.ndim != 3 or rms_envelope.shape[1] != 1:
            raise ValueError(
                f"Expected RMS envelope [B, 1, T] "
                f"got {tuple(rms_envelope.shape)} instead"
            )

        if cqt.shape[0] != rms_envelope.shape[0]:
            raise ValueError(
                f"CQT and RMS batch sizes differ: "
                f"{cqt.shape[0]} != {rms_envelope.shape[0]}"
            )

        if cqt.shape[-1] != rms_envelope.shape[-1]:
            raise ValueError(
                f"CQT and RMS frame counts differ: "
                f"{cqt.shape[-1]} != {rms_envelope.shape[-1]}"
            )

        cqt = F.pad(cqt, (3, 3), mode="constant", value=-1.0)
        cqt_features = self.conv_pre(cqt)

        rms_diagnostics = None

        if self.rms_embedding is not None:
            rms_features, rms_activity = self.rms_embedding(rms_envelope)

            if cqt_features.shape != rms_features.shape:
                raise RuntimeError(
                    "CQT and RMS embeddings have "
                    "different shapes: "
                    f"{tuple(cqt_features.shape)} "
                    f"versus "
                    f"{tuple(rms_features.shape)}."
                )

            x = cqt_features + rms_features

            if return_diagnostics:
                rms_diagnostics = self._rms_diagnostics(
                    activity=rms_activity,
                    rms_features=rms_features,
                    cqt_features=cqt_features,
                )
        else:
            x = cqt_features

        x = self.context_trunk(x)

        num_resblocks_per_stage = len(self.config.resblock_kernel_sizes)
        resblock_index = 0

        for up, eca in zip(self.ups, self.mrf_ecas):
            x = F.leaky_relu(x, negative_slope=0.1)
            x = up(x)

            fused = None

            for _ in range(num_resblocks_per_stage):
                rb_out = self.resblocks[resblock_index](x)
                fused = rb_out if fused is None else fused + rb_out
                resblock_index += 1

            x = fused / num_resblocks_per_stage

            x = eca(x)

        x = self.activation_post(x)
        pre_tanh = self.conv_post(x).float()

        audio = torch.tanh(pre_tanh) if self.config.final_tanh else pre_tanh

        if return_diagnostics:
            return GeneratorForwardOutput(
                audio=audio,
                pre_tanh=pre_tanh,
                rms_embedding=rms_diagnostics,
            )

        return audio

    def _rms_diagnostics(
            self,
            *,
            activity: torch.Tensor,
            rms_features: torch.Tensor,
            cqt_features: torch.Tensor,
    ) -> RMSEnvelopeEmbeddingOutput:
        if self.rms_embedding is None:
            raise RuntimeError(
                "RMS diagnostics requested while "
                "RMS conditioning is disabled."
            )

        activity_detached = activity.detach().float()
        rms_detached = rms_features.detach().float()
        cqt_detached = cqt_features.detach().float()

        layer_scale = self.rms_embedding.layer_scale.detach().float()

        return RMSEnvelopeEmbeddingOutput(
            activity_mean=activity_detached.mean(),
            silence_fraction=(activity_detached <= 0.369).float().mean(),
            embedding_mean=rms_detached.mean(),
            embedding_rms=torch.sqrt(rms_detached.square().mean() + 1e-8),
            embedding_max_abs=rms_detached.abs().amax(),
            to_cqt_ratio=rms_detached.norm() / cqt_detached.norm().clamp_min(1e-8),
            layer_scale_mean_abs=layer_scale.abs().mean(),
            layer_scale_max_abs=layer_scale.abs().amax(),
        )

    def remove_weight_norm(self) -> None:
        remove_parametrized_weight_norm(self.conv_pre)

        for up in self.ups:
            remove_parametrized_weight_norm(up)

        for block in self.resblocks:
            block.remove_weight_norm()

        remove_parametrized_weight_norm(self.conv_post)
