from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MultiLayerPatchNCEOutput:
    total: torch.Tensor
    per_layer: dict[str, torch.Tensor]


class PatchNCELoss(nn.Module):
    """
    Patchwise InfoNCE with within-example negatives.

    Inputs:
        query: [B, P, D]
            Projected features from the generated output.

        key: [B, P, D]
            Projected features from the source input.

    For each query patch i, key patch i is positive and all other
    key patches from the same example are negatives.
    """

    def __init__(
            self,
            *,
            temperature: float = 0.07,
            detach_key: bool = True,
            normalized_inputs: bool = True,
    ) -> None:
        super().__init__()

        if temperature <= 0.0:
            raise ValueError(
                "Temperature must be positive, "
                f"got {temperature}"
            )

        self.temperature = temperature
        self.detach_key = detach_key
        self.normalized_inputs = normalized_inputs

    @staticmethod
    def _validate_inputs(
            query: torch.Tensor,
            key: torch.Tensor,
    ) -> None:
        if query.ndim != 3:
            raise RuntimeError(
                f"PatchNCE query must have shape [B, P, D], "
                f"got {query.ndim}"
            )

        if key.ndim != 3:
            raise RuntimeError(
                f"PatchNCE key must have shape [B, P, D], "
                f"got {key.ndim}"
            )

        if query.shape != key.shape:
            raise RuntimeError(
                f"PatchNCE query and key must have the same shape, "
                f"got query={tuple(query.shape)}, key={tuple(key.shape)}"
            )

        batch_size, num_patches, embedding_dim = query.shape

        if batch_size <= 0:
            raise RuntimeError(
                f"PatchNCE Batch size must be positive, "
                f"got {batch_size}"
            )

        if num_patches < 2:
            raise RuntimeError(
                f"PatchNCE requires at least two patches per "
                f"example, got {num_patches}"
            )

        if embedding_dim <= 0:
            raise RuntimeError(
                f"PatchNCE Embedding dimension must be positive, "
                f"got {embedding_dim}"
            )

        if not query.is_floating_point():
            raise TypeError(
                f"PatchNCE query must be a floating-point "
                f"tensor, got {query.dtype}"
            )

        if not key.is_floating_point():
            raise TypeError(
                f"PatchNCE key must be a floating-point "
                f"tensor, got {key.dtype}"
            )

        if query.device != key.device:
            raise RuntimeError(
                f"PatchNCE query and key must have the same device, "
                f"got {query.device} and {key.device}"
            )

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(query, key)

        if self.detach_key:
            key = key.detach()

        # Compute similarity and cross-entropy in FP32
        query_fp32 = query.float()
        key_fp32 = key.float()

        if self.normalized_inputs:
            query_fp32 = F.normalize(
                query_fp32,
                p=2,
                dim=-1,
                eps=1e-12,
            )
            key_fp32 = F.normalize(
                key_fp32,
                p=2,
                dim=-1,
                eps=1e-12,
            )

        # [B, P, D] @ [B, P, D] -> [B, P, P]
        logits = torch.bmm(
            query_fp32,
            key_fp32.transpose(1, 2)
        )

        logits = logits / self.temperature

        batch_size, num_patches, _ = logits.shape

        labels = torch.arange(
            num_patches,
            device=logits.device,
            dtype=torch.long,
        )

        labels = labels.unsqueeze(0).expand(batch_size, num_patches)

        return F.cross_entropy(
            logits.reshape(
                batch_size * num_patches,
                num_patches,
            ),
            labels.reshape(
                batch_size * num_patches,
            ),
        )


class MultiLayerPatchNCELoss(nn.Module):
    def __init__(
            self,
            feature_layers: Sequence[str],
            *,
            temperature: float = 0.07,
            detach_key: bool = True,
            layer_weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()

        if not feature_layers:
            raise ValueError(
                f"feature_layers must not be empty"
            )

        if len(set(feature_layers)) != len(feature_layers):
            raise ValueError(
                f"feature_layers must not contain duplicates"
            )

        self.feature_layers = tuple(feature_layers)

        self.loss = PatchNCELoss(
            temperature=temperature,
            detach_key=detach_key,
            normalized_inputs=True,
        )

        if layer_weights is None:
            normalized_weights = {
                name: 1.0
                for name in self.feature_layers
            }
        else:
            missing = set(self.feature_layers) - set(layer_weights)
            extra = set(layer_weights) - set(self.feature_layers)

            if missing or extra:
                raise ValueError(
                    f"layer_weights do not match feature_layers: "
                    f"missing={sorted(missing)}, extra={sorted(extra)}"
                )

            normalized_weights = {}

            for name in self.feature_layers:
                weight = float(layer_weights[name])

                if weight < 0.0:
                    raise ValueError(
                        f"Layer weight for {name!r} must be non-negative, "
                        f"got {weight}"
                    )

                normalized_weights[name] = weight

        weight_sum = sum(normalized_weights.values())

        if weight_sum <= 0.0:
            raise ValueError(
                f"At least one PatchNCE layer weight musty be positive."
            )

        self.layer_weights = {
            name: weight / weight_sum
            for name, weight in normalized_weights.items()
        }

    def _validate_feature_maps(
            self,
            query_features: Mapping[str, torch.Tensor],
            key_features: Mapping[str, torch.Tensor],
    ) -> None:
        expected = set(self.feature_layers)
        query_names = set(query_features)
        key_names = set(key_features)

        if query_names != expected:
            raise ValueError(
                f"query_names do not match feature_layers: "
                f"missing={sorted(expected - query_names)}, "
                f"extra={sorted(query_names - expected)}"
            )

        if key_names != expected:
            raise ValueError(
                f"key_names do not match feature_layers: "
                f"missing={sorted(expected - key_names)}, "
                f"extra={sorted(key_names - expected)}"
            )

    def forward(
            self,
            query_features: Mapping[str, torch.Tensor],
            key_features: Mapping[str, torch.Tensor],
    ) -> MultiLayerPatchNCEOutput:
        self._validate_feature_maps(
            query_features,
            key_features,
        )

        per_layer: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None

        for name in self.feature_layers:
            layer_loss = self.loss(
                query_features[name],
                key_features[name],
            )

            per_layer[name] = layer_loss

            weighted = self.layer_weights[name] * layer_loss

            total = weighted if total is None else total + weighted

        if total is None:
            raise RuntimeError(
                f"No PatchNCE layers were evaluated."
            )

        return MultiLayerPatchNCEOutput(
            total=total,
            per_layer=per_layer,
        )


