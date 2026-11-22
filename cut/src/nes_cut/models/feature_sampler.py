from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


FeatureMapDict = dict[str, torch.Tensor]
PatchIdDict = dict[str, torch.Tensor]
ProjectedFeatureDict = dict[str, torch.Tensor]


class PatchProjectionHead(nn.Module):
    def __init__(
        self,
        input_channels: int,
        projection_dim: int = 256,
    ) -> None:
        super().__init__()

        if input_channels <= 0:
            raise ValueError(f"input_channels must be > 0, got {input_channels}")

        if projection_dim <= 0:
            raise ValueError(f"projection_dim must be > 0, got {projection_dim}")

        self.input_channels = input_channels
        self.projection_dim = projection_dim

        self.mlp = nn.Sequential(
            nn.Linear(
                input_channels,
                projection_dim,
            ),
            nn.GELU(),
            nn.Linear(
                projection_dim,
                projection_dim,
            ),
        )

        self.apply(initialize_projection_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise RuntimeError(
                f"PatchProjectionHead expects 3D tensor, got {tuple(x.shape)}"
            )

        if x.shape[-1] != self.input_channels:
            raise RuntimeError(
                f"PatchProjectionHead channel mismatch: "
                f"expected {self.input_channels}, got {x.shape[-1]}"
            )

        x = self.mlp(x)

        normalized = F.normalize(
            x.float(),
            p=2,
            dim=-1,
            eps=1e-12,
        )

        return normalized.to(dtype=x.dtype)


def initialize_projection_head(
    module: nn.Module,
) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(
            module.weight,
            std=0.02,
        )

        if module.bias is not None:
            nn.init.zeros_(module.bias)


class FeatureSampler(nn.Module):
    def __init__(
        self,
        feature_channels: Mapping[str, int],
        *,
        projection_dim: int = 256,
    ) -> None:
        super().__init__()

        if not feature_channels:
            raise ValueError(f"feature_channels must not be empty")

        if projection_dim <= 0:
            raise ValueError(f"projection_dim must be > 0, got {projection_dim}")

        normalized_channels: dict[str, int] = {}

        for name, channels in feature_channels.items():
            if not name:
                raise ValueError("Feature-layer names must not be empty")

            if channels <= 0:
                raise ValueError(f"Feature-layer {name!r} channels must be > 0, got {channels}")

            normalized_channels[name] = channels

        self.feature_channels = normalized_channels
        self.feature_names = tuple(normalized_channels)
        self.projection_dim = projection_dim

        self.projection_heads = nn.ModuleDict(
            {
                name: PatchProjectionHead(
                    input_channels=channels,
                    projection_dim=projection_dim,
                ) for name, channels in normalized_channels.items()
            }
        )

    def _validate_features(
            self,
            features: Mapping[str, torch.Tensor],
    ) -> None:
        expected = set(self.feature_names)
        provided = set(features)

        missing = sorted(expected - provided)
        extra = sorted(provided - expected)

        if missing or extra:
            raise ValueError(
                "Feature-map keys do not match sampler "
                f"configuration: missing={missing}, extra={extra}"
            )

        batch_sizes: set[int] = set()

        for name in self.feature_names:
            feature = features[name]

            if feature.ndim != 4:
                raise RuntimeError(
                    f"Feature {name!r} expects 4D tensor, got {tuple(feature.shape)}"
                )

            expected_channels = self.feature_channels[name]

            if feature.shape[1] != expected_channels:
                raise RuntimeError(
                    f"Feature {name!r} channel mismatch: "
                    f"expected {expected_channels}, got {feature.shape[1]}"
                )

            if feature.shape[0] <= 0:
                raise RuntimeError(
                    f"Feature {name!r} has an empty batch."
                )

            if feature.shape[-2] <= 0 or feature.shape[-1] <= 0:
                raise RuntimeError(
                    f"Feature {name!r} has invalid spatial shape "
                    f"{tuple(feature.shape[-2:])}."
                )

            if not feature.is_floating_point():
                raise TypeError(
                    f"Feature {name!r} must be floating point, got {feature.dtype}"
                )

            batch_sizes.add(feature.shape[0])

        if len(batch_sizes) != 1:
            raise RuntimeError(
                f"All feature maps must have the same batch size, got {sorted(batch_sizes)}"
            )

    @staticmethod
    def _sample_patch_ids(
            *,
            batch_size: int,
            spatial_size: int,
            num_patches: int,
            device: torch.device,
            generator: torch.Generator | None,
    ) -> torch.Tensor:
        if spatial_size <= 0:
            raise ValueError(f"spatial_size must be > 0, got {spatial_size}")

        if num_patches < 0:
            raise ValueError(f"num_patches must be >= 0, got {num_patches}")

        patches_per_example = min(
            num_patches,
            spatial_size,
        )

        if patches_per_example == 0:
            return torch.arange(
                spatial_size,
                device=device,
                dtype=torch.long,
            ).expand(
                batch_size,
                spatial_size,
            )

        random_scores = torch.rand(
            batch_size,
            spatial_size,
            device=device,
            generator=generator,
        )

        return random_scores.topk(
            k=patches_per_example,
            dim=1,
            largest=True,
            sorted=False,
        ).indices

    @staticmethod
    def _validate_patch_ids(
            patch_ids: torch.Tensor,
            *,
            name: str,
            batch_size: int,
            spatial_size: int,
    ) -> None:
        if patch_ids.ndim != 2:
            raise RuntimeError(
                f"Patch IDs {name!r} expects 2D tensor, got {tuple(patch_ids.shape)}"
            )

        if patch_ids.shape[0] != batch_size:
            raise RuntimeError(
                f"Patch IDs {name!r} batch size mismatch: "
                f"expected {batch_size}, got {patch_ids.shape[0]}"
            )

        if patch_ids.dtype != torch.long:
            raise RuntimeError(
                f"Patch IDs {name!r} dtype mismatch: "
                f"expected {torch.long}, got {patch_ids.dtype}"
            )

        if patch_ids.device.type != "meta":
            if torch.any(patch_ids < 0):
                raise ValueError(f"Patch IDs for {name!r} contain negative indices")

            if torch.any(patch_ids >= spatial_size):
                raise ValueError(f"Patch IDs for {name!r} exceed spatial size {spatial_size}")

    @staticmethod
    def _gather_patches(
            flattened: torch.Tensor,
            patch_ids: torch.Tensor,
    ) -> torch.Tensor:
        # flattened: [B, HW, C]
        # patch_ids: [B, P]
        gather_index = patch_ids.unsqueeze(-1).expand(-1, -1, flattened.shape[-1])

        return torch.gather(
            flattened,
            dim=1,
            index=gather_index,
        )

    def forward(
            self,
            features: Mapping[str, torch.Tensor],
            *,
            num_patches: int,
            patch_ids: Mapping[str, torch.Tensor] | None = None,
            generator: torch.Generator | None = None,
    ) -> tuple[ProjectedFeatureDict, PatchIdDict]:
        self._validate_features(features)

        if num_patches < 0:
            raise ValueError(f"num_patches must be >= 0, got {num_patches}")

        if patch_ids is not None:
            expected = set(self.feature_names)
            provided = set(patch_ids)

            if expected != provided:
                raise ValueError(
                    f"Patch ID keys do not match sampler configuration: "
                    f"missing={sorted(expected - provided)}, extra={sorted(provided - expected)}"
                )

        projected_features: ProjectedFeatureDict = {}
        selected_ids: PatchIdDict = {}

        for name in self.feature_names:
            feature = features[name]

            batch_size, channels, height, width = feature.shape

            spatial_size = height * width

            flattened = feature.permute(0, 2, 3, 1).reshape(batch_size, spatial_size, channels)

            if patch_ids is None:
                ids = self._sample_patch_ids(
                    batch_size=batch_size,
                    spatial_size=spatial_size,
                    num_patches=num_patches,
                    device=feature.device,
                    generator=generator,
                )
            else:
                ids = patch_ids[name]

                if ids.device != feature.device:
                    ids = ids.to(feature.device)

                self._validate_patch_ids(
                    ids,
                    name=name,
                    batch_size=batch_size,
                    spatial_size=spatial_size,
                )

            patches = self._gather_patches(
                flattened=flattened,
                patch_ids=ids,
            )

            projected_features[name] = self.projection_heads[name](patches)
            selected_ids[name] = ids

        return projected_features, selected_ids


