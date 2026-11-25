from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary positional embedding with optional partial rotation.

    Input:
        [B, H, T, D]

    The first `rotary_dimension` channels are rotated as adjacent
    two-dimensional pairs:

        (0, 1), (2, 3), ...

    Any remaining channels pass through unchanged.

    Notes:
        - RoPE calculations are performed in float32.
        - The result is restored to the input dtype.
        - Query and key can share one cosine/sine calculation through
          `rotate_queries_and_keys()`.
    """

    inverse_frequencies: torch.Tensor

    def __init__(
        self,
        head_dimension: int,
        *,
        base: float = 10_000.0,
        partial_rotary_factor: float = 1.0,
    ) -> None:
        super().__init__()

        if head_dimension <= 0:
            raise ValueError(
                "head_dimension must be positive, "
                f"got {head_dimension}."
            )

        if head_dimension % 2 != 0:
            raise ValueError(
                "RoPE head_dimension must be even, "
                f"got {head_dimension}."
            )

        if not torch.isfinite(torch.tensor(base)):
            raise ValueError(
                f"base must be finite, got {base}."
            )

        if base <= 0.0:
            raise ValueError(
                f"base must be positive, got {base}."
            )

        if not (0.0 < partial_rotary_factor <= 1.0):
            raise ValueError(
                "partial_rotary_factor must lie in "
                "(0.0, 1.0], got "
                f"{partial_rotary_factor}."
            )

        requested_rotary_dimension = head_dimension * partial_rotary_factor

        # Round down to the nearest positive even dimension.
        rotary_dimension = int(requested_rotary_dimension) // 2 * 2
        rotary_dimension = max(2, rotary_dimension)

        if rotary_dimension > head_dimension:
            raise ValueError(
                "rotary_dimension may not exceed "
                "head_dimension, got "
                f"{rotary_dimension} and "
                f"{head_dimension}."
            )

        self.head_dimension = int(head_dimension)
        self.rotary_dimension = int(rotary_dimension)
        self.base = float(base)
        self.partial_rotary_factor = float(partial_rotary_factor)
        self.realized_partial_rotary_factor = self.rotary_dimension / self.head_dimension

        frequency_indices = torch.arange(0, self.rotary_dimension, 2, dtype=torch.float32)

        inverse_frequencies = torch.pow(
            torch.tensor(
                self.base,
                dtype=torch.float32,
            ),
            -frequency_indices
            / self.rotary_dimension,
        )

        self.register_buffer(
            "inverse_frequencies",
            inverse_frequencies,
            persistent=False,
        )

    def _validate_input(
        self,
        value: torch.Tensor,
        *,
        name: str,
    ) -> None:
        if value.ndim != 4:
            raise ValueError(
                f"{name} must have shape [B,H,T,D], "
                f"got {tuple(value.shape)}."
            )

        if value.shape[-1] != self.head_dimension:
            raise ValueError(
                f"{name} head dimension mismatch: "
                f"expected {self.head_dimension}, "
                f"got {value.shape[-1]}."
            )

        if value.shape[-2] <= 0:
            raise ValueError(
                f"{name} sequence length must be "
                "positive."
            )

    def _cosine_and_sine(
        self,
        *,
        sequence_length: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence_length <= 0:
            raise ValueError(
                "sequence_length must be positive, "
                f"got {sequence_length}."
            )

        positions = torch.arange(sequence_length, device=device, dtype=torch.float32)

        inverse_frequencies = self.inverse_frequencies.to(device=device, dtype=torch.float32)

        # [T] outer [D_rot / 2]
        # -> [T, D_rot / 2]
        angles = torch.outer(positions,inverse_frequencies)

        # [1, 1, T, D_rot / 2]
        cosine = torch.cos(angles)[None, None, :, :]
        sine = torch.sin(angles)[None, None, :, :]

        return cosine, sine

    def _apply_rotary(
        self,
        value: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> torch.Tensor:
        original_dtype = value.dtype

        rotating = value[..., : self.rotary_dimension].float()

        expected_table_shape = (1, 1, value.shape[-2], self.rotary_dimension // 2)

        if cosine.shape != expected_table_shape:
            raise ValueError(
                "Unexpected cosine table shape: "
                f"expected {expected_table_shape}, "
                f"got {tuple(cosine.shape)}."
            )

        if sine.shape != cosine.shape:
            raise ValueError(
                "Cosine and sine table shapes "
                "must match."
            )

        # [B,H,T,D_rot]
        # -> [B,H,T,D_rot/2,2]
        paired = rotating.reshape(*rotating.shape[:-1], self.rotary_dimension // 2, 2)

        first = paired[..., 0]
        second = paired[..., 1]

        rotated_first = first * cosine - second * sine
        rotated_second = first * sine + second * cosine

        rotated = torch.stack((rotated_first, rotated_second), dim=-1)
        rotated = rotated.flatten(start_dim=-2)
        rotated = rotated.to(dtype=original_dtype)

        if self.rotary_dimension == self.head_dimension:
            return rotated

        passthrough = value[..., self.rotary_dimension :]

        return torch.cat((rotated, passthrough), dim=-1)

    def rotate_queries_and_keys(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Rotate query and key using one shared trigonometric table.

        This implementation is intended for self-attention, so query
        and key must have matching shapes and devices.
        """
        self._validate_input(query, name="query")
        self._validate_input(key, name="key")

        if query.shape != key.shape:
            raise ValueError(
                "Query and key shapes must match "
                "for self-attention RoPE, got "
                f"{tuple(query.shape)} and "
                f"{tuple(key.shape)}."
            )

        if query.device != key.device:
            raise ValueError(
                "Query and key must be on the "
                "same device, got "
                f"{query.device} and "
                f"{key.device}."
            )

        cosine, sine = self._cosine_and_sine(sequence_length=query.shape[-2], device=query.device)

        return (
            self._apply_rotary(query, cosine, sine),
            self._apply_rotary(key, cosine, sine),
        )

    def forward(
        self,
        value: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_input(value, name="value")

        cosine, sine = self._cosine_and_sine(sequence_length=value.shape[-2], device=value.device)

        return self._apply_rotary(value, cosine, sine)


class RoPETemporalSelfAttention1d(
    nn.Module
):
    """
    Pre-normalized temporal multi-head self-attention with partial RoPE.

    Input and output:
        [B, C, T]

    Temporal positions are sequence tokens and channels form each
    token embedding.

    When `window_size` is provided, each interior token may attend to
    exactly `window_size` positions. The window size must be odd so
    that the local window is symmetric around the current token.

    Example:
        window_size = 63

        31 frames left
        + current frame
        + 31 frames right
    """

    def __init__(
        self,
        channels: int,
        *,
        number_of_heads: int = 8,
        dropout: float = 0.0,
        rope_base: float = 500.0,
        partial_rotary_factor: float = 0.5,
        window_size: int | None = None,
        layer_scale_initial: float = 1e-4,
        qkv_bias: bool = True,
        output_bias: bool = False,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(
                "channels must be positive, "
                f"got {channels}."
            )

        if number_of_heads <= 0:
            raise ValueError(
                "number_of_heads must be "
                "positive, got "
                f"{number_of_heads}."
            )

        if channels % number_of_heads != 0:
            raise ValueError(
                "channels must be divisible by "
                "number_of_heads, got "
                f"channels={channels} and "
                f"heads={number_of_heads}."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must lie in [0,1), "
                f"got {dropout}."
            )

        if window_size is not None:
            if window_size <= 0:
                raise ValueError(
                    "window_size must be positive "
                    f"when set, got {window_size}."
                )

            if window_size % 2 == 0:
                raise ValueError(
                    "window_size must be odd for "
                    "symmetric temporal attention, "
                    f"got {window_size}."
                )

        if layer_scale_initial < 0.0:
            raise ValueError(
                "layer_scale_initial must be "
                "non-negative, got "
                f"{layer_scale_initial}."
            )

        head_dimension = channels // number_of_heads

        if head_dimension % 2 != 0:
            raise ValueError(
                "RoPE head dimension must be "
                "even, got "
                f"{head_dimension}."
            )

        self.channels = int(channels)
        self.number_of_heads = int(number_of_heads)
        self.head_dimension = int(head_dimension)
        self.dropout = float(dropout)
        self.window_size = int(window_size) if window_size is not None else None

        self.norm = nn.LayerNorm(
            normalized_shape=channels
        )

        self.qkv_projection = nn.Linear(
            in_features=channels,
            out_features=3 * channels,
            bias=qkv_bias,
        )

        self.rotary_embedding = (
            RotaryPositionEmbedding(
                head_dimension,
                base=rope_base,
                partial_rotary_factor=partial_rotary_factor,
            )
        )

        self.output_projection = nn.Linear(
            in_features=channels,
            out_features=channels,
            bias=output_bias,
        )

        self.layer_scale = nn.Parameter(
            torch.full((channels,), fill_value=layer_scale_initial, dtype=torch.float32)
        )

    def _split_heads(
        self,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert [B,T,C] to [B,H,T,D].
        """
        if value.ndim != 3:
            raise ValueError(
                "Expected token tensor [B,T,C], "
                f"got {tuple(value.shape)}."
            )

        batch_size, sequence_length, channels = value.shape

        if channels != self.channels:
            raise ValueError(
                "Attention channel mismatch "
                "while splitting heads: expected "
                f"{self.channels}, got {channels}."
            )

        value = value.reshape(batch_size, sequence_length, self.number_of_heads, self.head_dimension)

        return value.transpose(1,2)

    def _merge_heads(
        self,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert [B,H,T,D] to [B,T,C].
        """
        if value.ndim != 4:
            raise ValueError(
                "Expected attention tensor "
                "[B,H,T,D], got "
                f"{tuple(value.shape)}."
            )

        batch_size, number_of_heads, sequence_length, head_dimension = value.shape

        if number_of_heads != self.number_of_heads:
            raise ValueError(
                "Attention head mismatch while "
                "merging heads: expected "
                f"{self.number_of_heads}, got "
                f"{number_of_heads}."
            )

        if head_dimension != self.head_dimension:
            raise ValueError(
                "Attention head-dimension "
                "mismatch while merging heads: "
                f"expected {self.head_dimension}, "
                f"got {head_dimension}."
            )

        value = value.transpose(1,2).contiguous()

        return value.reshape(batch_size, sequence_length, self.channels)

    @staticmethod
    def _sliding_window_mask(
        *,
        sequence_length: int,
        window_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Return an allowed-attention mask [T,T].

        True:
            the key position is visible to the query.

        False:
            the key position is masked.

        For an interior token, exactly `window_size` positions are
        visible. Boundary tokens naturally see fewer positions.
        """
        if sequence_length <= 0:
            raise ValueError(
                "sequence_length must be "
                "positive, got "
                f"{sequence_length}."
            )

        if window_size <= 0:
            raise ValueError(
                "window_size must be positive, "
                f"got {window_size}."
            )

        if window_size % 2 == 0:
            raise ValueError(
                "window_size must be odd, got "
                f"{window_size}."
            )

        positions = torch.arange(sequence_length, device=device)

        distance = (positions[:, None] - positions[None, :]).abs()

        half_window = window_size // 2

        return distance <= half_window

    def forward(
        self,
        value: torch.Tensor,
    ) -> torch.Tensor:
        if value.ndim != 3:
            raise ValueError(
                "RoPETemporalSelfAttention1d "
                "expects [B,C,T], got "
                f"{tuple(value.shape)}."
            )

        if value.shape[1] != self.channels:
            raise ValueError(
                "Attention channel mismatch: "
                f"expected {self.channels}, "
                f"got {value.shape[1]}."
            )

        if value.shape[-1] <= 0:
            raise ValueError(
                "Attention sequence length "
                "must be positive."
            )

        input_dtype = value.dtype

        sequence_length = int(value.shape[-1])

        # [B,C,T] -> [B,T,C]
        residual = value.transpose(1,2)

        tokens = self.norm(residual)

        # [B,T,C] -> [B,T,3C]
        query_key_value = self.qkv_projection(tokens)

        query, key, attention_value = query_key_value.chunk(3, dim=-1)

        # [B,T,C] -> [B,H,T,D]
        query = self._split_heads(query)
        key = self._split_heads(key)

        attention_value = self._split_heads(attention_value)

        query, key = self.rotary_embedding.rotate_queries_and_keys(query, key)

        allowed_attention_mask: torch.Tensor | None = None

        if self.window_size is not None and sequence_length > self.window_size:
            allowed_attention_mask = (
                self._sliding_window_mask(
                    sequence_length=sequence_length,
                    window_size=self.window_size,
                    device=value.device,
                )
            )

        attention_output = (
            F.scaled_dot_product_attention(
                query,
                key,
                attention_value,
                attn_mask=allowed_attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        )

        # [B,H,T,D] -> [B,T,C]
        attention_output = self._merge_heads(attention_output)
        attention_output = self.output_projection(attention_output)

        # Perform LayerScale and residual accumulation in FP32.
        scaled_attention = attention_output.float() * self.layer_scale[None, None, :]

        tokens = (residual.float() + scaled_attention).to(dtype=input_dtype)

        # [B,T,C] -> [B,C,T]
        output = tokens.transpose(1,2)

        if output.shape != value.shape:
            raise RuntimeError(
                "Temporal attention changed "
                "tensor shape: input="
                f"{tuple(value.shape)}, output="
                f"{tuple(output.shape)}."
            )

        return output


