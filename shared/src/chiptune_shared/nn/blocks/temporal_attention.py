from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Positional Embedding for Temporal Attention.

    Input:
        [B, heads, time, head_dimension]

    The Final dimension must be even because adjacent feature
    values are interpreted as two-dimensional rotation pairs.
    """

    def __init__(
            self,
            head_dimension: int,
            *,
            base: float = 10_000.0,
    ) -> None:
        super().__init__()

        if head_dimension <= 0:
            raise ValueError(f"head_dimension must be positive, got {head_dimension}")

        if head_dimension % 2 != 0:
            raise ValueError(f"RoPE requires an even head dimension, got {head_dimension}")

        if base <= 0.0:
            raise ValueError(f"base must be positive, got {base}")

        self.head_dimension = head_dimension
        self.base = base

        inverse_frequencies = 1.0 / (base ** (torch.arange(0, head_dimension, 2, dtype=torch.float32) / head_dimension))

        self.register_buffer("inverse_frequencies", inverse_frequencies, persistent=False)

    def _cosine_and_sine(
            self,
            *,
            sequence_length: int,
            device: torch.device,
            dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(sequence_length, device=device, dtype=torch.float32)

        inverse_frequencies = self.inverse_frequencies.to(device=device, dtype=torch.float32)

        # [T] outer [D / 2] -> [T, D / 2]
        angles = torch.outer(positions, inverse_frequencies)

        cosine = torch.cos(angles).to(dtype=dtype)
        sine = torch.sin(angles).to(dtype=dtype)

        # Broadcast across batch and heads
        #
        # [1, 1, T, D / 2]
        cosine = cosine.unsqueeze(0).unsqueeze(0)
        sine = sine.unsqueeze(0).unsqueeze(0)

        return cosine, sine

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"RotaryPositionalEmbedding expects [B, H, T, D], got {tuple(x.shape)}")

        if x.shape[-1] != self.head_dimension:
            raise ValueError(f"RoPE head-dimension mismatch: expected {self.head_dimension}, got {x.shape[-1]}")

        sequence_length = x.shape[-2]

        cosine, sine = self._cosine_and_sine(
            sequence_length=sequence_length,
            device=x.device,
            dtype=x.dtype,
        )

        # [B, H, T, D] -> [B, H, T, D / 2, 2]
        paired = x.reshape(
            *x.shape[:-1],
            self.head_dimension // 2,
            2,
        )

        first = paired[..., 0]
        second = paired[..., 1]

        rotated_first = (first * cosine) + (-second * sine)
        rotated_second = (first * sine) + (second * cosine)

        # Interleave rotated pairs and restore [B, H, T, D]
        rotated = torch.stack(
            (rotated_first, rotated_second),
            dim=-1,
        )

        return rotated.flatten(start_dim=-2)


class RoPETemporalSelfAttention1d(nn.Module):
    """
    Temporal multi-head self-attention layer with rotary positional embedding.

    Input & Output:
        [B, C, T]

    Temporal positions are treated as sequence tokens.
    Hidden channels form each tokens embedding.
    """
    def __init__(
            self,
            channels: int,
            *,
            number_of_heads: int = 8,
            dropout: float = 0.0,
            rope_base: float = 10_000.0,
            layer_scale_initial: float = 1e-4,
            qkv_bias: bool = True,
            output_bias: bool = True,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")

        if number_of_heads <= 0:
            raise ValueError(f"number_of_heads must be positive, got {number_of_heads}")

        if channels % number_of_heads != 0:
            raise ValueError(f"channels must be divisible by number_of_heads, "
                             f"got channels={channels} and heads={number_of_heads}")

        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        if layer_scale_initial < 0.0:
            raise ValueError(f"layer_scale_initial must be positive, got {layer_scale_initial}")

        head_dim = channels // number_of_heads

        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head-dimension must be even, got {head_dim}")

        self.channels = channels
        self.number_of_heads = number_of_heads
        self.head_dim = head_dim
        self.dropout = dropout

        self.norm = nn.LayerNorm(channels)

        self.qkv_projection = nn.Linear(
            in_features=channels,
            out_features=3 * channels,
            bias=qkv_bias,
        )

        self.rotary_embedding = RotaryPositionEmbedding(
            head_dim,
            base=rope_base,
        )

        self.output_projection = nn.Linear(
            in_features=channels,
            out_features=channels,
            bias=output_bias,
        )

        self.layer_scale = nn.Parameter(
            torch.full(
                (channels,),
                fill_value=layer_scale_initial,
                dtype=torch.float32,
            )
        )

    def _split_heads(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        """
        [B, T, C] -> [B, H, T, D]
        """
        batch_size, sequence_length, channels = x.shape

        if channels != self.channels:
            raise ValueError(f"Attention channel mismatch while splitting heads: "
                             f"expected {self.channels}, got {channels}")

        x = x.reshape(
            batch_size,
            sequence_length,
            self.number_of_heads,
            self.head_dim,
        )

        return x.transpose(1, 2)

    def _merge_heads(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        """
        [B, H, T, D] -> [B, C, T]
        """
        batch_size, num_heads, sequence_length, head_dim = x.shape

        if num_heads != self.number_of_heads:
            raise ValueError(f"Attention heads mismatch while merging heads: "
                             f"expected {self.number_of_heads}, got {num_heads}")

        if head_dim != self.head_dim:
            raise ValueError(f"Attention head dimension mismatch while merging heads: "
                             f"expected {self.head_dim}, got {head_dim}")

        x = x.transpose(1, 2).contiguous()

        return x.reshape(
            batch_size,
            sequence_length,
            self.channels,
        )

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"RoPETemporalSelfAttention expects [B, T, C], got {tuple(x.shape)}")

        if x.shape[1] != self.channels:
            raise ValueError(f"Attention channel mismatch: "
                             f"expected {self.channels}, got {x.shape[1]}")

        # [B, C, T] -> [B, T, C]
        tokens = x.transpose(1, 2)
        residual = tokens

        tokens = self.norm(tokens)

        # [B, T, C] -> [B, T, 3C]
        qkv = self.qkv_projection(tokens)

        query, key, value = qkv.chunk(3, dim=-1)

        # [B, T, C] -> [B, H, T, D]
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        query = self.rotary_embedding(query)
        key = self.rotary_embedding(key)

        attention_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        # [B, H, T, D] -> [B, T, C]
        attention_output = self._merge_heads(attention_output)

        attention_output = self.output_projection(attention_output)

        attention_output = self.layer_scale * attention_output

        tokens = residual + attention_output

        # [B, T, C] -> [B, C, T]
        output = tokens.transpose(1, 2)

        if output.shape != x.shape:
            raise RuntimeError(f"Temporal attention changed tensor shape: "
                               f"input={tuple(x.shape)}, output={tuple(output.shape)}")

        return output

