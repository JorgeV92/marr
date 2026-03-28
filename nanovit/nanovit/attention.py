from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def window_partition(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    """Partition [B, H, W, C] into non-overlapping windows.

    Returns:
        windows: [B * num_windows, window_size * window_size, C]
        padded_hw: padded spatial size used for partitioning
        original_hw: original spatial size before padding
    """
    b, h, w, c = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        # F.pad expects NCHW, so temporarily permute.
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, (0, pad_w, 0, pad_h))
        x = x.permute(0, 2, 3, 1)

    hp, wp = x.shape[1], x.shape[2]
    x = x.view(b, hp // window_size, window_size, wp // window_size, window_size, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = x.view(-1, window_size * window_size, c)
    return windows, (hp, wp), (h, w)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    padded_hw: tuple[int, int],
    original_hw: tuple[int, int],
    batch_size: int,
) -> torch.Tensor:
    hp, wp = padded_hw
    h, w = original_hw
    c = windows.shape[-1]
    x = windows.view(batch_size, hp // window_size, wp // window_size, window_size, window_size, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(batch_size, hp, wp, c)
    return x[:, :h, :w, :]


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        qkv_bias: bool = True,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

        table_size = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # [2, ws, ws]
        coords_flatten = coords.flatten(1)  # [2, ws*ws]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, N, N]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [N, N, 2]
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # [N, N]
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B_windows, tokens_per_window, C]
        b_windows, n, c = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(b_windows, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B_windows, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores: [B_windows, heads, N, N]
        attn = (q * self.scale) @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(n, n, self.num_heads).permute(2, 0, 1)
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # [B_windows, heads, N, head_dim]
        out = out.transpose(1, 2).reshape(b_windows, n, c)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out