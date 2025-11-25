import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from trdlm_chat.utils.other import rms_norm


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        pos_embedding: nn.Module,
    ) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self._num_heads = num_heads
        self._pos_embedding = pos_embedding

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim must be divisible by num_heads, got {hidden_dim} and {num_heads}"
            )

        self._head_dim = hidden_dim // num_heads

        self._q = nn.Linear(hidden_dim, num_heads * self._head_dim, bias=False)
        self._k = nn.Linear(hidden_dim, num_heads * self._head_dim, bias=False)
        self._v = nn.Linear(hidden_dim, num_heads * self._head_dim, bias=False)
        self._proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()  # TODO: figure out why the dim is 3

        # Projection
        q = self._q(x).view(batch_size, seq_len, self._num_heads, self._head_dim)
        k = self._q(x).view(batch_size, seq_len, self._num_heads, self._head_dim)
        v = self._q(x).view(batch_size, seq_len, self._num_heads, self._head_dim)

        # Pos embedding
        q = self._pos_embedding(q)
        k = self._pos_embedding(k)
        q, k = rms_norm(q), rms_norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Attention
        y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        y = self._proj(y)
        return y


class SwiGLU(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int = 2048,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._gate_proj = nn.Linear(hidden_dim, ff_dim, bias=bias)
        self._open_proj = nn.Linear(hidden_dim, ff_dim, bias=bias)
        self._close_proj = nn.Linear(ff_dim, hidden_dim, bias=bias)
        self._dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self._gate_proj(x)
        value = self._open_proj(x)
        activated = F.silu(gate) * value
        activated = self._dropout(activated)
        return self._close_proj(activated)


class TransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        pos_embedding: nn.Module,
        ff_head: nn.Module,
    ) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self._num_heads = num_heads
        self._pos_embedding = pos_embedding
        self._ff_head = ff_head

        self._attention = SelfAttention(hidden_dim, num_heads, pos_embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._attention(rms_norm(x))
        x = x + self._ff_head(rms_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        pos_embedding: nn.Module,
        ff_head: nn.Module,
    ) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self._num_heads = num_heads
        self._pos_embedding = pos_embedding
        self._ff_head = copy.deepcopy(ff_head)

        self._layers = nn.ModuleList(
            [
                TransformerLayer(hidden_dim, num_heads, pos_embedding, ff_head)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self._layers:
            x = layer(x)
        return x
