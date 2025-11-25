from __future__ import annotations

import torch
import torch.nn as nn

from ...utils.other import rms_norm
from .attention import SwiGLU, Transformer
from .base import Core
from .rope import RotaryEmbedding


class DiffusionTransformerTRM(Core):
    """
    Diffusion Transformer TRM core for using it for the TRM model
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
        vocab_size: int = 65,
        pos_embedding: nn.Module | None = None,
        ff_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout = dropout
        self._vocab_size = vocab_size

        self._pos_embedding = (
            RotaryEmbedding(hidden_dim) if pos_embedding is None else pos_embedding
        )
        self._ff_head = (
            SwiGLU(hidden_dim, ff_dim=hidden_dim * 4, dropout=dropout)
            if ff_head is None
            else ff_head
        )

        self._transformer_encoder = Transformer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            pos_embedding=self._pos_embedding,
            ff_head=self._ff_head,
        )

        self._y_init = nn.Buffer(torch.randn((1, 1, hidden_dim)), persistent=True)
        self._z_init = nn.Buffer(torch.randn((1, 1, hidden_dim)), persistent=True)

    @property
    def y_init(self) -> nn.Buffer:
        return self._y_init

    @property
    def z_init(self) -> nn.Buffer:
        return self._z_init

    def forward(
        self, x: torch.Tensor | None, y: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x is None:
            if y.shape != z.shape:
                raise ValueError(
                    f"y and z must have the same shape, got y shape {y.shape} and z shape {z.shape}"
                )

            sum_in = y + z

        else:
            if x.shape != y.shape or y.shape != z.shape:
                raise ValueError(
                    f"x, y and z must have the same shape, got x shape {x.shape}, y shape {y.shape} "
                    f"and z shape {z.shape}"
                )

            sum_in = x + y + z

        transformer_output = self._transformer_encoder(sum_in)
        transformer_output = rms_norm(transformer_output, 1e-6)
        return (
            transformer_output,
            transformer_output,
        )  # A bit of a hack, but it should work with everything else

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def get_adamw_params(self) -> list[nn.Parameter]:
        return []

    def get_muon_params(self) -> list[nn.Parameter]:
        transformer_params = list(self._transformer_encoder.parameters())
        return transformer_params


class InputEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        self._vocab_size = vocab_size
        self._embedding = nn.Embedding(vocab_size + 1, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._embedding(x)


class LinearOutputHead(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self._vocab_size = vocab_size
        self._head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._head(x)


class LinearQOutputHead(nn.Module):
    def __init__(self, hidden_dim: int, seq_length: int) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self._seq_length = seq_length

        layers = []
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Linear(seq_length, 1))

        self._layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._layers[0](x)
        x = x.view(x.shape[0], -1)
        return self._layers[1](x)
