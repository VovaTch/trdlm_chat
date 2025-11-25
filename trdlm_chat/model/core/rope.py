import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Embedding (RoPE) based largely on Karpathy's implementation.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000) -> None:
        """
        Initializes the RoPE class

        Args:
            dim (int): dimension of the embedding
            max_seq_len (int): maximum sequence length
            base (int): frequence base of the rotation
        """
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(f"dim must be even, but is {dim}")

        self._cos, self._sin = self._precompute_embeddings(dim, max_seq_len, base)

    @staticmethod
    def _precompute_embeddings(
        dim: int, max_seq_len: int, base: int = 10000
    ) -> tuple[torch.Tensor, torch.Tensor]:
        channel_range = torch.arange(0, dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()  # keep them in bfloat16
        cos, sin = (
            cos[None, :, None, :],
            sin[None, :, None, :],
        )
        return cos, sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.ndim != 4:
            raise ValueError(
                f"Input dimension must be 4 for multi-head attention. Got {x.ndim}"
            )

        d = x.shape[3] // 2
        x1, x2 = x[..., :d], x[..., d:]
        y1 = x1 * self._cos.to(x.device) + x2 * self._sin.to(x.device)
        y2 = x1 * (-self._sin.to(x.device)) + x2 * self._cos.to(x.device)
        out = torch.cat([y1, y2], 3)
        out = out.to(x.dtype)
        return out
