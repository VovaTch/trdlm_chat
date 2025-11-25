import os
import torch


def rms_norm(
    hidden_states: torch.Tensor, variance_epsilon: float = 1e-9
) -> torch.Tensor:
    """
    RMS Norm implementation from the TRM repo.

    Args:
        hidden_states (torch.Tensor): Input tensor
        variance_epsilon (float): Epsilon value for numerical stability

    Returns:
        torch.Tensor: Normalized tensor
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


def get_token_bytes(path_dir: str, device: str = "cpu") -> torch.Tensor:
    tokenizer_dir = os.path.join(path_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(
        token_bytes_path
    ), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes
