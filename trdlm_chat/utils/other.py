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
