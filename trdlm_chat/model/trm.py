import torch
import torch.nn as nn

from .core.base import Core


class TinyRecursiveModel(nn.Module):
    """
    General TRM implementation, with support for latent recursion, deep recursion, and a general forward function
    """

    def __init__(
        self,
        core: Core,
        z_loop: int,
        y_loop: int,
        input_embedding: nn.Embedding,
        output_head: nn.Module,
        q_head: nn.Module,
    ) -> None:
        """
        Initializer

        Args:
            core (Core): core module
            z_loop (int): number of latent reasoning steps
            y_loop (int): number of deep recursion steps
            input_embedding (nn.Module): input embedding module
            output_head (nn.Module): output head from the network output
            q_head (nn.Module): network cut-off head
        """
        super().__init__()
        self._core = core
        self._z_loop = z_loop
        self._y_loop = y_loop
        self._input_embedding = input_embedding
        self._output_head = output_head
        self._q_head = q_head

    @property
    def core(self) -> Core:
        """
        Returns the core underlying model of the TRM.
        """
        return self._core

    @property
    def input_embedding(self) -> nn.Embedding:
        return self._input_embedding

    @property
    def output_head(self) -> nn.Module:
        return self._output_head

    @property
    def q_head(self) -> nn.Module:
        return self._q_head

    def latent_recursion(
        self, input: torch.Tensor, output: torch.Tensor, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Latent recursion method of the TRM.

        Args:
            input (torch.Tensor): Input tensor
            output (torch.Tensor): Output tensor
            latent (torch.Tensor): Latent tensor

        Returns:
            tuple: Tuple containing the output tensor and latent tensor
        """
        for _ in range(self._z_loop):
            _, latent = self._core(input, output, latent)
        output, _ = self._core(None, output, latent)
        return output, latent

    def deep_recursion(
        self, input: torch.Tensor, output: torch.Tensor, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Deep recursion method of the TRM.

        Args:
            input (torch.Tensor): Input tensor
            output (torch.Tensor): Output tensor
            latent (torch.Tensor): Latent tensor

        Returns:
            tuple: Tuple containing the output tensor, latent tensor, output head tensor, and stop tensor
        """
        input = self._input_embedding(input)
        with torch.no_grad():
            for _ in range(self._y_loop - 1):
                output, latent = self.latent_recursion(input, output, latent)
        output, latent = self.latent_recursion(input, output, latent)
        return (
            output.detach(),
            latent.detach(),
            self._output_head(output),
            self._q_head(output),
        )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward method of the TRM.

        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Output tensor
            z (torch.Tensor): Latent tensor

        Returns:
            tuple: Tuple containing the output tensor and latent tensor
        """
        _, _, output, latent = self.deep_recursion(x, y, z)
        return output, latent
