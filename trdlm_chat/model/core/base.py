from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Core(ABC, nn.Module):
    @abstractmethod
    def forward(
        self, x: torch.Tensor | None, y: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @property
    @abstractmethod
    def hidden_dim(self) -> int: ...

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def y_init(self) -> torch.nn.Buffer: ...

    @property
    @abstractmethod
    def z_init(self) -> torch.nn.Buffer: ...

    @abstractmethod
    def get_muon_params(self) -> list[nn.Parameter]: ...

    @abstractmethod
    def get_adamw_params(self) -> list[nn.Parameter]: ...
