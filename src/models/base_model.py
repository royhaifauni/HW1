import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class RecurrentBase(nn.Module, ABC):
    """Abstract base class for recurrent signal filters."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    @abstractmethod
    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """Initialize hidden states for the recurrent layer."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> tuple:
        """Forward pass through the network."""
        pass
