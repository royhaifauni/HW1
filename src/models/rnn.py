import torch
import torch.nn as nn
from .base_model import RecurrentBase

class RNNFilter(RecurrentBase):
    """Simple RNN implementation for signal filtering."""
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__(input_dim, hidden_dim, output_dim)
        
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_dim)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> tuple:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, 5)
            hidden: Initial hidden state
        Returns:
            out: Predicted signal (batch, seq_len, 1)
            hidden: Next hidden state
        """
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden
