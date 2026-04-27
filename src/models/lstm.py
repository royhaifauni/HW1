import torch
import torch.nn as nn
from .base_model import RecurrentBase

class LSTMFilter(RecurrentBase):
    """LSTM implementation for signal filtering with gated memory."""
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__(input_dim, hidden_dim, output_dim)
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def init_hidden(self, batch_size: int) -> tuple:
        """LSTM requires a tuple of (h0, c0)."""
        h0 = torch.zeros(1, batch_size, self.hidden_dim)
        c0 = torch.zeros(1, batch_size, self.hidden_dim)
        return (h0, c0)

    def forward(self, x: torch.Tensor, hidden: tuple) -> tuple:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, 5)
            hidden: (h0, c0) state tuple
        Returns:
            out: Predicted signal (batch, seq_len, 1)
            hidden: (hn, cn) state tuple
        """
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden
