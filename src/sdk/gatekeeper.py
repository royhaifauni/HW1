import torch
from typing import Any
from src.models.base_model import RecurrentBase

class APIGatekeeper:
    """Manages model access, device orchestration, and batch execution."""
    
    def __init__(self, model: RecurrentBase):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

    def set_train_mode(self):
        self.model.train()

    def set_eval_mode(self):
        self.model.eval()

    def run_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Executes a prediction safely through the gatekeeper."""
        x = x.to(self.device)
        batch_size = x.size(0)
        hidden = self.model.init_hidden(batch_size)
        
        # Handle tuple hidden states (LSTM) vs single tensor (RNN)
        if isinstance(hidden, tuple):
            hidden = tuple(h.to(self.device) for h in hidden)
        else:
            hidden = hidden.to(self.device)
            
        with torch.no_grad():
            out, _ = self.model(x, hidden)
        return out.cpu()

    def process_batch(self, x: torch.Tensor, hidden: Any) -> tuple:
        """Internal helper for training loops to move data to device."""
        x = x.to(self.device)
        if isinstance(hidden, tuple):
            hidden = tuple(h.to(self.device) for h in hidden)
        else:
            hidden = hidden.to(self.device)
        return x, hidden
