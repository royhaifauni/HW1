import torch
import json
import os
from typing import Any
from src.models.base_model import RecurrentBase

class APIGatekeeper:
    """Manages model access, device orchestration, and rate limits."""
    
    def __init__(self, model: RecurrentBase):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Loads rate limits from versioned JSON config."""
        config_path = os.path.join(os.path.dirname(__file__), "config/rate_limits.json")
        with open(config_path, "r") as f:
            return json.load(f)

    def set_train_mode(self):
        """Enable training mode for the underlying model."""
        self.model.train()

    def set_eval_mode(self):
        """Enable evaluation mode for the underlying model."""
        self.model.eval()

    def run_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Executes a prediction safely through the gatekeeper."""
        x = x.to(self.device)
        batch_size = x.size(0)
        hidden = self.model.init_hidden(batch_size)
        
        if isinstance(hidden, tuple):
            hidden = tuple(h.to(self.device) for h in hidden)
        else:
            hidden = hidden.to(self.device)
            
        with torch.no_grad():
            out, _ = self.model(x, hidden)
        return out.cpu()

    def process_batch(self, x: torch.Tensor, hidden: Any) -> tuple:
        """Moves training batches and hidden states to the active device."""
        x = x.to(self.device)
        if isinstance(hidden, tuple):
            hidden = tuple(h.to(self.device) for h in hidden)
        else:
            hidden = hidden.to(self.device)
        return x, hidden
