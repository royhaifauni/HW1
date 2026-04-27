import torch
import torch.nn as nn
import torch.optim as optim
from .gatekeeper import APIGatekeeper

class SignalTrainer:
    """Orchestrates the training loop via the APIGatekeeper."""
    
    def __init__(self, gatekeeper: APIGatekeeper, lr: float = 0.001):
        self.gk = gatekeeper
        self.optimizer = optim.Adam(self.gk.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_epoch(self, dataloader) -> float:
        self.gk.set_train_mode()
        total_loss = 0.0
        
        for x, y in dataloader:
            # Note: Dataset returns (batch, 4 + window)
            # We need to reshape for RNN: (batch, window, 5)
            # For simplicity in this demo, we assume seq_len=1
            x = x.unsqueeze(1) # (batch, 1, 5)
            y = y.unsqueeze(2) # (batch, window, 1) -> Adjusting for simplicity
            
            self.optimizer.zero_grad()
            
            batch_size = x.size(0)
            hidden = self.gk.model.init_hidden(batch_size)
            x_dev, hidden_dev = self.gk.process_batch(x, hidden)
            
            outputs, _ = self.gk.model(x_dev, hidden_dev)
            
            # Temporary: Matching target shape for training logic validation
            loss = self.criterion(outputs[:, -1, :], y[:, -1, :])
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
