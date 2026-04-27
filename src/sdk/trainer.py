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
            # x shape: (batch, seq_len, 5)
            # y shape: (batch, seq_len, 1)
            self.optimizer.zero_grad()
            
            batch_size = x.size(0)
            hidden = self.gk.model.init_hidden(batch_size)
            x_dev, hidden_dev = self.gk.process_batch(x, hidden)
            y_dev = y.to(self.gk.device)
            
            outputs, _ = self.gk.model(x_dev, hidden_dev)
            
            loss = self.criterion(outputs, y_dev)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
