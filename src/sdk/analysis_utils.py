import torch
import numpy as np
from torch.utils.data import DataLoader
from src.sdk.dataset import SignalDataset
from src.sdk.trainer import SignalTrainer
from src.sdk.gatekeeper import APIGatekeeper
from src.models.lstm import LSTMFilter

def run_sensitivity_sweep(noise_levels: list, frequencies_idx: list = None, epochs: int = 1):
    """
    Sweeps noise levels and evaluates MSE for target frequencies.
    frequencies_idx: Defaults to [0, 3] (1Hz and 7Hz).
    """
    if frequencies_idx is None:
        frequencies_idx = [0, 3]
        
    results = {idx: [] for idx in frequencies_idx}
    
    for noise in noise_levels:
        dataset = SignalDataset(num_samples=25, noise_level=noise)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        model = LSTMFilter(hidden_dim=32)
        gk = APIGatekeeper(model)
        trainer = SignalTrainer(gk, lr=0.01)
        
        for _ in range(epochs):
            trainer.train_epoch(loader)
            
        gk.set_eval_mode()
        mse_sum = {idx: [] for idx in frequencies_idx}
        
        with torch.no_grad():
            for x, y in loader:
                batch_size = x.size(0)
                hidden = model.init_hidden(batch_size)
                x_dev, h_dev = gk.process_batch(x, hidden)
                preds, _ = model(x_dev, h_dev)
                preds = preds.cpu()
                
                for i in range(batch_size):
                    freq_idx = torch.argmax(x[i, 0, :4]).item()
                    if freq_idx in frequencies_idx:
                        mse = torch.mean((preds[i] - y[i])**2).item()
                        mse_sum[freq_idx].append(mse)
                        
        for idx in frequencies_idx:
            results[idx].append(np.mean(mse_sum[idx]))
            
    return results
