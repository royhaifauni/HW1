import torch
from torch.utils.data import Dataset
import numpy as np
from .signal_gen import SignalGenerator

class SignalDataset(Dataset):
    """PyTorch-compatible dataset for signal recurrence tasks."""
    
    def __init__(self, num_samples: int = 100, window_size: int = 100):
        self.generator = SignalGenerator()
        self.window_size = window_size
        self.num_samples = num_samples
        self.data = []
        
        self._prepare_data()

    def _prepare_data(self):
        """Generates and slices data into windows."""
        for _ in range(self.num_samples):
            s_total, clean_comps, one_hot_base = self.generator.generate_composite()
            
            # For each frequency component (4)
            for i in range(4):
                c_vector = one_hot_base[i]
                target_clean = clean_comps[i]
                
                # Randomly sample one window from the 10s signal
                max_start = len(s_total) - self.window_size
                start = np.random.randint(0, max_start)
                end = start + self.window_size
                
                window_noisy = s_total[start:end]
                window_clean = target_clean[start:end]
                
                self.data.append({
                    "input_signal": window_noisy,
                    "c_vector": c_vector,
                    "target": window_clean
                })

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        
        # Concatenate C (4D) with Signal (Window Size)
        # Input shape: (4 + window_size,)
        x = np.concatenate([item["c_vector"], item["input_signal"]])
        y = item["target"]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
