import torch
from torch.utils.data import Dataset
import numpy as np
from .signal_gen import SignalGenerator


class SignalDataset(Dataset):
    """PyTorch-compatible dataset for signal recurrence tasks."""

    def __init__(
        self, num_samples: int = 100, window_size: int = 100, noise_level: float = None
    ):
        self.generator = SignalGenerator(noise_level=noise_level)
        self.window_size = window_size
        self.num_samples = num_samples
        self.data = []

        self._prepare_data()

    def _prepare_data(self):
        """Generates and slices data into window sequences."""
        for _ in range(self.num_samples):
            s_total, clean_comps, one_hot_base = self.generator.generate_composite()

            for i in range(4):
                c_vector = one_hot_base[i]
                target_clean = clean_comps[i]

                max_start = len(s_total) - self.window_size
                start = np.random.randint(0, max_start)
                end = start + self.window_size

                window_noisy = s_total[start:end]
                window_clean = target_clean[start:end]

                c_repeated = np.tile(c_vector, (self.window_size, 1))
                x_seq = np.column_stack([c_repeated, window_noisy])

                self.data.append(
                    {"input_seq": x_seq, "target_seq": window_clean.reshape(-1, 1)}
                )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        return (
            torch.tensor(item["input_seq"], dtype=torch.float32),
            torch.tensor(item["target_seq"], dtype=torch.float32),
        )
