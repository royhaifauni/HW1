import numpy as np
import torch
from src.sdk.signal_gen import SignalGenerator
from src.sdk.dataset import SignalDataset

def test_unique_phase_jitter():
    generator = SignalGenerator()
    _, noisy1 = generator.generate_component(1)
    _, noisy2 = generator.generate_component(1)
    assert not np.array_equal(noisy1, noisy2)

def test_dataset_output_dimensions():
    window_size = 50
    dataset = SignalDataset(num_samples=5, window_size=window_size)
    x, y = dataset[0]
    
    # x shape: (window_size, 5)
    assert x.shape == (50, 5)
    # y shape: (window_size, 1)
    assert y.shape == (50, 1)
    assert isinstance(x, torch.Tensor)

def test_one_hot_logic():
    dataset = SignalDataset(num_samples=1, window_size=10)
    # 4 frequencies * 1 sample = 4 entries
    assert len(dataset) == 4
    
    for i in range(4):
        x, _ = dataset[i]
        expected_c = torch.zeros(4)
        expected_c[i] = 1.0
        # Check first 4 columns (C-vector) of every timestep in the window
        assert torch.all(x[:, :4] == expected_c)
