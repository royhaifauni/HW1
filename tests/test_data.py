import numpy as np
import torch
from src.sdk.signal_gen import SignalGenerator
from src.sdk.dataset import SignalDataset

def test_unique_phase_jitter():
    generator = SignalGenerator()
    # Generate two components for the same frequency
    # They should have different phases (very high probability)
    _, noisy1 = generator.generate_component(1)
    _, noisy2 = generator.generate_component(1)
    
    assert not np.array_equal(noisy1, noisy2), "Phase jitter should be unique per call"

def test_composite_signal_shape():
    generator = SignalGenerator(sample_rate=1000, duration=1.0)
    s_total, clean_comps, one_hot = generator.generate_composite()
    
    assert s_total.shape == (1000,)
    assert clean_comps.shape == (4, 1000)
    assert one_hot.shape == (4, 4)

def test_dataset_output_dimensions():
    window_size = 50
    dataset = SignalDataset(num_samples=5, window_size=window_size)
    x, y = dataset[0]
    
    # Input x = 4 (One-Hot) + 50 (Window) = 54
    assert x.shape == (54,)
    # Target y = 50 (Window)
    assert y.shape == (50,)
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)

def test_one_hot_logic():
    dataset = SignalDataset(num_samples=1, window_size=10)
    # 1 sample generates 4 entries (one per frequency)
    assert len(dataset) == 4
    
    # Check each entry has a unique one-hot vector
    vectors = [dataset[i][0][:4] for i in range(4)]
    for i in range(4):
        expected = torch.zeros(4)
        expected[i] = 1.0
        assert torch.equal(vectors[i], expected)
