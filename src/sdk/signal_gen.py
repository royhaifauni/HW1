import numpy as np
from typing import Tuple

class SignalGenerator:
    """Generates composite signals with unique noise and phase jitter per frequency."""
    
    def __init__(self, sample_rate: int = 1000, duration: float = 10.0):
        self.fs = sample_rate
        self.duration = duration
        self.t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        self.frequencies = [1, 3, 5, 7]

    def generate_component(self, freq: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generates a single sine wave with unique phase jitter and amplitude noise."""
        phase_jitter = np.random.uniform(0, 2 * np.pi)
        clean_signal = np.sin(2 * np.pi * freq * self.t + phase_jitter)
        
        # Per-signal amplitude noise (Gaussian)
        noise = np.random.normal(0, 0.1, size=self.t.shape)
        noisy_signal = clean_signal + noise
        
        return clean_signal, noisy_signal

    def generate_composite(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates the total summed signal and individual clean components.
        
        Returns:
            S_total: The noisy summed signal.
            clean_components: Array of shape (4, N) containing pure sine waves.
            one_hot_base: Base one-hot identity for the 4 frequencies.
        """
        s_total = np.zeros_like(self.t)
        clean_components = []
        
        for freq in self.frequencies:
            clean, noisy = self.generate_component(freq)
            s_total += noisy
            clean_components.append(clean)
            
        return s_total, np.array(clean_components), np.eye(len(self.frequencies))
