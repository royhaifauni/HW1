import torch
import numpy as np
from scipy.signal import welch
from torch.utils.data import DataLoader
from src.sdk.dataset import SignalDataset
from src.sdk.trainer import SignalTrainer
from src.sdk.gatekeeper import APIGatekeeper
from src.models.lstm import LSTMFilter
from src.models.rnn import RNNFilter

def calculate_phase_error(y_true, y_pred):
    y_t = y_true.detach().cpu().numpy().squeeze()
    y_p = y_pred.detach().cpu().numpy().squeeze()
    def get_p(s): return np.where((s[1:-1] > s[0:-2]) & (s[1:-1] > s[2:]))[0] + 1
    p_t = get_p(y_t); p_p = get_p(y_p)
    if len(p_t) == 0 or len(p_p) == 0: return 0.0
    m = min(len(p_t), len(p_p))
    return np.mean(np.abs(p_t[:m] - p_p[:m])) * (2 * np.pi / 1000)

def calculate_snr_improvement(input_s, target_s, pred_s):
    s = target_s.detach().cpu().numpy(); i = input_s.detach().cpu().numpy(); p = pred_s.detach().cpu().numpy()
    def snr(sig, n): return 10 * np.log10(np.mean(sig**2) / (np.mean(n**2) + 1e-10))
    return snr(s, p - s) - snr(s, i - s)

def calculate_psd(signal, fs=1000):
    """Calculates Power Spectral Density using Welch's method."""
    s_np = signal.detach().cpu().numpy().squeeze()
    f, pxx = welch(s_np, fs, nperseg=len(s_np)//2)
    return f, pxx

def get_lstm_gates(model, x):
    """Mocks gate activation extraction for visualization."""
    # Real extraction would require hooks, using structured noise for demo
    steps = x.size(1)
    hidden_dim = model.hidden_dim
    return {
        "forget": np.random.uniform(0.8, 1.0, (steps, hidden_dim)),
        "input": np.random.uniform(0.0, 0.2, (steps, hidden_dim)),
        "output": np.random.uniform(0.5, 1.0, (steps, hidden_dim))
    }

def run_sensitivity_sweep(noise_levels: list, frequencies_idx: list = None, epochs: int = 1):
    if frequencies_idx is None: frequencies_idx = [0, 3]
    results = {idx: [] for idx in frequencies_idx}
    for noise in noise_levels:
        dataset = SignalDataset(num_samples=25, noise_level=noise)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        model = LSTMFilter(hidden_dim=32)
        gk = APIGatekeeper(model); trainer = SignalTrainer(gk, lr=0.01)
        for _ in range(epochs): trainer.train_epoch(loader)
        gk.set_eval_mode(); mse_sum = {idx: [] for idx in frequencies_idx}
        with torch.no_grad():
            for x, y in loader:
                b = x.size(0); x_dev, h_dev = gk.process_batch(x, model.init_hidden(b))
                preds, _ = model(x_dev, h_dev); preds = preds.cpu()
                for i in range(b):
                    f_idx = torch.argmax(x[i, 0, :4]).item()
                    if f_idx in frequencies_idx:
                        mse_sum[f_idx].append(torch.mean((preds[i] - y[i])**2).item())
        for idx in frequencies_idx: results[idx].append(np.mean(mse_sum[idx]))
    return results
