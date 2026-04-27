import torch
import numpy as np
from torch.utils.data import DataLoader
from src.sdk.dataset import SignalDataset
from src.sdk.trainer import SignalTrainer
from src.sdk.gatekeeper import APIGatekeeper
from src.models.lstm import LSTMFilter
from src.models.rnn import RNNFilter

def calculate_phase_error(y_true, y_pred):
    """Calculates average peak displacement in radians."""
    y_t = y_true.detach().cpu().numpy().squeeze()
    y_p = y_pred.detach().cpu().numpy().squeeze()
    
    def get_peaks(s):
        return np.where((s[1:-1] > s[0:-2]) & (s[1:-1] > s[2:]))[0] + 1
    
    p_t = get_peaks(y_t)
    p_p = get_peaks(y_p)
    
    if len(p_t) == 0 or len(p_p) == 0:
        return 0.0
    
    m_len = min(len(p_t), len(p_p))
    return np.mean(np.abs(p_t[:m_len] - p_p[:m_len])) * (2 * np.pi / 1000)

def calculate_snr_improvement(input_signal, target_signal, predicted_signal):
    """Calculates SNR improvement in dB."""
    s = target_signal.detach().cpu().numpy()
    i = input_signal.detach().cpu().numpy()
    p = predicted_signal.detach().cpu().numpy()
    
    def get_snr(sig, n):
        return 10 * np.log10(np.mean(sig**2) / (np.mean(n**2) + 1e-10))
    
    snr_in = get_snr(s, i - s)
    snr_out = get_snr(s, p - s)
    return snr_out - snr_in

def run_sensitivity_sweep(noise_levels: list, frequencies_idx: list = None, epochs: int = 1):
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
                    f_idx = torch.argmax(x[i, 0, :4]).item()
                    if f_idx in frequencies_idx:
                        mse = torch.mean((preds[i] - y[i])**2).item()
                        mse_sum[f_idx].append(mse)
        for idx in frequencies_idx:
            results[idx].append(np.mean(mse_sum[idx]))
    return results
