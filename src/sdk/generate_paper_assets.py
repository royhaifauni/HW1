import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.sdk.dataset import SignalDataset
from src.sdk.gatekeeper import APIGatekeeper
from src.models.lstm import LSTMFilter
from src.sdk.analysis_utils import run_sensitivity_sweep


def generate_paper_assets():
    os.makedirs("assets", exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Comparison Plots (1Hz and 7Hz)
    for freq_idx, freq_name in [(0, "1hz"), (3, "7hz")]:
        dataset = SignalDataset(num_samples=1, window_size=200, noise_level=0.3)
        # Force the specific frequency for this sample
        x, y = dataset[0]
        model = LSTMFilter(hidden_dim=32)
        gk = APIGatekeeper(model)
        pred = gk.run_inference(x.unsqueeze(0))

        plt.figure(figsize=(10, 5))
        plt.plot(x[:, 4].numpy(), label="Noisy Input", alpha=0.3, color="gray")
        plt.plot(y.numpy(), label="Target Pure", color="blue")
        plt.plot(pred[0].numpy(), label="Prediction", linestyle="--", color="red")
        plt.title(f"Signal Comparison: {freq_name.upper()}")
        plt.legend()
        plt.savefig(f"assets/{freq_name}_comparison.png")
        plt.close()

    # 2. Noise Sensitivity Curve
    noise_levels = [0.1, 0.3, 0.5]
    results = run_sensitivity_sweep(noise_levels, epochs=1)
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, results[0], marker="o", label="1Hz MSE")
    plt.plot(noise_levels, results[3], marker="s", label="7Hz MSE")
    plt.xlabel("Gaussian Noise Sigma")
    plt.ylabel("MSE")
    plt.title("Noise Sensitivity Analysis")
    plt.legend()
    plt.savefig("assets/noise_sensitivity_curve.png")
    plt.close()


if __name__ == "__main__":
    generate_paper_assets()
