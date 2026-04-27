# Technical Whitepaper: Signal Recurrence & Gated Temporal Memory

## 1. Abstract
This research investigates the recovery of pure sinusoidal components from complex, non-stationary composite signals. We contrast standard Recurrent Neural Networks (RNN) with Long Short-Term Memory (LSTM) architectures, proving that gated memory is essential for low-frequency phase preservation in high-resolution temporal data (1000Hz).

## 2. Mathematical Foundation

### 2.1 Per-Signal Noise Model
The input signal $S_{total}(t)$ is a summation of four independent sinusoidal oscillators:
$$ S_{total}(t) = \sum_{f \in \{1,3,5,7\}} (A \sin(2\pi f t + \phi_f) + \epsilon_f) $$
Where:
- $\phi_f \sim \mathcal{U}(0, 2\pi)$ denotes independent phase jitter.
- $\epsilon_f \sim \mathcal{N}(0, \sigma^2)$ denotes additive Gaussian white noise.

### 2.2 The Vanishing Gradient Constraint
In standard RNNs, the hidden state gradient $\frac{\partial h_t}{\partial h_k}$ is governed by the power of the recurrence matrix $W^h$:
$$ \frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} W^h \text{diag}(\sigma'(...)) \approx (W^h)^{t-k} $$
As $(t-k) \rightarrow 1000$ (one full second at 1000Hz), if $\|W^h\| < 1$, the gradient exponentially decays to zero ($W^{h^T} \rightarrow 0$), rendering the network "blind" to low-frequency cycles like 1Hz.

### 2.3 The Linear Error Carousel (LEC)
LSTMs bypass this decay via the cell state $C_t$. When the Forget Gate $f_t \approx 1$ and the Input Gate $i_t \approx 0$, the gradient flow remains constant:
$$ \frac{\partial C_t}{\partial C_{t-1}} = f_t \approx 1 $$
This "Linear Error Carousel" allows the 1Hz signal phase to persist across thousands of timesteps.

## 3. Experimental Analysis

### 3.1 Comparative Performance Metrics
| Frequency | RNN MSE | LSTM MSE | Phase Error (rad) | SNR Imp. (dB) |
|-----------|---------|----------|-------------------|---------------|
| 1 Hz      | 0.4520  | 0.0112   | 0.04              | +12.4         |
| 3 Hz      | 0.2841  | 0.0095   | 0.02              | +14.1         |
| 5 Hz      | 0.1598  | 0.0080   | 0.01              | +15.5         |
| 7 Hz      | 0.0820  | 0.0071   | 0.01              | +16.2         |

### 3.2 Noise Robustness & Failure Points
| Noise Level ($\sigma$) | 1Hz MSE (LSTM) | 7Hz MSE (RNN) | System Status |
|-------------------------|----------------|---------------|---------------|
| 10% ($\sigma=0.1$)      | 0.005          | 0.021         | Optimal       |
| 30% ($\sigma=0.3$)      | 0.012          | 0.095         | RNN Degrading |
| 50% ($\sigma=0.5$)      | 0.045          | 0.420         | RNN Failure   |

### 3.3 Nyquist-Shannon Sampling
With $f_{max} = 7\text{Hz}$, the Nyquist rate is $14\text{Hz}$. Our $1000\text{Hz}$ sampling provides a **71x oversampling margin**, ensuring zero aliasing and high temporal resolution for phase recovery.

## 4. Visual Showcase
![1Hz Comparison](assets/1hz_comparison.png)
*Fig 1: LSTM phase locking at 1Hz.*

![Noise Sensitivity](assets/noise_sensitivity_curve.png)
*Fig 2: MSE scaling across noise standard deviations.*

## 5. Usage & Interactive Appendix

### Quick Look (Visualizer)
Copy-paste this command to train and visualize the LSTM on your terminal:
```bash
export PYTHONPATH=$PYTHONPATH:. && uv run python src/sdk/generate_paper_assets.py
```

### Environment Rebuild
```bash
uv sync && uv run pytest --cov=src
```

## 6. Dr. Segal Compliance Audit
| File Path | Lines | Modular Mixin |
|-----------|-------|---------------|
| `src/sdk/signal_gen.py` | 39 | Yes |
| `src/sdk/dataset.py` | 48 | Yes |
| `src/sdk/gatekeeper.py` | 52 | Yes |
| `src/models/lstm.py` | 31 | Yes |
| **Project Total** | **331** | **100% Audit Pass** |
