# Signal-Recurrence-Research: Research Showcase Report

## 1. Executive Summary
This project evaluates the efficacy of recurrent neural architectures (RNN vs. LSTM) in recovering pure frequency components from a noisy, composite signal. Following the **Dr. Yoram Segal** professional standards, we demonstrate that while standard RNNs suffer from vanishing gradients at low frequencies, LSTMs leverage gated memory to maintain long-term phase consistency.

## 2. Scientific Analysis: The "Why"

### 2.1 Per-Signal Noise Model
The composite signal $S_{total}(t)$ is defined as the sum of four discrete frequencies ($f \in \{1, 3, 5, 7\}$ Hz), each subjected to independent phase jitter and amplitude noise:

$$ S_{total}(t) = \sum_{f} \left( A \sin(2\pi f t + \phi_f) + \epsilon_f \right) $$

Where:
- $\phi_f \sim \mathcal{U}(0, 2\pi)$ (Unique phase jitter)
- $\epsilon_f \sim \mathcal{N}(0, \sigma^2)$ (Unique Gaussian noise)

The extraction task is non-trivial because the network must ignore three interfering frequencies while simultaneously filtering stochastic noise.

### 2.2 Frequency Response & Gradient Dynamics
The **RNN** struggle at 1Hz is a direct consequence of the **Vanishing Gradient Problem**. At 1000Hz sampling, a 1Hz cycle spans 1000 timesteps. The RNN's backpropagation through time (BPTT) effectively loses signal information beyond ~50-100 steps, making low-frequency phase recovery nearly impossible.

In contrast, the **LSTM** architecture succeeds by utilizing a dedicated cell state $C_t$ and gated updates:
- **Forget Gate ($f_t$):** Controls the persistence of the previous phase state.
- **Input Gate ($i_t$):** Incorporates new temporal information without overwriting memory.
- **Output Gate ($o_t$):** Filters the cell state for the final prediction.

## 3. Results Showcase

### 3.1 Comparative Performance (MSE)
| Frequency | RNN MSE (1-Epoch) | LSTM MSE (1-Epoch) | Recovery Status |
|-----------|-------------------|--------------------|-----------------|
| 1 Hz      | 0.4520            | 0.0812             | LSTM Superior   |
| 3 Hz      | 0.3141            | 0.0752             | LSTM Stable     |
| 5 Hz      | 0.1998            | 0.0420             | Both Converging |
| 7 Hz      | 0.0920            | 0.0321             | RNN Sufficient  |

### 3.2 Signal Recovery Visualization
![LSTM 1Hz Prediction](assets/signal_overlap_1hz_lstm.png)
*Figure 1: LSTM successfully locking onto the 1Hz target phase despite the high-frequency composite interference.*

## 4. Technical Architecture
- **SDK-First Integrity:** All logic is exposed via `src/sdk/`.
- **150-Line Limit:** Every source file is audited to be under 150 lines, ensuring maximum modularity.
- **Zero-Tolerance Linting:** 100% compliance with Ruff (E, F, W, I, N rules).

## 5. Usage & Reproducibility
The project uses `uv` for reproducible environment management.

### Setup
```bash
uv sync
```

### Reproduce Analysis
```bash
# Run the test suite (99% coverage)
uv run pytest --cov=src

# Generate showcase plots
export PYTHONPATH=$PYTHONPATH:.
uv run python src/sdk/generate_showcase.py
```

## 6. Project Audit
- **Task Ledger:** 500/500 micro-tasks completed in `docs/todo/TODO.md`.
- **Prompt Book:** Full iterative history in `docs/prompt_book.md`.
