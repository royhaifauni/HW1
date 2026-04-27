# PRD: Signal Engine

## 1. Functional Specification
The Signal Engine is responsible for generating the training and evaluation datasets.

## 2. Signal Generation Logic
- **Base Components:** $s_f(t) = A \sin(2\pi f t + \phi)$.
- **Frequencies ($f$):** 1, 3, 5, 7 Hz.
- **Phase Jitter ($\phi$):** For each instance, $\phi_f \sim \text{Uniform}(0, 2\pi)$ for each $f$.
- **Noise:** Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ added to each $s_f$.
- **Composite Signal:** $S_{total}(t) = \sum_{f \in \{1, 3, 5, 7\}} (s_f(t) + \epsilon_f)$.

## 3. Data Format
- **Time Steps:** 10,000 samples (10s @ 1000Hz).
- **Control Vector ($C$):** 4D One-Hot vector.
  - $[1, 0, 0, 0] \rightarrow 1$ Hz
  - $[0, 1, 0, 0] \rightarrow 3$ Hz
  - $[0, 0, 1, 0] \rightarrow 5$ Hz
  - $[0, 0, 0, 1] \rightarrow 7$ Hz
- **Target:** The pure sine wave $s_f(t)$ without noise for the frequency indicated by $C$.
