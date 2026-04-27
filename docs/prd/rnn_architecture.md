# PRD: RNN Architecture

## 1. Overview
A basic Recurrent Neural Network (RNN) implementation designed for temporal signal filtering.

## 2. Model Structure
- **Input Layer:** Accepts $(C, S_{total})$ pair.
- **Hidden Layer:** Simple RNN cells with tanh/ReLU activation.
- **Output Layer:** Linear projection to single scalar value (predicted $s_f(t)$).
- **Sliding Window:** Input sequence length determined by the lowest frequency period to ensure sufficient context.

## 3. Training Objective
- Minimize MSE between the RNN output and the ground truth sine wave.
- Evaluate the impact of window size on extraction quality.
