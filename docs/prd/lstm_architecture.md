# PRD: LSTM Architecture

## 1. Overview
A Long Short-Term Memory (LSTM) network designed to handle long-range temporal dependencies in signal recurrence.

## 2. Model Structure
- **Input Layer:** Accepts $(C, S_{total})$ pair.
- **Memory Cells:** LSTM units with input, forget, and output gates.
- **State Management:** Evaluation of hidden and cell states over the 10s duration.
- **Output Layer:** Linear projection to single scalar value.

## 3. Advantages over RNN
- Expected to better preserve phase consistency over longer sequences.
- Improved handling of low-frequency components (1Hz).
