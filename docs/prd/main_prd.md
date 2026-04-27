# PRD: Signal-Recurrence-Research (Main)

## 1. Project Overview
The "Signal-Recurrence-Research" project aims to develop and evaluate tunable neural filters (RNN and LSTM) capable of extracting individual sine wave components from a composite, noisy signal.

## 2. Core Requirements
- **Input:** 4D One-Hot vector $C$ (indicating target frequency) + Summed signal $S_{total}$.
- **Signal Logic:**
  - Frequencies: 1, 3, 5, 7 Hz.
  - Sample Rate: 1000 Hz.
  - Duration: 10 seconds.
  - Phase Jitter: $0-2\pi$ applied uniquely to each frequency.
  - Amplitude Noise: Applied to each component before summation.
- **Models:**
  - Basic RNN (Sliding Window).
  - LSTM (Gated Memory).

## 3. Success Metrics
- Mean Squared Error (MSE) between extracted signal and ground truth pure sine wave.
- Phase preservation accuracy.

## 4. Architectural Standards
- SDK-First design.
- Red-Green-Refactor TDD workflow.
- File size limit: 150 lines per file in `src/`.
- Zero-tolerance for Ruff violations.
