# Project Ledger: Signal-Recurrence-Research

## Phase 1: Scaffold & Infrastructure (Tasks 1-50)
- [x] Task 1: Initialize project with `uv init`.
- [x] Task 2: Create directory structure (`docs/prd`, `docs/plan`, `docs/todo`, `src/sdk`, `src/models`, `tests`, `notebooks`).
- [x] Task 3: Create `.gitignore` with strict exclusion rules.
- [x] Task 4: Configure `pyproject.toml` with `ruff`.
- [x] Task 5: Configure `ruff` line length (88).
- [x] Task 6: Add `pytest` to development dependencies.
- [x] Task 7: Add `numpy` to project dependencies.
- [x] Task 8: Add `torch` (or preferred DL framework) to dependencies.
- [x] Task 9: Set up `.python-version` (3.12+).
- [x] Task 10: Create `src/sdk/__init__.py`.
- [x] Task 11: Create `src/models/__init__.py`.
- [x] Task 12: Create `tests/__init__.py`.
- [x] Task 13: Define `SAMPLE_RATE` constant in `src/sdk/constants.py`.
- [x] Task 14: Define `DURATION` constant in `src/sdk/constants.py`.
- [x] Task 15: Define `FREQUENCIES` constant (1, 3, 5, 7) in `src/sdk/constants.py`.
- [x] Task 16: Define `PHASE_RANGE` constant (0, 2*pi).
- [x] Task 17: Define `NOISE_LEVEL` constant.
- [x] Task 18: Initialize `docs/prd/main_prd.md`.
- [x] Task 19: Populate `main_prd.md` with project description.
- [x] Task 20: Define success metrics in `main_prd.md`.
- [x] Task 21: Initialize `docs/prd/signal_engine.md`.
- [x] Task 22: Define signal generation logic in `signal_engine.md`.
- [x] Task 23: Initialize `docs/prd/rnn_architecture.md`.
- [x] Task 24: Initialize `docs/prd/lstm_architecture.md`.
- [x] Task 25: Initialize `docs/plan/PLAN.md`.
- [x] Task 26: Define SDK-First design in `PLAN.md`.
- [x] Task 27: Define TDD workflow in `PLAN.md`.
- [ ] Task 28: Set up initial test file `tests/test_environment.py`.
- [ ] Task 29: Write test to verify `numpy` import.
- [ ] Task 30: Write test to verify `torch` (or framework) import.
- [x] Task 31: Run `uv sync` to build environment.
- [x] Task 32: Run `ruff check .` to verify initial linting.
- [x] Task 33: Verify folder structure completeness.
- [ ] Task 34: Document One-Hot mapping logic.
- [ ] Task 35: Task 35: Define `ONE_HOT_MAP` in `constants.py`.
- [ ] Task 36-50: (Detailed ruff and pytest configuration tasks).

## Phase 2: Signal Engine SDK (Tasks 51-150)
- [x] Task 51: Create `src/sdk/signal_utils.py`. (Merged into signal_gen.py)
- [x] Task 52: Define `generate_time_axis` function signature.
- [x] Task 53: Write unit test for `generate_time_axis`.
- [x] Task 54: Implement `generate_time_axis`.
- [x] Task 55: Verify `generate_time_axis` output shape.
- [x] Task 56: Define `generate_sine_wave` function signature.
- [x] Task 57: Write unit test for `generate_sine_wave` (frequency).
- [x] Task 58: Write unit test for `generate_sine_wave` (phase).
- [x] Task 59: Implement `generate_sine_wave`.
- [x] Task 60: Define `apply_noise` function signature.
- [x] Task 61: Write unit test for `apply_noise`.
- [x] Task 62: Implement `apply_noise`.
- [x] Task 63: Create `src/sdk/signal_gen.py`.
- [x] Task 64: Define `SignalGenerator` class.
- [x] Task 65: Implement `SignalGenerator.__init__` with constants.
- [x] Task 66: Define `SignalGenerator.generate_composite` method.
- [x] Task 67: Write test for composite signal shape.
- [x] Task 68: Implement logic to sum 4 sine waves.
- [x] Task 69: Implement unique phase jitter per frequency.
- [x] Task 70: Implement unique amplitude noise per frequency.
- [x] Task 71: Write test to ensure phase is random per call.
- [x] Task 72: Define `generate_one_hot` function.
- [x] Task 73: Write test for one-hot vector validity.
- [x] Task 74: Implement `generate_one_hot`.
- [x] Task 75: Create `src/sdk/dataset.py`.
- [x] Task 76: Implement `SignalDataset` class.
- [x] Task 77: Implement windowing logic in `SignalDataset`.
- [x] Task 78: Implement One-Hot concatenation in `__getitem__`.
- [x] Task 79: Write unit test for Dataset dimensions.
- [x] Task 80: Write unit test for One-Hot logic accuracy.
- [ ] Task 81-150: (Detailed data loader and normalization tasks).

## Phase 3: RNN & LSTM Core (Tasks 151-300)
- [ ] Task 151: Create `src/models/rnn_filter.py`.
- [ ] Task 152: Define `RNNFilter` hyperparameters in `constants.py`.
- [ ] Task 153: Implement `RNNFilter` constructor.
- [ ] Task 154: Implement `RNNFilter.forward`.
- [ ] Task 155: Create `src/models/lstm_filter.py`.
- [ ] Task 156: Define `LSTMFilter` hyperparameters.
- [ ] Task 157: Implement `LSTMFilter` constructor.
- [ ] Task 158: Implement `LSTMFilter.forward`.
- [ ] Task 159: Create `src/sdk/trainer.py`.
- [ ] Task 160: Implement `Trainer` class with SDK abstraction.
- [ ] Task 161-200: (Loss functions and optimization scheduling).
- [ ] Task 201-250: (Validation logic and early stopping implementation).
- [ ] Task 251-300: (Checkpointing and model persistence logic).

## Phase 4: Training & Evaluation (Tasks 301-450)
- [ ] Task 301: Initialize RNN training run.
- [ ] Task 302: Log training loss to console/file.
- [ ] Task 303: Monitor validation MSE.
- [ ] Task 304: Initialize LSTM training run.
- [ ] Task 305: Compare RNN vs LSTM convergence speed.
- [ ] Task 306-350: (Hyperparameter tuning for RNN window size).
- [ ] Task 351-400: (Hyperparameter tuning for LSTM hidden dimensions).
- [ ] Task 401-450: (Robustness testing against increased noise levels).

## Phase 5: Analysis & Reporting (Tasks 451-500)
- [ ] Task 451: Generate MSE distribution plots.
- [ ] Task 452: Visualize phase preservation for 1Hz signal.
- [ ] Task 453: Visualize phase preservation for 7Hz signal.
- [ ] Task 454: Plot ground truth vs prediction overlays.
- [ ] Task 455-480: (Detailed error analysis for each frequency component).
- [ ] Task 481-495: (Final PRD validation against success metrics).
- [ ] Task 496: Generate Final Report in `notebooks/`.
- [ ] Task 500: Final repository cleanup and documentation audit.
