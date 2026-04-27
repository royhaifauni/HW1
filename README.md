# Signal-Recurrence-Research: User Manual

## 1. Overview
The **Signal-Recurrence-Research** project is a tunable neural filtering system designed to extract pure sine waves from complex, noisy composite signals. Following the **Dr. Yoram Segal** professional standards, it features an SDK-First architecture, strict modularity, and high-precision scientific validation.

## 2. Quick Start

### Installation
Ensure you have `uv` installed. Synchronize the environment:
```bash
uv sync
```

### Signal Generation
Generate 10 seconds of 1000Hz composite data (1, 3, 5, 7 Hz):
```python
from src.sdk.signal_gen import SignalGenerator
generator = SignalGenerator(noise_level=0.1)
s_total, clean_comps, one_hot = generator.generate_composite()
```

### Model Training
Train an LSTM filter via the `APIGatekeeper`:
```python
from src.models.lstm import LSTMFilter
from src.sdk.gatekeeper import APIGatekeeper
from src.sdk.trainer import SignalTrainer

model = LSTMFilter(hidden_dim=64)
gk = APIGatekeeper(model)
trainer = SignalTrainer(gk, lr=0.001)
# Use PyTorch DataLoader with SignalDataset
```

## 3. Architecture
- **SDK Layer:** All business logic is encapsulated in `src/sdk/`.
- **API Gatekeeper:** Centralized management of device orchestration and rate limits (loaded from `rate_limits.json`).
- **Recurrent Models:** Modular RNN and LSTM implementations using an abstract base class.
- **TDD Workflow:** 85%+ test coverage enforced via `pytest-cov`.

## 4. Scientific Research
Detailed sensitivity analysis and theoretical frameworks (Nyquist-Shannon, LSTM gate mechanics) are available in the Research Notebook:
`notebooks/results_analysis.ipynb`

## 5. Development Standards
- **Linter:** Zero Ruff violations.
- **Modularity:** Files limited to 150 lines.
- **Audit:** Full task tracking in `docs/todo/TODO.md` (500 micro-tasks).
