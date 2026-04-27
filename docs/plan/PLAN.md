# Project Plan: Signal-Recurrence-Research

## 1. SDK-First Design
The project will prioritize the creation of a robust SDK (`src/sdk/`) that abstracts signal generation, preprocessing, and model interfaces. This ensures:
- Model-agnostic data pipelines.
- Reusable components for both RNN and LSTM experiments.
- Easy integration into notebooks for visualization.

## 2. Red-Green-Refactor Workflow (TDD)
We strictly adhere to Test-Driven Development:
1. **Red:** Write a failing test in `tests/` for a specific micro-task.
2. **Green:** Implement the minimal code in `src/` to pass the test.
3. **Refactor:** Clean the code while ensuring tests remain green, maintaining modularity and adhering to the 150-line file limit.

## 3. Implementation Phases
- **Phase 1:** Scaffold, PRDs, and environment setup (Current).
- **Phase 2:** Signal Engine implementation and validation.
- **Phase 3:** RNN Architecture implementation and training.
- **Phase 4:** LSTM Architecture implementation and training.
- **Phase 5:** Comparative analysis and final reporting.
