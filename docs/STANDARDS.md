# International Standards Compliance

## 1. ISO/IEC 25010: Product Quality Model

- **Maintainability:** Modular architecture with a 150-line file limit and strict OOP principles (Base classes + SDK).
- **Reliability:** Comprehensive TDD with a mandatory 85% coverage threshold and zero-tolerance for Ruff violations.
- **Portability:** Dependency management via `uv` ensures a clean rebuild of the environment on any machine.
- **Functional Suitability:** PRD-driven development ensures all signal extraction requirements are met.

## 2. Nielsen’s 10 Heuristics (Applied to SDK/CLI)

- **Visibility of system status:** The `APIGatekeeper` and `Trainer` provide clear feedback on training progress and device status.
- **Match between system and the real world:** Signal generation logic follows physical laws of sine waves and Gaussian noise.
- **User control and freedom:** Modular SDK allows users to swap models (RNN vs. LSTM) and tuning parameters easily.
- **Consistency and standards:** Strict adherence to PEP8 and Ruff formatting ensures a uniform coding style across the project.
- **Error prevention:** Centralized `APIGatekeeper` manages batch sizes and device memory to prevent runtime crashes.
