# Prompt Book: Signal-Recurrence-Research

This book documents the significant prompts and iterative improvements used during the development of the Signal-Recurrence-Research project.

## Prompt 1: Project Initialization
- **Role:** Senior Software Architect.
- **Goal:** Scaffold the project with Dr. Yoram Segal standards.
- **Key Outcome:** Established directory structure, granular PRDs, and the 500-task ledger.

## Prompt 2: Data Engineering SDK
- **Role:** Senior Data Engineer.
- **Goal:** Implement the Signal Generation logic and PyTorch Dataset.
- **Key Outcome:** Created `signal_gen.py` with unique phase jitter/noise and `dataset.py` with One-Hot integration.

## Prompt 3: Recurrent Architectures
- **Role:** Deep Learning Engineer.
- **Goal:** Build RNN/LSTM models and the API Gatekeeper.
- **Key Outcome:** Implemented abstract base class, specific recurrent models, and a centralized orchestration layer (`gatekeeper.py`).

## Prompt 4: Sensitivity Analysis & Research
- **Role:** Research Scientist.
- **Goal:** Execute Sensitivity Analysis and produce the Research Notebook.
- **Key Outcome:** Created `notebooks/results_analysis.ipynb` with LaTeX documentation of LSTM gates and Nyquist-Shannon theorem. Conducted a noise sweep and visualized frequency-dependent MSE.

## Iterative Improvement: Logic Validation (Audit)
- **Goal:** Verify compliance with code standards and logical validity.
- **Key Correction:** Identified and fixed a dimension mismatch in `SignalDataset` where the output was a flattened vector instead of a sequence of 5D features.
