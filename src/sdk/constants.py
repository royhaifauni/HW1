import numpy as np

# Signal Parameters
SAMPLE_RATE = 1000
DURATION = 10.0
FREQUENCIES = [1, 3, 5, 7]
PHASE_RANGE = (0, 2 * np.pi)
NOISE_LEVEL = 0.1

# Frequency Components (Explicit s1-s4 as examples)
S1_FREQ = 1
S2_FREQ = 3
S3_FREQ = 5
S4_FREQ = 7

# One-Hot Mapping
ONE_HOT_MAP = {1: [1, 0, 0, 0], 3: [0, 1, 0, 0], 5: [0, 0, 1, 0], 7: [0, 0, 0, 1]}

# Training Hyperparameters (Scaffold)
WINDOW_SIZE = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
