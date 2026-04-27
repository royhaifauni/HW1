import pytest
from src.sdk.analysis_utils import run_sensitivity_sweep

def test_run_sensitivity_sweep():
    noise_levels = [0.1, 0.2]
    # Small sweep for testing
    results = run_sensitivity_sweep(noise_levels, epochs=1)
    
    assert 0 in results
    assert 3 in results
    assert len(results[0]) == 2
    assert len(results[3]) == 2
    assert all(isinstance(val, float) for val in results[0])
