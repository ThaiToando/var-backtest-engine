import numpy as np
from src.backtest import kupiec_test, christoffersen_independence_test


def test_kupiec_passes_when_rate_matches_expected():
    # Exactly 5 exceptions in 100 days = 5%, matching a 95% confidence model precisely
    exceptions = np.array([1] * 5 + [0] * 95)
    result = kupiec_test(exceptions, confidence=0.95)
    assert result["observed_rate"] == 0.05
    assert not result["reject_model"]


def test_kupiec_rejects_when_rate_way_off():
    # 30 exceptions in 100 days is way more than the 5% a 95% model should produce
    exceptions = np.array([1] * 30 + [0] * 70)
    result = kupiec_test(exceptions, confidence=0.95)
    assert result["reject_model"]


def test_christoffersen_flags_clustering():
    # All exceptions bunched together at the start -> should be flagged as clustered
    clustered = np.array([1, 1, 1, 1, 1] + [0] * 95)
    result = christoffersen_independence_test(clustered)
    assert result["reject_independence"]


def test_christoffersen_passes_when_spread_evenly():
    # Same total exception count as above, but spread out -> should NOT be flagged
    n = 100
    spread = np.zeros(n, dtype=int)
    spread[::20] = 1  # every 20th day
    result = christoffersen_independence_test(spread)
    assert not result["reject_independence"]