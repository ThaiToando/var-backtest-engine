import pandas as pd
from src.returns import simple_returns, compute_weights, portfolio_returns


def test_simple_returns_basic():
    prices = pd.DataFrame({"A": [100, 110, 99], "B": [50, 55, 60.5]})
    rets = simple_returns(prices)
    assert rets.shape == (2, 2)
    assert abs(rets["A"].iloc[0] - 0.10) < 1e-9   # 100 -> 110 is +10%
    assert abs(rets["B"].iloc[1] - 0.10) < 1e-9   # 55 -> 60.5 is +10%


def test_compute_weights_equal():
    rets = pd.DataFrame({"A": [0.01, -0.02], "B": [0.03, 0.01], "C": [-0.01, 0.02]})
    w = compute_weights(rets, scheme="equal")
    assert abs(w.sum() - 1.0) < 1e-9
    assert all(abs(w - 1 / 3) < 1e-9)


def test_compute_weights_inverse_vol():
    # B is 10x more volatile than A -> A should get roughly 10x the weight
    rets = pd.DataFrame({"A": [0.001, -0.001, 0.001], "B": [0.05, -0.05, 0.05]})
    w = compute_weights(rets, scheme="inverse_vol")
    assert abs(w.sum() - 1.0) < 1e-9
    assert w["A"] > w["B"]


def test_portfolio_returns_matches_manual_calc():
    rets = pd.DataFrame({"A": [0.10, -0.10], "B": [0.02, 0.02]})
    weights = pd.Series({"A": 0.5, "B": 0.5})
    port = portfolio_returns(rets, weights)
    assert abs(port.iloc[0] - 0.06) < 1e-9   # 0.5*0.10 + 0.5*0.02 = 0.06
    assert abs(port.iloc[1] - (-0.04)) < 1e-9  # 0.5*-0.10 + 0.5*0.02 = -0.04