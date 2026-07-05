import numpy as np
import pandas as pd
from src.var_historical import historical_var, historical_es
from src.var_parametric import parametric_var


def test_historical_var_matches_manual_quantile():
    # 100 evenly spaced returns from -0.10 to +0.09; 5th percentile is easy to hand-check
    returns = pd.Series(np.linspace(-0.10, 0.09, 100))
    var_95 = historical_var(returns, confidence=0.95)
    expected = -returns.quantile(0.05)
    assert abs(var_95 - expected) < 1e-9


def test_historical_var_increases_with_confidence():
    np.random.seed(0)
    returns = pd.Series(np.random.normal(0, 0.01, 1000))
    var_95 = historical_var(returns, confidence=0.95)
    var_99 = historical_var(returns, confidence=0.99)
    assert var_99 > var_95  # a stricter confidence level must imply a larger loss threshold


def test_historical_es_worse_than_var():
    np.random.seed(0)
    returns = pd.Series(np.random.normal(0, 0.01, 1000))
    var_95 = historical_var(returns, confidence=0.95)
    es_95 = historical_es(returns, confidence=0.95)
    assert es_95 >= var_95  # Expected Shortfall must be at least as bad as VaR by definition


def test_parametric_var_positive_and_increasing():
    np.random.seed(1)
    n = 500
    rets = pd.DataFrame({
        "A": np.random.normal(0, 0.01, n),
        "B": np.random.normal(0, 0.015, n),
    })
    weights = pd.Series({"A": 0.5, "B": 0.5})
    var_95 = parametric_var(rets, weights, confidence=0.95)
    var_99 = parametric_var(rets, weights, confidence=0.99)
    assert var_95 > 0
    assert var_99 > var_95