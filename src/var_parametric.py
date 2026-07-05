"""
Parametric (Variance-Covariance) VaR: assumes portfolio returns are normally
distributed. Portfolio variance is computed analytically from the full
covariance matrix of the individual assets (not just the std of the blended
portfolio return series), so the correlation structure between assets is
explicit and inspectable.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.data_loader import load_config, load_prices
from src.returns import simple_returns, compute_weights, portfolio_returns


def covariance_matrix(returns: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    sample = returns if window is None else returns.iloc[-window:]
    return sample.cov()


def correlation_matrix(returns: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    sample = returns if window is None else returns.iloc[-window:]
    return sample.corr()


def parametric_var(
    returns: pd.DataFrame,
    weights: pd.Series,
    confidence: float,
    window: int | None = None,
    include_mean: bool = True,
) -> float:
    """
    Analytical VaR under the normal-distribution assumption.
    portfolio_variance = w' * Sigma * w  (the actual "variance-covariance" step)
    """
    sample = returns if window is None else returns.iloc[-window:]
    aligned_weights = weights.reindex(sample.columns).values

    mu_assets = sample.mean().values
    mu_p = float(aligned_weights @ mu_assets) if include_mean else 0.0

    sigma = sample.cov().values  # Sigma, the covariance matrix
    var_p = float(aligned_weights @ sigma @ aligned_weights)  # w' Sigma w
    sigma_p = np.sqrt(var_p)

    alpha = 1 - confidence
    z = norm.ppf(alpha)  # negative number, e.g. -1.645 for 95%
    return -(mu_p + z * sigma_p)


def parametric_es(
    returns: pd.DataFrame,
    weights: pd.Series,
    confidence: float,
    window: int | None = None,
    include_mean: bool = True,
) -> float:
    """
    Closed-form Expected Shortfall under normality:
    ES = mu + sigma * phi(z_alpha) / alpha
    (phi = standard normal PDF). This is the analytical tail-average, the
    parametric counterpart to averaging the historical tail observations.
    """
    sample = returns if window is None else returns.iloc[-window:]
    aligned_weights = weights.reindex(sample.columns).values

    mu_assets = sample.mean().values
    mu_p = float(aligned_weights @ mu_assets) if include_mean else 0.0

    sigma = sample.cov().values
    var_p = float(aligned_weights @ sigma @ aligned_weights)
    sigma_p = np.sqrt(var_p)

    alpha = 1 - confidence
    z = norm.ppf(alpha)
    es = -(mu_p - sigma_p * norm.pdf(z) / alpha)
    return es


if __name__ == "__main__":
    config = load_config()
    prices = load_prices()
    rets = simple_returns(prices)
    weights = compute_weights(rets, scheme=config["portfolio"]["weighting_scheme"])
    port_rets = portfolio_returns(rets, weights)

    print("--- Correlation matrix (full sample) ---")
    print(correlation_matrix(rets).round(2))
    print()

    for window in config["var"]["lookback_windows"]:
        print(f"--- Lookback window: {window} trading days ---")
        for cl in config["var"]["confidence_levels"]:
            var = parametric_var(rets, weights, cl, window=window)
            es = parametric_es(rets, weights, cl, window=window)
            print(f"  Confidence {cl:.0%}: VaR = {var:.4%}   ES = {es:.4%}")

        # sanity check: parametric std-based VaR should roughly match the
        # empirical std of the blended portfolio series over the same window
        empirical_std = port_rets.iloc[-window:].std()
        print(f"  (sanity check) empirical portfolio std over window: {empirical_std:.4%}")
        print()