"""
Computes asset-level and portfolio-level returns from cached price data.

Uses simple (arithmetic) returns for portfolio aggregation because portfolio
return is the weight-averaged sum of constituent simple returns so this is
exact. Log returns are time-additive (useful for multi-day compounding) but
NOT exactly additive across assets in a portfolio so using them for
portfolio-level aggregation would introduce a (usually small, but avoidable)
approximation error. We compute log returns too, for reference/diagnostics.
"""
import numpy as np
import pandas as pd

from src.data_loader import load_config, get_tickers, load_prices


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="all")


def compute_weights(returns: pd.DataFrame, scheme: str = "equal") -> pd.Series:
    """
    Returns portfolio weights as a Series indexed by ticker, summing to 1.

    - "equal": 1/N per asset. Simplest, no estimation risk, but ignores that
      some assets (e.g. UNG) are far more volatile than others -- an equal
      dollar weight is NOT an equal risk contribution.
    - "inverse_vol": weight inversely proportional to each asset's historical
      volatility, so no single volatile asset dominates portfolio risk.
      Still ignores correlation (a true risk-parity scheme would use the
      full covariance matrix), but it's a meaningful step up from equal-weight
      and cheap to explain/defend.
    """
    n = returns.shape[1]
    if scheme == "equal":
        w = pd.Series(1.0 / n, index=returns.columns)
    elif scheme == "inverse_vol":
        vol = returns.std()
        inv_vol = 1.0 / vol
        w = inv_vol / inv_vol.sum()
    else:
        raise ValueError(f"Unknown weighting_scheme: {scheme}")
    return w


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Weighted sum of asset-level simple returns = exact portfolio simple return."""
    aligned_weights = weights.reindex(returns.columns)
    return returns.dot(aligned_weights)


def summarize_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Per-asset descriptive stats -- this is where the fat-tails story starts
    to show up numerically (skew/kurtosis), before we even build a VaR model.
    """
    summary = pd.DataFrame({
        "mean": returns.mean(),
        "std": returns.std(),
        "skew": returns.skew(),
        "kurtosis": returns.kurtosis(),  # pandas reports EXCESS kurtosis (normal = 0)
        "min": returns.min(),
        "max": returns.max(),
    })
    return summary


if __name__ == "__main__":
    config = load_config()
    prices = load_prices()

    rets = simple_returns(prices)
    weights = compute_weights(rets, scheme=config["portfolio"]["weighting_scheme"])
    port_rets = portfolio_returns(rets, weights)

    print("Weights:\n", weights, "\n")
    print("Per-asset return stats:\n", summarize_returns(rets), "\n")
    print("Portfolio return stats:\n")
    print(portfolio_returns(rets, weights).describe())