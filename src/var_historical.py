"""
Historical Simulation VaR: makes no distributional assumption. Uses the
empirical distribution of actual historical portfolio returns and reads off
the loss at the desired percentile.

Convention used throughout this project: VaR is reported as a POSITIVE number
representing the magnitude of loss (e.g. VaR_95 = 0.023 means "a 2.3% loss",
not -2.3%). 
"""
import pandas as pd

from src.data_loader import load_config, load_prices
from src.returns import simple_returns, compute_weights, portfolio_returns


def historical_var(returns: pd.Series, confidence: float, window: int | None = None) -> float:
    """
    Computes historical simulation VaR at the given confidence level.

    - window=None uses the full return series provided.
    - window=N uses only the most recent N observations (for the rolling
      backtest in a later step, the caller will pass in a sliced window,
      so this function itself stays simple and reusable).
    """
    sample = returns if window is None else returns.iloc[-window:]
    alpha = 1 - confidence
    quantile = sample.quantile(alpha)  # e.g. the 5th percentile for 95% confidence
    return -quantile  # flip sign: quantile is a negative return (loss), VaR is reported positive


def historical_es(returns: pd.Series, confidence: float, window: int | None = None) -> float:
    """
    Expected Shortfall (CVaR): the AVERAGE loss on days that breached VaR.
    Answers "given that we're in the tail, how bad is it on average?" --
    which VaR alone cannot tell you. We're building this now since it's a
    two-line addition on top of historical VaR, though we'll discuss ES
    properly (and why regulators moved to it) once all three VaR methods exist.
    """
    sample = returns if window is None else returns.iloc[-window:]
    var = historical_var(sample, confidence)
    tail_losses = sample[sample <= -var]  # days at least as bad as the VaR threshold
    return -tail_losses.mean()


if __name__ == "__main__":
    config = load_config()
    prices = load_prices()
    rets = simple_returns(prices)
    weights = compute_weights(rets, scheme=config["portfolio"]["weighting_scheme"])
    port_rets = portfolio_returns(rets, weights)

    print(f"Portfolio has {len(port_rets)} days of returns.\n")

    for window in config["var"]["lookback_windows"]:
        print(f"--- Lookback window: {window} trading days ---")
        for cl in config["var"]["confidence_levels"]:
            var = historical_var(port_rets, cl, window=window)
            es = historical_es(port_rets, cl, window=window)
            print(f"  Confidence {cl:.0%}: VaR = {var:.4%}   ES (CVaR) = {es:.4%}")
        print()