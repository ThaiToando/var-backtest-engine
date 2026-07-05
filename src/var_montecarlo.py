"""
Monte Carlo VaR: simulate thousands of hypothetical portfolio return
outcomes by drawing correlated random returns from a fitted distribution,
then read VaR/ES off the simulated distribution -- same mechanics as
Historical Simulation, but on synthetic data instead of actual history.

Distribution choice matters a lot here:
- "normal": multivariate normal draws, calibrated to the covariance matrix.
  This is essentially Parametric VaR reproduced via simulation -- same
  normality assumption, same blind spot to fat tails, just noisier.
- "t": multivariate Student-t draws, same covariance/correlation structure
  but heavier tails (controlled by degrees_of_freedom). This is the version
  that can actually correct Parametric VaR's fat-tail underestimation while
  still respecting cross-asset correlation, unlike simply patching one
  asset's marginal distribution in isolation.
"""
import numpy as np
import pandas as pd

from src.data_loader import load_config, load_prices
from src.returns import simple_returns, compute_weights, portfolio_returns


def _simulate_asset_returns(
    mean: np.ndarray,
    cov: np.ndarray,
    num_simulations: int,
    distribution: str,
    degrees_of_freedom: int,
    random_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)

    if distribution == "normal":
        return rng.multivariate_normal(mean, cov, size=num_simulations)

    elif distribution == "t":
        # Standard construction of multivariate Student-t:
        # X = mean + Z / sqrt(W/df), where Z ~ N(0, cov), W ~ chi-squared(df)
        z = rng.multivariate_normal(np.zeros_like(mean), cov, size=num_simulations)
        w = rng.chisquare(degrees_of_freedom, size=num_simulations)
        scaling = np.sqrt(degrees_of_freedom / w)
        return mean + z * scaling[:, None]

    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def monte_carlo_var_es(
    returns: pd.DataFrame,
    weights: pd.Series,
    confidence: float,
    num_simulations: int,
    window: int | None = None,
    distribution: str = "t",
    degrees_of_freedom: int = 5,
    random_seed: int = 42,
) -> tuple[float, float]:
    sample = returns if window is None else returns.iloc[-window:]
    aligned_weights = weights.reindex(sample.columns).values

    mean = sample.mean().values
    cov = sample.cov().values

    sim_asset_returns = _simulate_asset_returns(
        mean, cov, num_simulations, distribution, degrees_of_freedom, random_seed
    )
    sim_portfolio_returns = sim_asset_returns @ aligned_weights

    alpha = 1 - confidence
    var = -np.quantile(sim_portfolio_returns, alpha)
    tail_losses = sim_portfolio_returns[sim_portfolio_returns <= -var]
    es = -tail_losses.mean()
    return var, es


if __name__ == "__main__":
    config = load_config()
    prices = load_prices()
    rets = simple_returns(prices)
    weights = compute_weights(rets, scheme=config["portfolio"]["weighting_scheme"])

    mc_cfg = config["var"]["monte_carlo"]

    for window in config["var"]["lookback_windows"]:
        print(f"--- Lookback window: {window} trading days ---")
        for distribution in ["normal", "t"]:
            print(f"  Distribution: {distribution}")
            for cl in config["var"]["confidence_levels"]:
                var, es = monte_carlo_var_es(
                    rets, weights, cl,
                    num_simulations=mc_cfg["num_simulations"],
                    window=window,
                    distribution=distribution,
                    degrees_of_freedom=mc_cfg["degrees_of_freedom"],
                    random_seed=mc_cfg["random_seed"],
                )
                print(f"    Confidence {cl:.0%}: VaR = {var:.4%}   ES = {es:.4%}")
        print()