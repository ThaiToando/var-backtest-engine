"""
Rolling backtest engine: walks a fixed-size estimation window forward through
history day by day, re-estimating VaR at each point using only PRIOR data,
then checking whether the actual next-day loss exceeded it. This is what
turns a VaR model into something empirically validated rather than asserted.

Also implements the Kupiec (unconditional coverage) test: a formal
statistical test of whether the observed exception rate is consistent with
the rate the model claims (e.g. a 95% VaR model should be exceeded ~5% of
the time). Answers "is this deviation from 5% just noise, or is the model
actually miscalibrated?" with a p-value instead of eyeballing it.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2

from src.data_loader import load_config, load_prices
from src.returns import simple_returns, compute_weights, portfolio_returns
from src.var_historical import historical_var
from src.var_parametric import parametric_var
from src.var_montecarlo import simulate_portfolio_returns, var_es_from_simulated


def kupiec_test(exceptions: np.ndarray, confidence: float, significance: float = 0.05) -> dict:
    n = len(exceptions)
    x = int(exceptions.sum())
    p = 1 - confidence  # expected exception rate
    observed_rate = x / n

    # Likelihood ratio statistic -- handle x=0 and x=n edge cases separately
    # to avoid log(0).
    if x == 0:
        log_l_null = n * np.log(1 - p)
        log_l_alt = n * np.log(1 - observed_rate) if observed_rate < 1 else 0
    elif x == n:
        log_l_null = n * np.log(p)
        log_l_alt = n * np.log(observed_rate)
    else:
        log_l_null = x * np.log(p) + (n - x) * np.log(1 - p)
        log_l_alt = x * np.log(observed_rate) + (n - x) * np.log(1 - observed_rate)

    lr_stat = -2 * (log_l_null - log_l_alt)
    p_value = 1 - chi2.cdf(lr_stat, df=1)
    reject = p_value < significance

    return {
        "n_obs": n,
        "n_exceptions": x,
        "expected_rate": p,
        "observed_rate": observed_rate,
        "lr_statistic": lr_stat,
        "p_value": p_value,
        "reject_model": reject,  # True = model FAILS the test (miscalibrated)
    }


def rolling_backtest(
    rets: pd.DataFrame,
    weights: pd.Series,
    confidence: float,
    estimation_window: int,
    method: str,
    num_simulations: int = 100_000,
    degrees_of_freedom: int = 5,
    base_seed: int = 42,
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by date with columns: var_estimate,
    actual_loss, exception (bool). One row per day tested (i.e. every day
    after the first estimation_window days).
    """
    port_rets = portfolio_returns(rets, weights)
    dates, var_estimates, actual_losses = [], [], []

    for t in range(estimation_window, len(rets)):
        window_rets = rets.iloc[t - estimation_window:t]          # strictly prior data
        window_port_rets = port_rets.iloc[t - estimation_window:t]

        if method == "historical":
            var = historical_var(window_port_rets, confidence)
        elif method == "parametric":
            var = parametric_var(window_rets, weights, confidence)
        elif method in ("monte_carlo_normal", "monte_carlo_t"):
            distribution = "normal" if method == "monte_carlo_normal" else "t"
            sim = simulate_portfolio_returns(
                window_rets, weights, num_simulations,
                distribution=distribution, degrees_of_freedom=degrees_of_freedom,
                random_seed=base_seed + t,  # vary seed per day so paths aren't identical
            )
            var, _ = var_es_from_simulated(sim, confidence)
        else:
            raise ValueError(f"Unknown method: {method}")

        actual_return = port_rets.iloc[t]
        actual_loss = -actual_return

        dates.append(rets.index[t])
        var_estimates.append(var)
        actual_losses.append(actual_loss)

    df = pd.DataFrame({
        "var_estimate": var_estimates,
        "actual_loss": actual_losses,
    }, index=pd.Index(dates, name="date"))
    df["exception"] = df["actual_loss"] > df["var_estimate"]
    return df


if __name__ == "__main__":
    config = load_config()
    prices = load_prices()
    rets = simple_returns(prices)
    weights = compute_weights(rets, scheme=config["portfolio"]["weighting_scheme"])

    estimation_window = config["backtest"]["estimation_window"]
    mc_cfg = config["var"]["monte_carlo"]
    methods = ["historical", "parametric", "monte_carlo_normal", "monte_carlo_t"]

    print(f"Rolling backtest: {len(rets) - estimation_window} days tested, "
          f"{estimation_window}-day estimation window\n")

    for method in methods:
        print(f"=== Method: {method} ===")
        for cl in config["var"]["confidence_levels"]:
            result = rolling_backtest(
                rets, weights, cl, estimation_window, method,
                num_simulations=mc_cfg["num_simulations"],
                degrees_of_freedom=mc_cfg["degrees_of_freedom"],
                base_seed=mc_cfg["random_seed"],
            )
            test = kupiec_test(result["exception"].values, cl)
            verdict = "REJECTED (miscalibrated)" if test["reject_model"] else "not rejected"
            print(
                f"  Confidence {cl:.0%}: {test['n_exceptions']}/{test['n_obs']} exceptions "
                f"({test['observed_rate']:.2%} vs expected {test['expected_rate']:.2%}) "
                f"-- Kupiec p={test['p_value']:.3f} -> {verdict}"
            )
        print()