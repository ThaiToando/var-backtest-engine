"""
Rolling backtest engine: walks a fixed-size estimation window forward through
history day by day, re-estimating VaR at each point using only PRIOR data,
then checking whether the actual next-day loss exceeded it. This is what
turns a VaR model into something empirically validated rather than asserted.

Also implements:
- Kupiec (unconditional coverage) test: is the observed exception rate
  statistically consistent with the rate the model claims?
- Christoffersen (independence) test: do exceptions cluster together in
  time, rather than being spread out? Kupiec alone can't see this.
- Combined conditional coverage test: both checks at once.
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
        "reject_model": reject,
    }


def christoffersen_independence_test(exceptions: np.ndarray, significance: float = 0.05) -> dict:
    exc = exceptions.astype(int)
    n00 = n01 = n10 = n11 = 0
    for prev, curr in zip(exc[:-1], exc[1:]):
        if prev == 0 and curr == 0: n00 += 1
        elif prev == 0 and curr == 1: n01 += 1
        elif prev == 1 and curr == 0: n10 += 1
        elif prev == 1 and curr == 1: n11 += 1

    n0, n1 = n00 + n01, n10 + n11
    pi01 = n01 / n0 if n0 > 0 else 0.0
    pi11 = n11 / n1 if n1 > 0 else 0.0
    pi = (n01 + n11) / (n0 + n1) if (n0 + n1) > 0 else 0.0

    def term(count, prob):
        return count * np.log(prob) if count > 0 and prob > 0 else 0.0

    log_l_null = term(n00 + n10, 1 - pi) + term(n01 + n11, pi)
    log_l_alt = term(n00, 1 - pi01) + term(n01, pi01) + term(n10, 1 - pi11) + term(n11, pi11)

    lr_ind = -2 * (log_l_null - log_l_alt)
    p_value = 1 - chi2.cdf(lr_ind, df=1)

    return {
        "n01": n01, "n11": n11,
        "pi01": pi01, "pi11": pi11,
        "lr_statistic": lr_ind,
        "p_value": p_value,
        "reject_independence": p_value < significance,
    }


def conditional_coverage_test(kupiec_result: dict, christoffersen_result: dict, significance: float = 0.05) -> dict:
    lr_cc = kupiec_result["lr_statistic"] + christoffersen_result["lr_statistic"]
    p_value = 1 - chi2.cdf(lr_cc, df=2)
    return {"lr_statistic": lr_cc, "p_value": p_value, "reject_model": p_value < significance}


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
    port_rets = portfolio_returns(rets, weights)
    dates, var_estimates, actual_losses = [], [], []

    for t in range(estimation_window, len(rets)):
        window_rets = rets.iloc[t - estimation_window:t]
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
                random_seed=base_seed + t,
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
            kupiec = kupiec_test(result["exception"].values, cl)
            christoffersen = christoffersen_independence_test(result["exception"].values)
            cc = conditional_coverage_test(kupiec, christoffersen)

            kupiec_verdict = "REJECTED" if kupiec["reject_model"] else "pass"
            indep_verdict = "CLUSTERED" if christoffersen["reject_independence"] else "independent"
            cc_verdict = "REJECTED" if cc["reject_model"] else "pass"

            print(
                f"  Confidence {cl:.0%}: {kupiec['n_exceptions']}/{kupiec['n_obs']} exceptions "
                f"({kupiec['observed_rate']:.2%} vs expected {kupiec['expected_rate']:.2%})"
            )
            print(f"    Kupiec (coverage):         p={kupiec['p_value']:.3f} -> {kupiec_verdict}")
            print(f"    Christoffersen (timing):   p={christoffersen['p_value']:.3f} -> {indep_verdict}")
            print(f"    Combined (both):           p={cc['p_value']:.3f} -> {cc_verdict}")
        print()