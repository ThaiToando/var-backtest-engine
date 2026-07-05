"""
Single entry point that reproduces the full core VaR analysis end to end:
  1. Fetch/load price data
  2. Compute returns
  3. Compute VaR/ES via all three methods (Historical, Parametric, Monte Carlo)
  4. Generate charts and the interactive simulation into outputs/

Deliberately does NOT run the full rolling backtest (src/backtest.py) --
with 100,000 Monte Carlo simulations across ~1,900 rolling days, that takes
several minutes and is meant to be run once, intentionally, to validate the
models statistically. This script is the fast "does everything work"
smoke test; run `python -m src.backtest` separately for the full validation.
"""
import subprocess
import sys

STEPS = [
    ("Loading and caching price data", ["src.data_loader"]),
    ("Computing returns and portfolio weights", ["src.returns"]),
    ("Historical Simulation VaR/ES", ["src.var_historical"]),
    ("Parametric (Variance-Covariance) VaR/ES", ["src.var_parametric"]),
    ("Monte Carlo VaR/ES (normal and Student-t)", ["src.var_montecarlo"]),
    ("Generating distribution and comparison charts", ["src.report"]),
    ("Generating interactive Monte Carlo path simulation", ["src.interactive_report"]),
]

if __name__ == "__main__":
    for description, module_args in STEPS:
        print(f"\n{'=' * 60}\n{description}\n{'=' * 60}")
        result = subprocess.run([sys.executable, "-m", *module_args])
        if result.returncode != 0:
            print(f"\nPipeline stopped: '{module_args[0]}' failed.")
            sys.exit(1)

    print(f"\n{'=' * 60}\nPipeline complete. Charts and simulation saved to outputs/.")
    print("For the full statistical backtest (several minutes), run:")
    print("  python -m src.backtest\n")