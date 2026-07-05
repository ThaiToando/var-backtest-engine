# Interview Prep Notes — VaR Backtest Engine

Running log of design decisions, tradeoffs, and the reasoning behind them.
Read this before an interview — every entry is something you should be able to explain out loud.

## Project setup
- **venv + pip over poetry/conda**: simplest, most transparent dependency management for a
  project-sized (not library-sized) codebase. Matches what most quant/risk teams use for
  research/backtest code. Maps directly into Docker later (`pip install -r requirements.txt`).
- **requirements.txt pinned via `pip freeze`**: guarantees reproducibility — anyone (including
  Docker) installs the exact versions I developed against.

## Repo hygiene
- **outputs/ gitignored except curated final charts**: avoids repo bloat/diff noise from every
  experimental rerun, while still letting a recruiter see final results directly on GitHub
  without cloning/running anything. Compromise between "ignore everything" (clean but invisible)
  and "commit everything" (visible but noisy).
- **Raw price data (data/*.csv): [decision pending — revisit at data_loader step]**

## Portfolio construction
- **9 assets: 6 equities (1 per major sector) + 3 commodity ETFs** (AAPL, JPM, XOM, JNJ, PG, CAT,
  GLD, USO, UNG). Chosen deliberately, not arbitrarily:
  1. Sector spread means the covariance matrix is actually meaningful — an all-tech portfolio
     would make parametric VaR's correlation assumptions uninteresting to discuss.
  2. Mixing equities + commodities creates a natural fat-tails story: Parametric (Normal) VaR
     assumes returns are normally distributed. Commodities (esp. UNG, USO) have fatter tails/higher
     kurtosis than staples like JNJ/PG, so Parametric VaR should systematically underestimate risk
     there relative to Historical/Monte Carlo VaR — a real, demonstrable model weakness, not just
     an assertion.
  3. Gold (GLD) is a classic low/negative-correlation diversifier vs equities, especially in
     selloffs — good talking point on correlation breakdown during crises (correlations trend
     toward 1 in a crash — a known real-world risk-management failure mode).
  4. Backtest exceptions (VaR breaches) likely cluster more in commodities during stress windows
     (COVID crash, 2022 energy shock) than in defensive equities — this is the actual point of
     backtesting: showing where and why a risk model fails, not just that it "works on average."

## Config design
- **Single `config.yaml` as source of truth** (tickers, weights, confidence levels, lookback
  windows, horizon, stress-test date ranges) instead of hardcoding values in each script.
  Rationale: with 9 assets × 3 VaR methods × 2 confidence levels × 2 lookback windows, needing to
  rerun with different parameters (including live, if asked "what if you used 99% instead of
  95%?") should mean changing one YAML value, not hunting through code. Mirrors how production
  risk systems are typically structured.
- **`lookback_windows: [250, 500]`**: 250 trading days ≈ 1yr, 500 ≈ 2yr. Tradeoff: shorter windows
  react faster to new volatility regimes but are noisier; longer windows are more stable but slow
  to "learn" that a regime change happened.
- **`estimation_window` (backtest) kept separate from VaR `lookback_windows`**: backtesting rolls
  forward day-by-day — use prior N days to estimate today's VaR, compare to the actual realized
  loss, slide the window forward one day, repeat. This rolling mechanism is what makes it a
  genuine backtest rather than one static point-in-time estimate.
- **Fixed Monte Carlo random seed (42)**: reproducibility — results should be identical run to
  run for anyone reviewing the project.
- **`price_field` (Close vs Adj Close) — to finalize at data_loader step.**