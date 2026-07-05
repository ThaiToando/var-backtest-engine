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
- - **Resolved: using `Adj Close`**, confirmed via data_loader.py — 2134 trading days x 9 tickers,
  2018-01-02 to latest available. Adjusted prices avoid fake "losses" from splits/dividends.
  ## Return statistics findings (src/returns.py output)
- Portfolio (equal-weight) daily return: mean 0.000574, std 0.010666, min -0.0842, max 0.0769
  over 2133 days (2018-2026).
- Excess kurtosis is high portfolio-wide (JPM 13.2, USO 13.8, PG 10.2, JNJ 8.5), NOT concentrated
  in commodities as the naive "commodities have fatter tails" story predicted. UNG (the most
  volatile single asset, std 0.036) actually has the LOWEST kurtosis (3.0).
- Real explanation: excess kurtosis is regime-relative -- it measures tail extremity relative to
  an asset's OWN typical volatility, not raw riskiness. UNG's big moves are "normal" for UNG given
  its already-high baseline vol. JPM/USO are normally calmer, so a shared shock (COVID 2020, and
  USO's negative-oil-price event in April 2020, skew -1.01) looks extreme relative to their usual
  behavior.
- This is why full-sample stats can mislead, and why rolling-window backtesting (not one static
  estimate) is the right approach -- a single 2018-2026 average blends very different volatility
  regimes together.
- ## Historical VaR findings (src/var_historical.py output)
- 250-day window: 95% VaR=0.906%, ES=1.387%; 99% VaR=1.765%, ES=2.078%
- 500-day window: 95% VaR=1.082%, ES=1.772%; 99% VaR=1.925%, ES=3.500%
- Longer window's 99% ES is much worse (3.50% vs 2.08%) -- it captures rougher days the
  shorter window doesn't. Concrete evidence for the responsiveness-vs-stability tradeoff.
- IMPORTANT: neither rolling window (as of dataset end, 2026-06-30) reaches back far enough to
  include the COVID crash (2020) or 2022 energy shock defined in config's stress_periods --
  today's rolling VaR is realistic (a live desk only sees recent history) but means the
  backtest step must explicitly walk the window back through those crisis periods to actually
  test model behavior during stress, not just rely on the current window.
## Parametric VaR findings (src/var_parametric.py output)
- Correlation matrix confirms design hypotheses: XOM-USO = 0.57 (energy equity/commodity link),
  GLD near-zero/negative vs everything (JPM -0.02, JNJ 0.05, CAT 0.08) -- genuine diversifier.
  Unexpected: UNG is nearly uncorrelated with the whole portfolio (0.02-0.13 range) -- driven by
  idiosyncratic supply/weather factors.
- 99% confidence, 500-day window: Historical ES = 3.50% vs Parametric ES = 2.17% -- parametric
  substantially UNDERSTATES tail risk here, consistent with fat tails (high kurtosis) found earlier.
- 95% confidence, 250-day window: Parametric VaR (1.072%) is actually HIGHER than Historical
  (0.906%) -- parametric OVERSTATES risk at this level.
- This crossover (parametric conservative at 95%, permissive at 99%) is the classic signature of
  approximating a leptokurtic (fat-tailed) distribution with a normal one: fat tails concentrate
  more mass in the center and extreme tails, less in the "shoulders," than a normal curve with
  matching std. Strong, specific talking point -- not just "parametric underestimates risk."
## Monte Carlo VaR findings (src/var_montecarlo.py output)
- MC "normal" matches Parametric VaR almost exactly (e.g. 250-day/99%: both 1.5710%) --
  cross-validates both implementations independently.
- 500-day/99%: MC "t" ES = 3.67% vs Parametric ES = 2.17% vs Historical ES = 3.50%.
  Switching Monte Carlo's distribution from normal to Student-t (df=5) recovers almost exactly
  the fat-tail risk Parametric was missing, while still respecting full 9-asset correlation --
  this is the complete, demonstrated answer to "why does distribution choice matter."
- 250-day/99%: MC "t" ES = 3.16%, HIGHER than Historical's 2.08%. Not a contradiction: the
  250-day window (mid-2025 to mid-2026) is a calm period with no major crisis, so Historical
  Simulation -- limited to only the losses actually observed in-sample -- likely UNDERSTATES
  true tail risk here. Monte Carlo with fat tails isn't bounded by the sample's worst day, so it
  correctly allows for worse outcomes even when none happened to occur in that particular window.