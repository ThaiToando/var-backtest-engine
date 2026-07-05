# VaR Backtest Engine

A Value-at-Risk (VaR) model for a 9-asset equity + commodity portfolio, built three different
ways (Historical Simulation, Parametric, Monte Carlo) and then backtested against real market
data to see which one actually holds up.

**[Live interactive simulation ▶](https://thaitoando.github.io/var-backtest-engine/outputs/monte_carlo_paths.html)**
— 300 simulated 60-day portfolio paths, animated, with VaR breaches highlighted live.

## Why

Any risk desk can produce a VaR number. The harder and more important question is whether that
number is actually correct or whether a model that claims "you won't lose more than X% on 95% of
days" holds up when you run it against history. That's what backtesting is for by rolling the model
forward day by day, count how often it was wrong and check statistically whether that error
rate matches what the model promised. This project does that properly with two formal tests
(Kupiec and Christoffersen).

## The portfolio

Equal-weighted, 6 equities across different sectors (AAPL, JPM, XOM, JNJ, PG, CAT) plus 3
commodity ETFs (GLD, USO, UNG), 2018–2026 daily data. Mixing equities and commodities was
deliberate as it gives the correlation structure something real to say (oil and energy stocks
move together, gold doesn't move with much of anything) and creates genuine variation in tail
risk across the portfolio instead of nine stocks that all behave the same way.

## Three models

Historical Simulation uses actual past returns, no distributional assumption but can
  never see a worse day than what's already in its window.
Parametric covariance: a closed-form formula assuming returns are normally
  distributed.
Monte Carlo simulation : 100,000 simulated scenarios, configurable to draw from a normal or
  fat-tailed Student-t distribution.

Expected Shortfall (the average loss once VaR is breached) is computed for all three, since VaR
alone doesn't say how bad things get once you're in the tail as the reason Basel moved to ES as
the regulatory standard after 2008.

## What I'd do next

Add a GARCH(1,1) volatility model and the clustering finding above is a direct pointer toward it.
Calibrate the Student-t degrees of freedom to this portfolio's actual kurtosis instead of using
a fixed default of 5. Try an inverse-volatility weighting scheme as the default instead of equal
weight since it's already implemented but not used in the headline results.