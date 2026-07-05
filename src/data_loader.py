"""
Loads price data for the configured portfolio, computes daily returns,
and caches results to data/prices.csv so we don't hit the API every run.
"""
import os
import yaml
import pandas as pd
import yfinance as yf

CONFIG_PATH = "config.yaml"
DATA_DIR = "data"
PRICES_CACHE = os.path.join(DATA_DIR, "prices.csv")


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_tickers(config: dict) -> list[str]:
    equities = [item["ticker"] for item in config["portfolio"]["equities"]]
    commodities = [item["ticker"] for item in config["portfolio"]["commodities"]]
    return equities + commodities


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Pull adjusted close prices for all tickers in one batched call.
    Using 'Adj Close' (not raw 'Close') because it accounts for dividends
    and stock splits — without this, a split would show up as a fake
    ~50% overnight "loss" in the return series, corrupting the VaR model.
    """
    raw = pd.DataFrame(yf.download(tickers, start=start, end=end, auto_adjust=False)["Adj Close"])
    raw = raw.dropna(how="all")  # drop rows where every ticker is NaN (market holidays etc.)
    return raw


def load_prices(force_refresh: bool = False) -> pd.DataFrame:
    """
    Returns a DataFrame of adjusted close prices, indexed by date, one column per ticker.
    Uses the on-disk cache unless force_refresh=True or the cache doesn't exist yet.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if not force_refresh and os.path.exists(PRICES_CACHE):
        prices = pd.read_csv(PRICES_CACHE, index_col=0, parse_dates=True)
        return prices

    config = load_config()
    tickers = get_tickers(config)
    prices = fetch_prices(
        tickers,
        start=config["data"]["start_date"],
        end=config["data"]["end_date"],
    )
    prices.to_csv(PRICES_CACHE)
    return prices


if __name__ == "__main__":
    prices = load_prices(force_refresh=True)
    print(prices.shape)
    print(prices.head())
    print(prices.tail())