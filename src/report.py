"""
Generates the two flagship charts for this project:
1. return_distribution.png -- empirical portfolio return distribution with
   each VaR method's 99% threshold overlaid, visualizing the fat-tails story.
2. var_es_comparison.png -- grouped bar comparison of VaR/ES across all
   three methods and both confidence levels.

"""
import os
import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import load_config, load_prices
from src.returns import simple_returns, compute_weights, portfolio_returns
from src.var_historical import historical_var, historical_es
from src.var_parametric import parametric_var, parametric_es
from src.var_montecarlo import monte_carlo_var_es

OUTPUT_DIR = "outputs"

COLOR_ACTUAL = "#2C3E50"      # dark slate -- the real data
COLOR_HIST = "#2C3E50"        # matches actual (historical IS the empirical data)
COLOR_PARAM = "#E67E22"       # orange -- parametric/normal
COLOR_MC = "#27AE60"          # green -- monte carlo (t)
BACKGROUND = "#FAFAFA"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "text.color": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def _clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)


def plot_return_distribution(port_rets, window, confidence, hist_var, param_var, mc_var, save_path):
    sample = port_rets.iloc[-window:]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    ax.hist(
        sample, bins=60, density=True, color=COLOR_ACTUAL, alpha=0.35,
        label=f"Actual portfolio returns ({window}-day window)", edgecolor="white", linewidth=0.3,
    )

    for var_value, color, label in [
        (hist_var, COLOR_HIST, f"Historical VaR {confidence:.0%}"),
        (param_var, COLOR_PARAM, f"Parametric VaR {confidence:.0%}"),
        (mc_var, COLOR_MC, f"Monte Carlo (t) VaR {confidence:.0%}"),
    ]:
        ax.axvline(-var_value, color=color, linestyle="--", linewidth=2, label=f"{label}: {var_value:.2%} loss")

    ax.set_title(
        f"Portfolio Daily Return Distribution vs. VaR Estimates ({confidence:.0%} confidence)",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_xlabel("Daily portfolio return")
    ax.set_ylabel("Density")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    _clean_axes(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_var_es_comparison(results: dict, confidence_levels, save_path):
    """
    results: {method_name: {confidence: {"var": x, "es": y}}}
    """
    methods = list(results.keys())
    colors = [COLOR_HIST, COLOR_PARAM, COLOR_MC]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=150, sharey=True)

    for ax, metric, title in zip(axes, ["var", "es"], ["Value at Risk (VaR)", "Expected Shortfall (ES)"]):
        x = np.arange(len(confidence_levels))
        width = 0.25

        for i, (method, color) in enumerate(zip(methods, colors)):
            values = [results[method][cl][metric] for cl in confidence_levels]
            bars = ax.bar(x + i * width, values, width, label=method, color=color, alpha=0.85)
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                        f"{v:.2%}", ha="center", va="bottom", fontsize=8)

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"{cl:.0%}" for cl in confidence_levels])
        ax.set_xlabel("Confidence level")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
        _clean_axes(ax)

    axes[0].set_ylabel("Loss magnitude")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("VaR and Expected Shortfall Across Methods (500-day window)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config = load_config()
    prices = load_prices()
    rets = simple_returns(prices)
    weights = compute_weights(rets, scheme=config["portfolio"]["weighting_scheme"])
    port_rets = portfolio_returns(rets, weights)

    window = 500
    confidence_levels = config["var"]["confidence_levels"]
    mc_cfg = config["var"]["monte_carlo"]

    results = {"Historical": {}, "Parametric": {}, "Monte Carlo (t)": {}}
    for cl in confidence_levels:
        h_var = historical_var(port_rets, cl, window=window)
        h_es = historical_es(port_rets, cl, window=window)
        p_var = parametric_var(rets, weights, cl, window=window)
        p_es = parametric_es(rets, weights, cl, window=window)
        mc_var, mc_es = monte_carlo_var_es(
            rets, weights, cl, num_simulations=mc_cfg["num_simulations"], window=window,
            distribution="t", degrees_of_freedom=mc_cfg["degrees_of_freedom"], random_seed=mc_cfg["random_seed"],
        )
        results["Historical"][cl] = {"var": h_var, "es": h_es}
        results["Parametric"][cl] = {"var": p_var, "es": p_es}
        results["Monte Carlo (t)"][cl] = {"var": mc_var, "es": mc_es}

    plot_return_distribution(
        port_rets, window, confidence=0.99,
        hist_var=results["Historical"][0.99]["var"],
        param_var=results["Parametric"][0.99]["var"],
        mc_var=results["Monte Carlo (t)"][0.99]["var"],
        save_path=os.path.join(OUTPUT_DIR, "return_distribution.png"),
    )

    plot_var_es_comparison(
        results, confidence_levels,
        save_path=os.path.join(OUTPUT_DIR, "var_es_comparison.png"),
    )

    print("Saved charts to outputs/return_distribution.png and outputs/var_es_comparison.png")