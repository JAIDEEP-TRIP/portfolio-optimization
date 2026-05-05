import time
import warnings
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

START_TIME = time.time()


# Configuration


TICKERS = {
    "Tech":        ["AAPL", "MSFT", "GOOGL", "NVDA"],
    "Healthcare":  ["JNJ",  "PFE",  "UNH",   "ABBV"],
    "Utilities":   ["NEE",  "DUK",  "SO",    "AEP"],
    "Commodities": ["XOM",  "CVX",  "COP",   "SLB"],
    "Finance":     ["JPM",  "BAC",  "GS",    "WFC"]
}

ALL_TICKERS     = [t for s in TICKERS.values() for t in s]
SECTOR_MAP      = {t: s for s, ticks in TICKERS.items() for t in ticks}

START_DATE      = "2023-01-01"
END_DATE        = "2024-01-01"
NUM_STOCKS      = 10
MIN_WEIGHT      = 0.05
MAX_WEIGHT      = 0.30
RISK_FREE_RATE  = 0.05 / 252
WINDOW          = 5
INITIAL_CAPITAL = 10_000.0
MA_SHORT        = 5
MA_LONG         = 20
OUTPUT_DIR      = "outputs/"



# Data Download


def download_prices():
    import time as t
    print("Downloading stock data...")
    all_data = []
    for ticker in ALL_TICKERS:
        for attempt in range(5):
            try:
                df = yf.download(ticker, start=START_DATE, end=END_DATE,
                                 progress=False, auto_adjust=True, threads=False)
                if df is not None and not df.empty:
                    close = df["Close"]
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                    close = close.dropna()
                    close.index = pd.to_datetime(close.index).normalize()
                    close.name = ticker
                    all_data.append(close)
                    print(f"  Got {ticker}")
                    break
            except Exception:
                pass
            t.sleep(3)
        else:
            print(f"  FAILED {ticker}")

    if not all_data:
        raise ValueError("No data downloaded.")

    prices = pd.concat(all_data, axis=1).dropna()
    print(f"Price matrix shape: {prices.shape}")
    return prices


def download_spy():
    df = yf.download("SPY", start=START_DATE, end=END_DATE,
                     auto_adjust=True, progress=False, threads=False)
    if df is None or df.empty:
        raise ValueError("SPY download failed.")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    close.index = pd.to_datetime(close.index).normalize()
    return close


def compute_returns(prices):
    return prices.pct_change().dropna()



# Portfolio Metrics


def portfolio_performance(weights, mean_returns, cov_matrix):
    ret = np.dot(weights, mean_returns) * 252
    vol = np.sqrt(np.dot(weights, np.dot(cov_matrix * 252, weights)))
    sharpe = (ret - RISK_FREE_RATE * 252) / vol if vol > 0 else 0
    return ret, vol, sharpe



# Stock Selection — Top 2 Sharpe per Sector


def select_top_stocks(returns):
    selected = []
    for sector, tickers in TICKERS.items():
        available = [t for t in tickers if t in returns.columns]
        sharpes = {}
        for t in available:
            r = returns[t]
            sharpes[t] = (r.mean() - RISK_FREE_RATE) / r.std() if r.std() > 0 else 0
        top2 = sorted(sharpes, key=sharpes.get, reverse=True)[:2]
        selected.extend(top2)
    return selected[:NUM_STOCKS]



# Optimization — Maximum Sharpe Ratio (SLSQP)


def optimize_portfolio(returns_subset):
    n = len(returns_subset.columns)
    if n < 2:
        return None, None, None, None

    mean_ret = returns_subset.mean()
    cov_mat  = returns_subset.cov()

    def neg_sharpe(w):
        r, v, s = portfolio_performance(w, mean_ret, cov_mat)
        return -s

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds      = tuple((MIN_WEIGHT, MAX_WEIGHT) for _ in range(n))
    init_w      = np.full(n, 1.0 / n)

    result = minimize(neg_sharpe, init_w, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"ftol": 1e-9, "maxiter": 500})

    if result.success:
        w = np.clip(result.x, MIN_WEIGHT, MAX_WEIGHT)
        w /= w.sum()
        r, v, s = portfolio_performance(w, mean_ret, cov_mat)
        return w, r, v, s
    return None, None, None, None



# Strategy 1 — MPT Sliding Window Backtest
# Days 1-5 → optimize weights → apply on day 6
# Slides forward one day at a time


def run_mpt_backtest(prices):
    returns   = compute_returns(prices)
    all_dates = returns.index

    portfolio_val     = INITIAL_CAPITAL
    portfolio_values  = []
    rebalance_records = []

    print(f"\nRunning MPT {WINDOW}-day sliding window backtest ({len(all_dates)} trading days)...")

    for i in range(WINDOW, len(all_dates)):
        window_ret = returns.iloc[i - WINDOW: i]
        today_date = all_dates[i]
        today_ret  = returns.iloc[i]

        selected = select_top_stocks(window_ret)
        if len(selected) < 2:
            portfolio_values.append((today_date, portfolio_val))
            continue

        weights, ann_ret, ann_vol, sharpe = optimize_portfolio(window_ret[selected])

        if weights is None:
            weights = np.full(len(selected), 1.0 / len(selected))
            ann_ret = ann_vol = sharpe = None

        daily_return  = np.dot(weights, today_ret[selected].values)
        portfolio_val *= (1 + daily_return)

        portfolio_values.append((today_date, portfolio_val))
        rebalance_records.append({
            "date":       today_date,
            "stocks":     selected,
            "weights":    dict(zip(selected, weights.round(4))),
            "ann_return": round(ann_ret, 4) if ann_ret else None,
            "ann_vol":    round(ann_vol, 4) if ann_vol else None,
            "sharpe":     round(sharpe, 4)  if sharpe  else None
        })

    port_df = pd.DataFrame(portfolio_values, columns=["Date", "Value"]).set_index("Date")
    return port_df, rebalance_records



# Strategy 2 — Moving Average Crossover (Golden Cross)
# 5-day MA > 20-day MA  → invested in SPY
# 5-day MA < 20-day MA  → hold cash (no return)


def run_ma_strategy(spy_series):
    print("Running Moving Average Crossover strategy...")

    df = pd.DataFrame({"price": spy_series})
    df["ma_short"] = df["price"].rolling(MA_SHORT).mean()
    df["ma_long"]  = df["price"].rolling(MA_LONG).mean()
    df = df.dropna()

    val    = INITIAL_CAPITAL
    values = []

    for i in range(1, len(df)):
        today     = df.index[i]
        ret       = df["price"].iloc[i] / df["price"].iloc[i - 1] - 1
        in_market = df["ma_short"].iloc[i - 1] > df["ma_long"].iloc[i - 1]

        if in_market:
            val *= (1 + ret)

        values.append((today, val))

    ma_df = pd.DataFrame(values, columns=["Date", "Value"]).set_index("Date")
    return ma_df



# S&P 500 Buy-and-Hold Baseline


def build_spy_baseline(spy_series, port_index):
    spy_ret = spy_series.pct_change().dropna()
    val     = INITIAL_CAPITAL
    values  = []
    for date, r in spy_ret.items():
        val *= (1 + r)
        values.append((date, val))

    spy_df = pd.DataFrame(values, columns=["Date", "Value"]).set_index("Date")
    port_index_norm = pd.to_datetime(port_index).normalize()
    common = port_index_norm.intersection(spy_df.index)

    if common.empty:
        raise ValueError(
            f"No date overlap between SPY and portfolio.\n"
            f"SPY:  {spy_df.index[0]} → {spy_df.index[-1]}\n"
            f"Port: {port_index[0]} → {port_index[-1]}"
        )

    spy_df = spy_df.loc[common]
    spy_df["Value"] = spy_df["Value"] / spy_df["Value"].iloc[0] * INITIAL_CAPITAL
    return spy_df



# Align all three strategies to common dates


def align_strategies(port_df, spy_df, ma_df):
    common = port_df.index.intersection(spy_df.index).intersection(ma_df.index)
    port_df = port_df.loc[common].copy()
    spy_df  = spy_df.loc[common].copy()
    ma_df   = ma_df.loc[common].copy()

    for df in [spy_df, ma_df]:
        df["Value"] = df["Value"] / df["Value"].iloc[0] * INITIAL_CAPITAL

    return port_df, spy_df, ma_df



# Compute Summary Metrics


def compute_metrics(series, label):
    daily        = series.pct_change().dropna()
    total_return = (series.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    sharpe       = (daily.mean() - RISK_FREE_RATE) / daily.std() * np.sqrt(252)
    max_drawdown = ((series / series.cummax()) - 1).min() * 100
    return {
        "Strategy":            label,
        "Final Value ($)":     round(series.iloc[-1], 2),
        "Total Return (%)":    round(total_return, 2),
        "Annualized Sharpe":   round(sharpe, 4),
        "Max Drawdown (%)":    round(max_drawdown, 2)
    }



# Plot 1 — Efficient Frontier


def plot_efficient_frontier(returns, selected_stocks, optimal_weights):
    print("Plotting efficient frontier...")
    mean_ret = returns[selected_stocks].mean()
    cov_mat  = returns[selected_stocks].cov()
    n        = len(selected_stocks)

    sim_r, sim_v, sim_s = [], [], []
    for _ in range(3000):
        w = np.random.dirichlet(np.ones(n))
        r, v, s = portfolio_performance(w, mean_ret, cov_mat)
        sim_r.append(r); sim_v.append(v); sim_s.append(s)

    opt_r, opt_v, opt_s = portfolio_performance(optimal_weights, mean_ret, cov_mat)

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(sim_v, sim_r, c=sim_s, cmap="viridis", alpha=0.4, s=12,
                    label="Random Portfolios")
    ax.scatter(opt_v, opt_r, color="red", s=200, zorder=5, marker="*",
               label=f"Max Sharpe = {opt_s:.2f}")
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")
    ax.set_xlabel("Annualized Volatility", fontsize=12)
    ax.set_ylabel("Annualized Return",     fontsize=12)
    ax.set_title("Efficient Frontier — Random Portfolios vs Optimal Point", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = OUTPUT_DIR + "efficient_frontier.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")



# Plot 2 — Portfolio Value: All 3 Strategies


def plot_performance(port_df, spy_df, ma_df):
    print("Plotting performance comparison...")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(port_df.index, port_df["Value"], label="MPT Portfolio",
            color="steelblue",  linewidth=2)
    ax.plot(spy_df.index,  spy_df["Value"],  label="S&P 500 (Buy & Hold)",
            color="darkorange", linewidth=2, linestyle="--")
    ax.plot(ma_df.index,   ma_df["Value"],   label="MA Crossover (5/20)",
            color="green",      linewidth=2, linestyle="-.")
    ax.set_xlabel("Date",                fontsize=11)
    ax.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax.set_title("Strategy Comparison — $10,000 Starting Capital (2023)", fontsize=13)
    ax.legend(fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    plt.tight_layout()
    path = OUTPUT_DIR + "performance_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")



# Plot 3 — Daily Allocation Shift


def plot_allocation_shift(rebalance_records):
    print("Plotting allocation shift...")
    dates         = [r["date"] for r in rebalance_records]
    weight_matrix = [{t: r["weights"].get(t, 0.0) for t in ALL_TICKERS}
                     for r in rebalance_records]
    df_w   = pd.DataFrame(weight_matrix, index=dates)
    colors = plt.cm.tab20.colors

    fig, ax = plt.subplots(figsize=(13, 5))
    bottom  = np.zeros(len(df_w))
    for idx, col in enumerate(df_w.columns):
        vals = df_w[col].values
        ax.fill_between(df_w.index, bottom, bottom + vals,
                        label=col, alpha=0.8, color=colors[idx % len(colors)])
        bottom += vals

    ax.set_xlabel("Date",   fontsize=11)
    ax.set_ylabel("Weight", fontsize=11)
    ax.set_title("Daily Allocation Shift Across Stocks (MPT)", fontsize=13)
    ax.legend(loc="upper right", fontsize=7, ncol=4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    plt.tight_layout()
    path = OUTPUT_DIR + "allocation_shift.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")



# Plot 4 — Rolling 21-Day Sharpe Ratio


def plot_rolling_sharpe(port_df, spy_df, ma_df):
    print("Plotting rolling Sharpe...")
    roll = 21

    def rolling_sharpe(series):
        daily = series.pct_change().dropna()
        return (daily.rolling(roll).mean() - RISK_FREE_RATE) / \
                daily.rolling(roll).std() * np.sqrt(252)

    rs_mpt = rolling_sharpe(port_df["Value"])
    rs_spy = rolling_sharpe(spy_df["Value"])
    rs_ma  = rolling_sharpe(ma_df["Value"])

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(rs_mpt.index, rs_mpt, label="MPT Portfolio",
            color="steelblue",  linewidth=1.5)
    ax.plot(rs_spy.index, rs_spy, label="S&P 500 (Buy & Hold)",
            color="darkorange", linewidth=1.5, linestyle="--")
    ax.plot(rs_ma.index,  rs_ma,  label="MA Crossover (5/20)",
            color="green",      linewidth=1.5, linestyle="-.")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Date",                       fontsize=11)
    ax.set_ylabel(f"Rolling {roll}-Day Sharpe", fontsize=11)
    ax.set_title(f"Rolling {roll}-Day Sharpe Ratio — All Strategies", fontsize=13)
    ax.legend(fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    plt.tight_layout()
    path = OUTPUT_DIR + "rolling_sharpe.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")



# Print + Save Summary


def print_and_save_summary(port_df, spy_df, ma_df, rebalance_records, elapsed):
    metrics = [
        compute_metrics(port_df["Value"], "MPT Portfolio"),
        compute_metrics(spy_df["Value"],  "S&P 500 Buy & Hold"),
        compute_metrics(ma_df["Value"],   "MA Crossover (5/20)"),
    ]

    print("\n" + "=" * 66)
    print("PERFORMANCE SUMMARY")
    print("=" * 66)
    print(f"{'Metric':<28} {'MPT':>12} {'S&P 500':>12} {'MA Cross':>10}")
    print("-" * 66)
    for key in ["Final Value ($)", "Total Return (%)", "Annualized Sharpe", "Max Drawdown (%)"]:
        vals = [str(m[key]) for m in metrics]
        print(f"{key:<28} {vals[0]:>12} {vals[1]:>12} {vals[2]:>10}")
    print("=" * 66)
    print(f"\nTotal runtime: {elapsed:.1f} seconds")

    if rebalance_records:
        last = rebalance_records[-1]
        print(f"\nLast Rebalance Date: {last['date'].date()}")
        print(f"{'Ticker':<8} {'Sector':<14} {'Weight':>8}")
        print("-" * 32)
        for ticker, w in sorted(last["weights"].items(), key=lambda x: -x[1]):
            print(f"{ticker:<8} {SECTOR_MAP.get(ticker, 'N/A'):<14} {w*100:>7.1f}%")

    summary_df = pd.DataFrame(metrics)
    path = OUTPUT_DIR + "summary.csv"
    summary_df.to_csv(path, index=False)
    print(f"\nSaved: {path}")



# Main

def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Data 
    prices = download_prices()
    spy    = download_spy()

    # Initial selection + frontier 
    returns          = compute_returns(prices)
    selected_initial = select_top_stocks(returns)
    print(f"\nInitial stock selection: {selected_initial}")

    opt_w, _, _, opt_s = optimize_portfolio(returns[selected_initial])
    if opt_w is None:
        opt_w = np.full(len(selected_initial), 1.0 / len(selected_initial))
    if opt_s:
        print(f"Optimal Sharpe (full period): {opt_s:.4f}")

    plot_efficient_frontier(returns, selected_initial, opt_w)

    # Run all three strategies 
    port_df, rebalance_records = run_mpt_backtest(prices)
    spy_df                     = build_spy_baseline(spy, port_df.index)
    ma_df                      = run_ma_strategy(spy)

    # Align to common dates 
    port_df, spy_df, ma_df = align_strategies(port_df, spy_df, ma_df)

    # Generate all 4 plots 
    plot_performance(port_df, spy_df, ma_df)
    plot_allocation_shift(rebalance_records)
    plot_rolling_sharpe(port_df, spy_df, ma_df)

    # Save portfolio values 
    port_df.to_csv(OUTPUT_DIR + "portfolio_values.csv")
    print(f"Saved: {OUTPUT_DIR}portfolio_values.csv")

    # Final summary 
    elapsed = time.time() - START_TIME
    print_and_save_summary(port_df, spy_df, ma_df, rebalance_records, elapsed)


if __name__ == "__main__":
    main()
