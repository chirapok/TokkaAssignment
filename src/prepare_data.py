import numpy as np
import pandas as pd

import datetime as dt

def _ensure_ts_us(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.Series:
    """
    Binance timestamps in your screenshots look like ~1.754e15 => microseconds since epoch.
    Return int64 microseconds.
    """
    ts = df[ts_col].astype("int64")
    # sanity check: if it's milliseconds (1e12) or microseconds (1e15)
    med = int(ts.median())
    if med < 10**13:
        # seconds
        ts_us = ts * 1_000_000
    elif med < 10**15:
        # milliseconds
        ts_us = ts * 1_000
    else:
        # microseconds
        ts_us = ts
    return ts_us.astype("int64")

def _build_event_stream(quotes: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    q_events = pd.DataFrame({
        "ts_us": quotes["ts_us"].to_numpy(),
        "ts": quotes["ts"].to_numpy(),
        "etype": "quote",
        "qidx": np.arange(len(quotes), dtype=np.int32),
        "tidx": np.int32(-1),
    })

    t_events = pd.DataFrame({
        "ts_us": trades["ts_us"].to_numpy(),
        "ts": trades["ts"].to_numpy(),
        "etype": "trade",
        "qidx": np.int32(-1),
        "tidx": np.arange(len(trades), dtype=np.int32),
    })

    events = pd.concat([q_events, t_events], ignore_index=True)

    # tie-breaker priority: trade first, then quote
    prio = {"trade": 0, "quote": 1}
    events["prio"] = events["etype"].map(prio).astype(np.int8)

    events = events.sort_values(["ts_us", "prio"], kind="mergesort").drop(columns=["prio"]).reset_index(drop=True)
    return events

def _read_in_data(asset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read in the quote and trade data for the given asset."""
    quote_df = pd.read_csv(f"../data/{asset_name}_quote.csv")
    trade_df = pd.read_csv(f"../data/{asset_name}_trade.csv")

    quote_df["ts_us"] = _ensure_ts_us(quote_df, "timestamp")
    quote_df['ts'] = pd.to_datetime(quote_df['timestamp'], unit='us')

    trade_df["ts_us"] = _ensure_ts_us(trade_df, "timestamp")
    trade_df['ts'] = pd.to_datetime(trade_df['timestamp'], unit='us')

    # 0.1 Also store the date (no time) for grouping
    quote_df['date'] = quote_df['ts'].dt.date
    trade_df['date'] = trade_df['ts'].dt.date

    trade_df["side"] = trade_df["side"].str.lower().replace({"b": "buy", "s": "sell"})

    # Adjust indices if your best level is asks[0]/bids[0] (it looks like it is)
    BEST_ASK_P = "asks[0].price"
    BEST_BID_P = "bids[0].price"
    # BEST_ASK_Q = "asks[0].amount"
    # BEST_BID_Q = "bids[0].amount"

    quote_df["best_ask"] = quote_df[BEST_ASK_P].astype(float)
    quote_df["best_bid"] = quote_df[BEST_BID_P].astype(float)
    quote_df["mid"] = 0.5 * (quote_df["best_ask"] + quote_df["best_bid"])
    quote_df["spread"] = (quote_df["best_ask"] - quote_df["best_bid"])
    quote_df["spread_bps"] = (quote_df["spread"] / quote_df["mid"]) * 1e4

    return quote_df, trade_df

def load_and_process_data(asset_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """_summary_

    Args:
        asset_name (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    quote_df, trade_df = _read_in_data(asset_name)
    event_df = _build_event_stream(quote_df, trade_df)

    # Precomputing rolling stats
    # quotes_df = quotes_df.sort_values("ts_us").reset_index(drop=True)
    # quotes_df["ts"] = pd.to_datetime(quotes_df["ts_us"], unit="us", utc=True)
    quote_df = quote_df.set_index("ts")

    BASELINE_WINDOW = "200ms"
    quote_df["spread_med_bps"] = quote_df["spread_bps"].rolling(BASELINE_WINDOW).median()

    # restore columns used by engine
    quote_df = quote_df.reset_index()

    return quote_df, trade_df, event_df

# # --- quotes ---
# hype_quote_df = hype_quote_df.copy()
# hype_quote_df["ts_us"] = _ensure_ts_us(hype_quote_df, "timestamp")
# hype_quote_df["ts"] = pd.to_datetime(hype_quote_df["ts_us"], unit="us", utc=True)
# hype_quote_df = hype_quote_df.sort_values("ts_us", kind="mergesort").reset_index(drop=True)

# # --- trades ---
# hype_trade_df = hype_trade_df.copy()
# hype_trade_df["ts_us"] = _ensure_ts_us(hype_trade_df, "timestamp")
# hype_trade_df["ts"] = pd.to_datetime(hype_trade_df["ts_us"], unit="us", utc=True)
# hype_trade_df = hype_trade_df.sort_values("ts_us", kind="mergesort").reset_index(drop=True)

# # Optional: normalize side
# hype_trade_df["side"] = hype_trade_df["side"].str.lower().replace({"b": "buy", "s": "sell"})