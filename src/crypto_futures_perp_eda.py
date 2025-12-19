import datetime as dt
import numpy as np
import pandas as pd

def summarize_symbol(quote_df: pd.DataFrame, trade_df: pd.DataFrame) -> pd.Series:
    """Return a Series of summary stats for one symbol."""
    q = quote_df.copy()
    t = trade_df.copy()
    
    # --- timestamps to datetime & sort ---
    q['dt'] = pd.to_datetime(q['timestamp'], unit='us')
    t['dt'] = pd.to_datetime(t['timestamp'], unit='us')
    q = q.sort_values('dt').reset_index(drop=True)
    t = t.sort_values('dt').reset_index(drop=True)
    q['date'] = q['dt'].dt.date
    t['date'] = t['dt'].dt.date

    # --- 0. basic data quality checks ---

    # monotonic time
    quote_time_mono = (q['dt'].diff().dropna() >= pd.Timedelta(0)).all()
    trade_time_mono = (t['dt'].diff().dropna() >= pd.Timedelta(0)).all()

    # invalid prices / amounts
    invalid_quote_prices  = q[(q.filter(like='price') <= 0).any(axis=1)]
    invalid_quote_amounts = q[(q.filter(like='amount') < 0).any(axis=1)]
    invalid_trade         = t[(t['price'] <= 0) | (t['amount'] <= 0)]

    # crossed markets
    invalid_cross = q[q['bids[0].price'] >= q['asks[0].price']]

    # depth ladder checks (fast, vectorised)
    ask_price_cols = [c for c in q.columns if 'asks[' in c and '.price' in c]
    bid_price_cols = [c for c in q.columns if 'bids[' in c and '.price' in c]

    ask_errors = 0
    bid_errors = 0
    if ask_price_cols:
        asks_arr = q[ask_price_cols].to_numpy()
        # asks must be non-decreasing: diff < 0 is bad
        ask_errors = (np.diff(asks_arr, axis=1) < 0).any(axis=1).sum()
    if bid_price_cols:
        bids_arr = q[bid_price_cols].to_numpy()
        # bids must be non-increasing: diff > 0 is bad
        bid_errors = (np.diff(bids_arr, axis=1) > 0).any(axis=1).sum()

    # --- 1. date coverage ---
    quote_dates = q['date'].unique()
    trade_dates = t['date'].unique()

    # take first (there should be only 1)
    quote_date = quote_dates[0] if len(quote_dates) > 0 else None
    trade_date = trade_dates[0] if len(trade_dates) > 0 else None

    # --- 2. daily volume / counts ---
    daily_trade = (
        t.groupby('date')
         .agg(total_volume=('amount', 'sum'),
              num_trades   =('amount', 'count'))
    )
    total_volume = float(daily_trade['total_volume'].iloc[0])
    num_trades   = int(daily_trade['num_trades'].iloc[0])

    daily_quotes = (
        q.groupby('date')
         .agg(num_quotes=('exchange', 'count'))
    )
    num_quotes = int(daily_quotes['num_quotes'].iloc[0])

    # --- 3. spread & spread bps ---
    best_ask = q['asks[0].price']
    best_bid = q['bids[0].price']
    q['mid'] = (best_ask + best_bid) / 2
    q['spread'] = best_ask - best_bid
    q['spread_bps'] = q['spread'] / q['mid'] * 1e4

    daily_spread = (
        q.groupby('date')
         .agg(spread_median     =('spread', 'median'),
              spread_mean       =('spread', 'mean'),
              spread_bps_median =('spread_bps', 'median'),
              spread_bps_mean   =('spread_bps', 'mean'))
    )
    spread_median     = float(daily_spread['spread_median'].iloc[0])
    spread_mean       = float(daily_spread['spread_mean'].iloc[0])
    spread_bps_median = float(daily_spread['spread_bps_median'].iloc[0])
    spread_bps_mean   = float(daily_spread['spread_bps_mean'].iloc[0])

    # --- 4. depth & imbalance ---
    q_bid = q['bids[0].amount']
    q_ask = q['asks[0].amount']
    q['imbalance'] = (q_bid - q_ask) / (q_bid + q_ask)

    # depth_stats = (
    #     q.groupby('date')
    #      .agg(bid_depth_mean =('bids[0].amount', 'mean'),
    #           ask_depth_mean =('asks[0].amount', 'mean'),
    #           imbalance_mean =('imbalance', 'mean'))
    # )
    # bid_depth_mean = float(depth_stats['bid_depth_mean'].iloc[0])
    # ask_depth_mean = float(depth_stats['ask_depth_mean'].iloc[0])
    # Compute notional depth first
    q['bid_depth_notional'] = q['bids[0].amount'] * q['bids[0].price']
    q['ask_depth_notional'] = q['asks[0].amount'] * q['asks[0].price']

    # Now compute daily averages
    depth_stats = (
        q.groupby('date')
        .agg(
            bid_depth_mean_contracts = ('bids[0].amount', 'mean'),
            ask_depth_mean_contracts = ('asks[0].amount', 'mean'),
            bid_depth_mean_notional  = ('bid_depth_notional', 'mean'),
            ask_depth_mean_notional  = ('ask_depth_notional', 'mean'),
            imbalance_mean           = ('imbalance', 'mean')
        )
    )
    bid_depth_mean_contracts = float(depth_stats['bid_depth_mean_contracts'].iloc[0])
    ask_depth_mean_contracts = float(depth_stats['ask_depth_mean_contracts'].iloc[0])
    bid_depth_mean_notional = float(depth_stats['bid_depth_mean_notional'].iloc[0])
    ask_depth_mean_notional = float(depth_stats['ask_depth_mean_notional'].iloc[0])

    imbalance_mean = float(depth_stats['imbalance_mean'].iloc[0])
    print(imbalance_mean)

    # --- 5. trade-flow stats ---
    t['signed_volume'] = np.where(t['side'] == 'buy', t['amount'], -t['amount'])

    flow_stats = (
        t.groupby('date')
         .agg(net_signed_volume =('signed_volume', 'sum'),
              abs_signed_volume =('signed_volume', lambda x: x.abs().sum()))
    )
    net_signed_volume = float(flow_stats['net_signed_volume'].iloc[0])
    abs_signed_volume = float(flow_stats['abs_signed_volume'].iloc[0])

    # --- 6. Distribution of time difference ---
    #q['diff_ms'] = q['dt'].diff() / 1000
    q['diff_ms'] = q['dt'].diff().dt.total_seconds() * 1000
    diff_data = q['diff_ms'].dropna()
    if not diff_data.empty:
        diff_mean = diff_data.mean()
        diff_median = diff_data.median()
        diff_max = diff_data.max()
        diff_min = diff_data.min()
    else:
        diff_mean = np.nan
        diff_median = np.nan
        diff_max = np.nan
        diff_min = np.nan  

    # --- assemble results into a Series (rows of your Excel) ---
    s = pd.Series({
        'Is quote time monotonic': quote_time_mono,
        'Is trade time monotonic': trade_time_mono,
        'Invalid quote price rows': int(len(invalid_quote_prices)),
        'Invalid quote amount rows': int(len(invalid_quote_amounts)),
        'Invalid trade rows': int(len(invalid_trade)),
        'bid[0] >= ask[0] rows': int(len(invalid_cross)),
        'Rows invalid ASK price layer': int(ask_errors),
        'Rows invalid BID price layer': int(bid_errors),
        'How many quote dates in data': int(len(quote_dates)),
        'How many trade dates in data': int(len(trade_dates)),
        'Max time diff (ms) between quote': diff_max,
        'Mean time diff (ms) between quote': diff_mean,
        'Median time diff (ms) between quote': diff_median,
        'Min time diff (ms) between quote': diff_min,
        'How many trade dates in data': int(len(trade_dates)),
        'How many trade dates in data': int(len(trade_dates)),
        'Quote date': quote_date,
        'Trade date': trade_date,
        'Total_volume': total_volume,
        'Num_trades': num_trades,
        'Num_quotes': num_quotes,
        'Spread_median': spread_median,
        'Spread_mean': spread_mean,
        'Spread_bps_median': spread_bps_median,
        'Spread_bps_mean': spread_bps_mean,
        'Bid_depth_mean_contracts': bid_depth_mean_contracts,
        'Ask_depth_mean_contracts': ask_depth_mean_contracts,
        'Bid_depth_mean_notional': bid_depth_mean_notional,
        'Ask_depth_mean_notional': ask_depth_mean_notional,
        'Imbalance_mean': imbalance_mean,
        'Net_signed_volume': net_signed_volume,
        'Abs_signed_volume': abs_signed_volume,
    })

    return s