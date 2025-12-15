from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd

@dataclass
class BookSnapshot:
    ts_us: int = 0
    best_bid: float = np.nan
    best_ask: float = np.nan
    bid_qty: float = np.nan
    ask_qty: float = np.nan
    mid: float = np.nan
    spread: float = np.nan
    spread_bps: float = np.nan

    # Optional: depth ladders for queue modelling at top N levels
    bid_px: Optional[np.ndarray] = None
    bid_qtys: Optional[np.ndarray] = None
    ask_px: Optional[np.ndarray] = None
    ask_qtys: Optional[np.ndarray] = None


class Book:
    def __init__(self, tick_size: float, depth_levels: int = 5):
        self.tick_size = float(tick_size)
        self.depth_levels = int(depth_levels)
        self.snap = BookSnapshot()

    def update_from_quote_row(self, row: pd.Series, use_depth: bool = True):
        self.snap.ts_us = int(row["ts_us"])
        self.snap.best_bid = float(row["best_bid"])
        self.snap.best_ask = float(row["best_ask"])
        self.snap.mid = float(row["mid"])
        self.snap.spread = float(row["spread"])
        self.snap.spread_bps = float(row["spread_bps"])

        # best qtys (you said these columns exist)
        self.snap.bid_qty = float(row["bids[0].amount"])
        self.snap.ask_qty = float(row["asks[0].amount"])

        if use_depth:
            L = self.depth_levels
            bid_px, bid_qtys, ask_px, ask_qtys = [], [], [], []
            for i in range(L):
                bid_px.append(float(row[f"bids[{i}].price"]))
                bid_qtys.append(float(row[f"bids[{i}].amount"]))
                ask_px.append(float(row[f"asks[{i}].price"]))
                ask_qtys.append(float(row[f"asks[{i}].amount"]))
            self.snap.bid_px = np.array(bid_px, dtype=np.float64)
            self.snap.bid_qtys = np.array(bid_qtys, dtype=np.float64)
            self.snap.ask_px = np.array(ask_px, dtype=np.float64)
            self.snap.ask_qtys = np.array(ask_qtys, dtype=np.float64)

    def is_valid(self) -> bool:
        return np.isfinite(self.snap.best_bid) and np.isfinite(self.snap.best_ask)

    def price_to_tick(self, px: float) -> int:
        return int(round(px / self.tick_size))

    def tick_to_price(self, t: int) -> float:
        return float(t) * self.tick_size