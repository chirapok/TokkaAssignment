from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd

# import sys
# from pathlib import Path

from book import Book
from order import Order, Fill

class ExchangeSim:
    def __init__(
        self,
        asset_name: str,
        tick_size: float,
        maker_fee: float = 0.0002,   # 0.02%
        taker_fee: float = 0.0005,   # 0.05% (unused for now; you can add market orders later)
        latency_us: int = 5_000,     # 5 ms default; tune later
        depth_levels: int = 5,
    ):
        if asset_name == 'HYPE':
            self.qty_step = 0.01
        elif asset_name == 'BTC' or asset_name == 'ETH':
            self.qty_step = 0.001
        elif asset_name == 'SOL':
            self.qty_step = 0.01
        elif asset_name == 'XRP':
            self.qty_step = 0.1

        self.book = Book(tick_size=tick_size, depth_levels=depth_levels)
        self.maker_fee = float(maker_fee)
        self.taker_fee = float(taker_fee)
        self.latency_us = int(latency_us)

        self._next_oid = 1
        self.orders: Dict[int, Order] = {}
        self.active_ids: set[int] = set()

        # accounting
        self.pos = 0.0       # + long, - short
        self.cash = 0.0      # cash PnL
        self.fills: List[Fill] = []

    # ---------- helpers ----------
    def _estimate_queue_ahead(self, side: str, price: float) -> float:
        """
        Very first-cut:
        - If order is at best level, use displayed qty at that level (bids[0]/asks[0]).
        - If improving inside spread, assume small queue ahead (near-front).
        """
        s = self.book.snap
        if not self.book.is_valid():
            return 0.0

        if side == "buy":
            if abs(price - s.best_bid) < 1e-12:
                return float(s.bid_qty)
            if price > s.best_bid and price < s.best_ask:
                return 0.0
            return 0.0
        else:  # sell
            if abs(price - s.best_ask) < 1e-12:
                return float(s.ask_qty)
            if price < s.best_ask and price > s.best_bid:
                return 0.0
            return 0.0

    def _activate_orders(self, ts_us: int):
        for oid, o in self.orders.items():
            if (not o.is_done) and (not o.is_active) and ts_us >= o.live_us:
                o.is_active = True
                self.active_ids.add(oid)
    
    def _round_qty(self, q: float) -> float:
        return np.floor(q / self.qty_step) * self.qty_step
    
    # ---------- external API ----------
    def on_quote(self, quote_row: pd.Series):
        self.book.update_from_quote_row(quote_row, use_depth=True)

    def place_limit(self, side: str, price: float, qty: float, ts_us: int) -> int:
        side = side.lower()
        assert side in ("buy", "sell")
        price = float(price)
        qty = float(qty)

        qty = self._round_qty(qty)
        if qty <= 0:
            raise ValueError("Order qty must be positive after rounding.")

        oid = self._next_oid
        self._next_oid += 1

        q_ahead = self._estimate_queue_ahead(side, price)
        o = Order(
            order_id=oid,
            side=side,
            price=price,
            qty=qty,
            ts_place_us=int(ts_us),
            live_us=int(ts_us) + self.latency_us,
            queue_ahead=float(q_ahead),
        )
        self.orders[oid] = o
        return oid

    def cancel(self, order_id: int):
        o = self.orders.get(order_id)
        if o is None or o.is_done:
            return
        o.is_done = True
        o.is_active = False
        self.active_ids.discard(order_id)

    def cancel_all(self):
        for oid in list(self.active_ids):
            self.cancel(oid)

    def on_trade(self, trade_row: pd.Series, ts_us: int):
        """
        Trade processing:
        - Activate any orders whose latency has elapsed.
        - Update queue and fill eligible orders based on trade side.
        """
        self._activate_orders(ts_us)

        side = str(trade_row["side"]).lower()
        trade_px = float(trade_row["price"])
        trade_qty = float(trade_row["amount"])

        # Which resting orders are impacted?
        # If trade side == "sell": market sell consumes bids -> may fill our BUY limits at/better than trade price
        # If trade side == "buy" : market buy consumes asks -> may fill our SELL limits at/better than trade price
        if side == "sell":
            self._apply_trade_to_buys(ts_us, trade_px, trade_qty)
        elif side == "buy":
            self._apply_trade_to_sells(ts_us, trade_px, trade_qty)
        else:
            return

    # ---------- fill logic ----------
    def _apply_trade_to_buys(self, ts_us: int, trade_px: float, trade_qty: float):
        # Eligible buys: active, not done, price >= trade_px (trade prints at/through their level)
        # Conservative: require trade_px <= order.price
        eligible = []
        for oid in self.active_ids:
            o = self.orders[oid]
            if o.is_done or o.side != "buy":
                continue
            if trade_px <= o.price + 1e-12:
                eligible.append(o)

        # Higher price has priority (better bid)
        eligible.sort(key=lambda x: (-x.price, x.ts_place_us))

        remaining = trade_qty
        for o in eligible:
            if remaining <= 0:
                break

            # consume queue ahead first
            if o.queue_ahead > 0:
                dq = min(o.queue_ahead, remaining)
                o.queue_ahead -= dq
                remaining -= dq
                if remaining <= 0:
                    break

            # now fill our order
            fill_qty = min(o.qty_remaining, remaining)
            fill_qty = self._round_qty(fill_qty)

            if fill_qty <= 0:
                return
            
            if fill_qty > 0:
                self._record_fill(ts_us, o, fill_qty, maker=True)
                o.qty_remaining -= fill_qty
                remaining -= fill_qty

                if o.qty_remaining <= 1e-12:
                    o.is_done = True
                    o.is_active = False
                    self.active_ids.discard(o.order_id)

    def _apply_trade_to_sells(self, ts_us: int, trade_px: float, trade_qty: float):
        # Eligible sells: active, not done, price <= trade_px
        eligible = []
        for oid in self.active_ids:
            o = self.orders[oid]
            if o.is_done or o.side != "sell":
                continue
            if trade_px >= o.price - 1e-12:
                eligible.append(o)

        # Lower price has priority (better ask)
        eligible.sort(key=lambda x: (x.price, x.ts_place_us))

        remaining = trade_qty
        for o in eligible:
            if remaining <= 0:
                break

            if o.queue_ahead > 0:
                dq = min(o.queue_ahead, remaining)
                o.queue_ahead -= dq
                remaining -= dq
                if remaining <= 0:
                    break

            fill_qty = min(o.qty_remaining, remaining)
            if fill_qty > 0:
                self._record_fill(ts_us, o, fill_qty, maker=True)
                o.qty_remaining -= fill_qty
                remaining -= fill_qty

                if o.qty_remaining <= 1e-12:
                    o.is_done = True
                    o.is_active = False
                    self.active_ids.discard(o.order_id)

    def _record_fill(self, ts_us: int, o: Order, qty: float, maker: bool):
        px = o.price
        fee_rate = self.maker_fee if maker else self.taker_fee
        notional = px * qty
        fee = notional * fee_rate

        # inventory/cash updates
        if o.side == "buy":
            self.pos += qty
            self.cash -= notional
            self.cash -= fee
        else:
            self.pos -= qty
            self.cash += notional
            self.cash -= fee

        self.fills.append(Fill(
            ts_us=int(ts_us),
            order_id=o.order_id,
            side=o.side,
            price=px,
            qty=float(qty),
            fee=float(fee),
        ))