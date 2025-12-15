from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd

class BTLogger:
    def __init__(self):
        self.rows = []

    def log(self, *, ts_us: int, event_i: int, etype: str, action: str, **kwargs):
        self.rows.append({
            "ts_us": int(ts_us),
            "event_i": int(event_i),
            "etype": str(etype),
            "action": str(action),
            **kwargs
        })

    def to_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows)
        if len(df):
            df["ts"] = pd.to_datetime(df["ts_us"], unit="us", utc=True)
            df = df.sort_values(["ts_us", "event_i"], kind="mergesort").reset_index(drop=True)
        return df
    
@dataclass
class SWRParams:
    # quoting
    quote_both_sides: bool = True
    layers: int = 1

    # sizing
    qty_per_side: float = 3.00   # contracts (HYPE)
    qty_step: float = 0.01       # already in sim for HYPE

    # inventory
    Q_max: float = 15.00

    # spread trigger
    baseline_window_ms: int = 200
    k_enter: float = 7.0
    k_exit: float = 3.0
    spread_floor_bps: float = 0.5

    # toxicity window + thresholds (starter defaults)
    tox_window_ms: int = 200
    net_signed_vol_max: float = 10.0
    trade_count_max: int = 20
    total_vol_max: float = 50.0

    # execution
    tick_size: float = 0.001
    ttl_ms: int = 100

    min_regime_hold_ms: int = 150
    enter_need_quotes: int = 1
    cooldown_ms: int = 300

class ToxicityTracker:
    def __init__(self, window_us: int):
        self.window_us = int(window_us)
        self.q = deque()  # (ts_us, signed_qty, qty)
        self.count = 0
        self.total = 0.0
        self.net = 0.0

    def update(self, ts_us: int, side: str, qty: float):
        signed = qty if side == "buy" else -qty
        self.q.append((ts_us, signed, qty))
        self.count += 1
        self.total += qty
        self.net += signed
        self._evict(ts_us)

    def _evict(self, ts_us: int):
        cutoff = ts_us - self.window_us
        while self.q and self.q[0][0] < cutoff:
            _, signed, qty = self.q.popleft()
            self.count -= 1
            self.total -= qty
            self.net -= signed

    def snapshot(self):
        return self.count, self.total, self.net

@dataclass
class Order:
    order_id: int
    side: str # "buy" or "sell"
    price: float
    qty: float
    ts_place_us: int
    live_us: int # ts when order becomes active (latency)
    queue_ahead: float # how much volume is ahead of you at that level
    qty_remaining: float = field(init=False)
    is_active: bool = False
    is_done: bool = False

    def __post_init__(self):
        self.qty_remaining = float(self.qty)


@dataclass
class Fill:
    ts_us: int
    order_id: int
    side: str
    price: float
    qty: float
    fee: float
    source: str

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
    # def on_quote(self, quote_row: pd.Series):
    #     self.book.update_from_quote_row(quote_row, use_depth=True)
    def on_quote(self, quote_row: pd.Series):
        """
        Mode 2: quote update can also generate fills (quote-cross heuristic).
        """
        ts_us = int(quote_row["ts_us"])

        # 1) Activate any orders that are now live (latency elapsed)
        self._activate_orders(ts_us)

        # 2) Update book snapshot
        self.book.update_from_quote_row(quote_row, use_depth=True)

        # 3) Quote-cross fills (conservative touch-fill)
        # BUY fills if best_ask <= order.price
        # SELL fills if best_bid >= order.price
        if not self.book.is_valid() or not self.active_ids:
            return

        best_bid = self.book.snap.best_bid
        best_ask = self.book.snap.best_ask
        top_bid_qty = float(self.book.snap.bid_qty)   # bids[0].amount
        top_ask_qty = float(self.book.snap.ask_qty)   # asks[0].amount

        # iterate on a copy because we might remove from active_ids
        for oid in list(self.active_ids):
            o = self.orders[oid]
            if o.is_done or (not o.is_active):
                continue

            if o.side == "buy":
                if best_ask <= o.price + 1e-12:
                    fill_qty = min(o.qty_remaining, top_ask_qty)
                    fill_qty = self._round_qty(fill_qty)
                    if fill_qty > 0:
                        # Conservative: fill at our limit price (not price-improved)
                        self._record_fill(ts_us, o, fill_qty, maker=True, source="quote_cross")
                        o.qty_remaining -= fill_qty
                        if o.qty_remaining <= 1e-12:
                            o.is_done = True
                            o.is_active = False
                            self.active_ids.discard(o.order_id)

            else:  # sell
                if best_bid >= o.price - 1e-12:
                    fill_qty = min(o.qty_remaining, top_bid_qty)
                    fill_qty = self._round_qty(fill_qty)
                    if fill_qty > 0:
                        self._record_fill(ts_us, o, fill_qty, maker=True, source="quote_cross")
                        o.qty_remaining -= fill_qty
                        if o.qty_remaining <= 1e-12:
                            o.is_done = True
                            o.is_active = False
                            self.active_ids.discard(o.order_id)


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
                self._record_fill(ts_us, o, fill_qty, maker=True, source="trade_through")
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
            fill_qty = self._round_qty(fill_qty)
            if fill_qty <= 0:
                return
            
            if fill_qty > 0:
                self._record_fill(ts_us, o, fill_qty, maker=True, source="trade_through")
                o.qty_remaining -= fill_qty
                remaining -= fill_qty

                if o.qty_remaining <= 1e-12:
                    o.is_done = True
                    o.is_active = False
                    self.active_ids.discard(o.order_id)

    def _record_fill(self, ts_us: int, o: Order, qty: float, maker: bool, source: str):
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
            source=str(source)
        ))

class OpportunisticSpreadWidenMM:
    def __init__(self, params: SWRParams, logger: Optional[BTLogger] = None):
        self.p = params
        self.logger = logger
        self.bid_oid = None
        self.ask_oid = None
        self.tox = ToxicityTracker(window_us=params.tox_window_ms * 1000)

        self.in_regime = False  # spread-widen regime state (hysteresis)
        self.regime_enter_ts_us: Optional[int] = None
        self.enter_streak = 0
        self.cooldown_until_ts_us = 0

    def _slog(self, ctx, action, **kwargs):
        if self.logger is None:
            return
        self.logger.log(
            ts_us=ctx["ts_us"],
            event_i=ctx["event_i"],
            etype=ctx["etype"],
            action=action,
            **kwargs
        )

    # ----- helpers -----
    def _round_tick(self, px: float) -> float:
        t = round(px / self.p.tick_size)
        return float(t) * self.p.tick_size

    def _desired_quotes(self, book) -> Tuple[float, float]:
        """
        Place inside spread by 1 tick.
        If spread too tight to place both, return (nan, nan) and strategy will not quote.
        """
        best_bid = book.snap.best_bid
        best_ask = book.snap.best_ask

        bid_px = self._round_tick(best_bid + self.p.tick_size)
        ask_px = self._round_tick(best_ask - self.p.tick_size)

        # Need strict inside: bid < ask
        if bid_px >= ask_px - 1e-12:
            return (np.nan, np.nan)
        return (bid_px, ask_px)

    def _tox_ok(self) -> bool:
        count, total, net = self.tox.snapshot()
        if abs(net) >= self.p.net_signed_vol_max:
            return False
        if count >= self.p.trade_count_max:
            return False
        if total >= self.p.total_vol_max:
            return False
        return True

    def _order_age_us(self, sim, oid: int, now_us: int) -> int:
        o = sim.orders.get(oid)
        if o is None:
            return 10**18
        return now_us - int(o.ts_place_us)

    def _cancel_if_exists(self, sim, oid):
        if oid is not None:
            sim.cancel(oid)

    # ----- callbacks -----
    def on_trade(self, sim, trade_row: pd.Series, ts_us: int):
        # update toxicity stats first
        self.tox.update(ts_us, str(trade_row["side"]).lower(), float(trade_row["amount"]))

        # inventory emergency gating (optional immediate defense)
        if abs(sim.pos) >= self.p.Q_max:
            # cancel the side that increases inventory risk
            if sim.pos > 0:
                # long -> cancel bid
                self._cancel_if_exists(sim, self.bid_oid)
                self.bid_oid = None
            elif sim.pos < 0:
                # short -> cancel ask
                self._cancel_if_exists(sim, self.ask_oid)
                self.ask_oid = None

    def on_quote(self, sim, quote_row: pd.Series, ts_us: int, ctx: dict):
        # spread regime detection (enter/exit with hysteresis)
        spread_bps = float(quote_row["spread_bps"])
        med = float(quote_row.get("spread_med_bps", np.nan))

        prev = self.in_regime

        # If median not available early in day, skip regime
        if not np.isfinite(med) or med <= 0:
            self.in_regime = False
        else:
            # enter = spread_bps > max(self.p.spread_floor_bps, self.p.k_enter * med)
            # exit_ = spread_bps < max(self.p.spread_floor_bps, self.p.k_exit * med)

            enter_cond = spread_bps > max(self.p.spread_floor_bps, self.p.k_enter * med)
            exit_cond = spread_bps < max(self.p.spread_floor_bps, self.p.k_exit * med)

            if ts_us < self.cooldown_until_ts_us:
                # force no entry during cooldown
                enter_cond = False

            if enter_cond:
                self.enter_streak += 1
            else:
                self.enter_streak = 0

            if (not self.in_regime) and (self.enter_streak >= self.p.enter_need_quotes):
            #if (not self.in_regime) and enter:
                self.in_regime = True
                self.regime_enter_ts_us = ts_us
                self.enter_streak = 0
            elif self.in_regime and exit_cond:
                # minimum hold protection
                hold_us = self.p.min_regime_hold_ms * 1000
                if self.regime_enter_ts_us is not None:
                    if ts_us - self.regime_enter_ts_us < hold_us:
                        # still in mandatory hold window -> ignore exit
                        self._slog(
                            ctx,
                            "REGIME",
                            reason="exit_suppressed_min_hold",
                            elapsed_ms=(ts_us - self.regime_enter_ts_us) / 1000,
                            pos=float(sim.pos),
                        )
                        pass
                    else:
                        self.in_regime = False
                        self.regime_enter_ts_us = None
                        self.cooldown_until_ts_us = ts_us + self.p.cooldown_ms * 1000
        
        if (not prev) and self.in_regime:
            # self._slog(ctx, "REGIME", reason="enter_regime",
            #            spread_bps=spread_bps, med_bps=med, pos=float(sim.pos))
            
            # thr = max(self.p.spread_floor_bps, self.p.k_enter * med)
            # self._slog(ctx, "REGIME", reason="enter_regime",
            #         spread_bps=spread_bps, med_bps=med,
            #         enter_thr_bps=thr,
            #         enter_excess_bps=spread_bps - thr)
            thr = max(self.p.spread_floor_bps, self.p.k_enter * med)
            self._slog(
                ctx, "REGIME", reason="enter_regime",
                spread_bps=spread_bps, med_bps=med, pos=float(sim.pos),
                enter_thr_bps=thr,
                enter_excess_bps=spread_bps - thr
            )

        if prev and (not self.in_regime):
            self._slog(ctx, "REGIME", reason="exit_regime",
                       spread_bps=spread_bps, med_bps=med, pos=float(sim.pos))
            if self.bid_oid is not None:
                self._slog(ctx, "CANCEL", reason="exit_regime", side="buy", order_id=int(self.bid_oid), pos=float(sim.pos))
            if self.ask_oid is not None:
                self._slog(ctx, "CANCEL", reason="exit_regime", side="sell", order_id=int(self.ask_oid), pos=float(sim.pos))

        # If not in opportunity regime, cancel and do nothing
        if not self.in_regime:
            self._cancel_if_exists(sim, self.bid_oid); self.bid_oid = None
            self._cancel_if_exists(sim, self.ask_oid); self.ask_oid = None
            return

        # Toxicity filter gating
        if not self._tox_ok():
            self._slog(ctx, "CANCEL_ALL", reason="tox_fail", pos=float(sim.pos))
            count, total, net = self.tox.snapshot()
            # Log reason also
            self._slog(ctx, "CANCEL_ALL", reason="tox_fail",
                    tox_count=count, tox_total=total, tox_net=net,
                    spread_bps=spread_bps, med_bps=med, pos=float(sim.pos))
            
            if self.bid_oid is not None:
                self._slog(ctx, "CANCEL", reason="tox_fail", side="buy", order_id=int(self.bid_oid), pos=float(sim.pos))
            if self.ask_oid is not None:
                self._slog(ctx, "CANCEL", reason="tox_fail", side="sell", order_id=int(self.ask_oid), pos=float(sim.pos))
            
            self._cancel_if_exists(sim, self.bid_oid)
            self.bid_oid = None

            self._cancel_if_exists(sim, self.ask_oid)
            self.ask_oid = None
            
            return

        # Inventory gating
        allow_bid = True
        allow_ask = True
        if sim.pos >= self.p.Q_max:
            allow_bid = False   # long -> don't buy more
            self._slog(ctx, "GATE", reason="inv_gate", gated_side="buy", pos=float(sim.pos), Q_max=float(self.p.Q_max))
        if sim.pos <= -self.p.Q_max:
            allow_ask = False   # short -> don't sell more
            self._slog(ctx, "GATE", reason="inv_gate", gated_side="sell", pos=float(sim.pos), Q_max=float(self.p.Q_max))

        bid_px, ask_px = self._desired_quotes(sim.book)
        if not np.isfinite(bid_px) or not np.isfinite(ask_px):
            # can't place inside spread (too tight) -> cancel
            self._slog(ctx, "CANCEL_ALL", reason="tight_spread", pos=float(sim.pos),
                       best_bid=float(sim.book.snap.best_bid), best_ask=float(sim.book.snap.best_ask))
            self._cancel_if_exists(sim, self.bid_oid); self.bid_oid = None
            self._cancel_if_exists(sim, self.ask_oid); self.ask_oid = None
            return

        # cancel/replace logic based on TTL or wrong price
        ttl_us = self.p.ttl_ms * 1000

        if allow_bid:
            replace = False
            cancel_reason = None
            place_reason = None

            if self.bid_oid is None or self.bid_oid not in sim.orders:
                replace = True
                place_reason = "enter_regime"  # first time / missing

            else:
                o = sim.orders[self.bid_oid]
                if o.is_done:
                    self.bid_oid = None
                    replace = True
                    place_reason = "replaced_done"
                else:
                    if abs(o.price - bid_px) > 1e-12:
                        replace = True
                        cancel_reason = "price_change"
                        place_reason = "price_change"
                    elif self._order_age_us(sim, self.bid_oid, ts_us) > ttl_us:
                        replace = True
                        cancel_reason = "ttl"
                        place_reason = "ttl"

            if replace:
                if self.bid_oid is not None:
                    self._slog(ctx, "CANCEL", reason=cancel_reason or place_reason,
                            side="buy", order_id=int(self.bid_oid),
                            old_price=float(sim.orders[self.bid_oid].price) if self.bid_oid in sim.orders else np.nan,
                            new_price=float(bid_px),
                            pos=float(sim.pos))
                    self._cancel_if_exists(sim, self.bid_oid)

                oid = sim.place_limit("buy", bid_px, self.p.qty_per_side, ts_us)
                self.bid_oid = oid
                self._slog(ctx, "PLACE", reason=place_reason or "enter_regime",
                           side="buy", order_id=int(oid), price=float(bid_px),
                           qty=float(self.p.qty_per_side), pos=float(sim.pos))
        else:
            if self.bid_oid is not None:
                self._slog(ctx, "CANCEL", reason="inv_gate", side="buy", order_id=int(self.bid_oid), pos=float(sim.pos))
            self._cancel_if_exists(sim, self.bid_oid)
            self.bid_oid = None

        if allow_ask:
            replace = False
            cancel_reason = None
            place_reason = None
            if self.ask_oid is None or self.ask_oid not in sim.orders:
                replace = True
                place_reason = "enter_regime"  # first time / missing
            else:
                o = sim.orders[self.ask_oid]
                if o.is_done:
                    self.ask_oid = None
                    replace = True
                    place_reason = "replaced_done"
                else:
                    if abs(o.price - ask_px) > 1e-12:
                        replace = True
                        cancel_reason = "price_change"
                        place_reason = "price_change"
                    elif self._order_age_us(sim, self.ask_oid, ts_us) > ttl_us:
                        replace = True
                        cancel_reason = "ttl"
                        place_reason = "ttl"

            if replace:
                if self.ask_oid is not None:
                    self._slog(ctx, "CANCEL", reason=cancel_reason or place_reason,
                            side="sell", order_id=int(self.ask_oid),
                            old_price=float(sim.orders[self.ask_oid].price) if self.ask_oid in sim.orders else np.nan,
                            new_price=float(ask_px),
                            pos=float(sim.pos))
                    self._cancel_if_exists(sim, self.ask_oid)
                oid = sim.place_limit("sell", ask_px, self.p.qty_per_side, ts_us)
                self.ask_oid = oid
                self._slog(ctx, "PLACE", reason=place_reason or "enter_regime",
                           side="sell", order_id=int(oid), price=float(ask_px),
                           qty=float(self.p.qty_per_side), pos=float(sim.pos))
                #self.ask_oid = sim.place_limit("sell", ask_px, self.p.qty_per_side, ts_us)
        else:
            if self.ask_oid is not None:
                self._slog(ctx, "CANCEL", reason="inv_gate", side="sell", order_id=int(self.ask_oid), pos=float(sim.pos))
            self._cancel_if_exists(sim, self.ask_oid)
            self.ask_oid = None

def run_backtest(events_df, quotes_df, trades_df, sim, strategy, logger: BTLogger):
    equity = []
    ts_list = []

    for i, ev in enumerate(events_df.itertuples(index=False)):
        ts_us = int(ev.ts_us)
        ctx = {"event_i": i, "etype": ev.etype, "ts_us": ts_us}

        prev_nf = len(sim.fills)

        if ev.etype == "quote":
            qrow = quotes_df.iloc[ev.qidx]
            sim.on_quote(qrow)                # may fill (quote_cross)
            strategy.on_quote(sim, qrow, int(ev.ts_us), ctx) # strategy logs place/cancel reasons

        else:
            trow = trades_df.iloc[ev.tidx]
            sim.on_trade(trow, ts_us=ts_us)   # may fill (trade_through)
            strategy.on_trade(sim, trow, ts_us=int(ev.ts_us))

        # log NEW fills with event pointer + source
        if len(sim.fills) > prev_nf:
            for f in sim.fills[prev_nf:]:
                logger.log(
                    ts_us=f.ts_us,
                    event_i=i,
                    etype=ev.etype,
                    action="FILL",
                    fill_source=f.source,
                    order_id=int(f.order_id),
                    side=str(f.side),
                    price=float(f.price),
                    qty=float(f.qty),
                    fee=float(f.fee),
                    pos=float(sim.pos),
                    cash=float(sim.cash),
                )

        # mark-to-market
        if sim.book.is_valid():
            mtm = sim.cash + sim.pos * sim.book.snap.mid
            equity.append(mtm)
            ts_list.append(ts_us)

    eq = pd.DataFrame({"ts_us": ts_list, "equity": equity})
    eq["ts"] = pd.to_datetime(eq["ts_us"], unit="us", utc=True)
    return eq