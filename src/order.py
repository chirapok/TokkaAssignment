from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd

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