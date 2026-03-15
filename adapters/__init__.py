"""
Pluggable Adapter Pattern Module for Trading Adapters.

This module exports the base adapter and various exchange-specific
implementations (Alpaca, Polymarket, Binance) designed to provide a
unified interface for data fetching and order execution.
"""

from .base_adapter import BaseAdapter
from .alpaca_adapter import AlpacaAdapter
from .polymarket_adapter import PolymarketAdapter
from .binance_adapter import BinanceAdapter

__all__ = [
    "BaseAdapter",
    "AlpacaAdapter",
    "PolymarketAdapter",
    "BinanceAdapter",
]
