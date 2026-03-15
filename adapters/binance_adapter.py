from abc import ABC, abstractmethod
import pandas as pd
from typing import Any
from adapters.base_adapter import BaseAdapter


class BinanceAdapter(BaseAdapter):
    """
    Adapter implementation for Binance API.
    
    This is an intentional stub for future implementation using the ccxt library.
    """

    def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Fetch historical OHLCV data for a given symbol."""
        # TODO: Implement using ccxt library — ccxt.binance().method_name()
        raise NotImplementedError("Binance adapter not yet implemented")

    def get_portfolio(self) -> dict[str, Any]:
        """Fetch current portfolio balances and value."""
        # TODO: Implement using ccxt library — ccxt.binance().method_name()
        raise NotImplementedError("Binance adapter not yet implemented")

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market"
    ) -> dict[str, Any]:
        """Submit a new trading order."""
        # TODO: Implement using ccxt library — ccxt.binance().method_name()
        raise NotImplementedError("Binance adapter not yet implemented")

    def get_current_price(self, symbol: str) -> float:
        """Fetch the most recent available price for a symbol."""
        # TODO: Implement using ccxt library — ccxt.binance().method_name()
        raise NotImplementedError("Binance adapter not yet implemented")

    def cancel_all_orders(self) -> None:
        """Cancel all open orders for the account."""
        # TODO: Implement using ccxt library — ccxt.binance().method_name()
        raise NotImplementedError("Binance adapter not yet implemented")

    def is_market_open(self) -> bool:
        """Check if the market for the adapter's assets is currently open."""
        # TODO: Implement using ccxt library — ccxt.binance().method_name()
        raise NotImplementedError("Binance adapter not yet implemented")

    def get_positions(self) -> list[dict[str, Any]]:
        """Fetch all currently open positions."""
        # TODO: Implement using ccxt library — ccxt.binance().method_name()
        raise NotImplementedError("Binance adapter not yet implemented")
