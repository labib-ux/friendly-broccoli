from abc import ABC, abstractmethod
import pandas as pd
from typing import Any


class BaseAdapter(ABC):
    """
    Abstract base class defining the standard interface for all trading adapters.
    
    All subclasses must implement these methods to ensure a consistent
    API across different exchanges and data providers.
    """

    @abstractmethod
    def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a given symbol.

        Args:
            symbol: The asset ticker or ID.
            timeframe: The time resolution (e.g., '1m', '1h', '1d').
            limit: The maximum number of bars to return.

        Returns:
            A pandas DataFrame with columns: [open, high, low, close, volume]
        """
        pass

    @abstractmethod
    def get_portfolio(self) -> dict[str, Any]:
        """
        Fetch current portfolio balances and value.

        Returns:
            A dictionary with exactly these keys: 'cash', 'positions', 'total_value'
        """
        pass

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market"
    ) -> dict[str, Any]:
        """
        Submit a new trading order.

        Args:
            symbol: The asset ticker or ID.
            side: The order side ('buy' or 'sell').
            qty: The quantity to trade.
            order_type: The type of order (default is 'market').

        Returns:
            A dictionary representing the serialized order response.
        """
        pass

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """
        Fetch the most recent available price for a symbol.

        Args:
            symbol: The asset ticker or ID.

        Returns:
            The current price as a float.
        """
        pass

    @abstractmethod
    def cancel_all_orders(self) -> None:
        """
        Cancel all open orders for the account.
        """
        pass

    @abstractmethod
    def is_market_open(self) -> bool:
        """
        Check if the market for the adapter's assets is currently open.

        Returns:
            True if the market is open, False otherwise.
        """
        pass

    @abstractmethod
    def get_positions(self) -> list[dict[str, Any]]:
        """
        Fetch all currently open positions.

        Returns:
            A list of dicts containing position details.
        """
        pass
