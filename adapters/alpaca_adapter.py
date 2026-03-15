import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Any
from alpaca.trading.client import TradingClient
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
from adapters.base_adapter import BaseAdapter


class AlpacaAdapter(BaseAdapter):
    """
    Adapter implementation for Alpaca Crypto API.
    
    Provides methods to fetch historical OHLCV data, check current crypto prices,
    submit orders, and manage portfolio state using the Alpaca SDK.
    """

    def __init__(self) -> None:
        """
        Initialize the AlpacaAdapter with credentials and client instances.
        """
        paper = "paper" in ALPACA_BASE_URL.lower()
        self.trading_client = TradingClient(
            ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=paper
        )
        self.data_client = CryptoHistoricalDataClient(
            ALPACA_API_KEY, ALPACA_SECRET_KEY
        )
        self.logger = logging.getLogger(__name__)

    def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a given symbol from Alpaca.

        Args:
            symbol: The asset ticker.
            timeframe: The time resolution ('1m', '1h', '4h', '1d').
            limit: The maximum number of bars to return.

        Returns:
            A pandas DataFrame with columns: [open, high, low, close, volume]
        """
        try:
            if timeframe == "1m":
                tf = TimeFrame.Minute
                delta = timedelta(minutes=limit)
            elif timeframe == "1h":
                tf = TimeFrame(1, TimeFrameUnit.Hour)
                delta = timedelta(hours=limit)
            elif timeframe == "4h":
                tf = TimeFrame(4, TimeFrameUnit.Hour)
                delta = timedelta(hours=limit * 4)
            elif timeframe == "1d":
                tf = TimeFrame.Day
                delta = timedelta(days=limit)
            else:
                self.logger.warning("Unrecognized timeframe '%s', defaulting to 1h", timeframe)
                tf = TimeFrame.Hour
                delta = timedelta(hours=limit)

            start_time = datetime.utcnow() - delta

            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start_time
            )
            bars = self.data_client.get_crypto_bars(request)
            
            df = bars.df
            df = df.reset_index()
            df = df.drop(columns=['symbol'], errors='ignore')
            df = df.set_index('timestamp')
            df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            
            return df.tail(limit)
        except Exception as e:
            self.logger.error("Error in get_ohlcv", exc_info=True)
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    def get_portfolio(self) -> dict[str, Any]:
        """
        Fetch current portfolio balances and value from Alpaca.

        Returns:
            A dictionary with keys: 'cash', 'positions', 'total_value'
        """
        try:
            account = self.trading_client.get_account()
            return {
                "cash": float(account.cash),
                "positions": self.get_positions(),
                "total_value": float(account.portfolio_value)
            }
        except Exception as e:
            self.logger.error("Error in get_portfolio", exc_info=True)
            return {"cash": 0.0, "positions": [], "total_value": 0.0}

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market"
    ) -> dict[str, Any]:
        """
        Submit a new trading order to Alpaca.

        Args:
            symbol: The asset ticker.
            side: The order side ('buy' or 'sell').
            qty: The quantity to trade.
            order_type: The type of order (default is 'market').

        Returns:
            A dictionary representing the serialized order response.
        """
        try:
            side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side_enum,
                time_in_force=TimeInForce.GTC
            )
            order = self.trading_client.submit_order(request)
            # Serialize the order correctly as a dict. If it's a pydantic model, model_dump() or dict() works.
            # Alpaca SDK models typically support built-in casting to dict.
            return dict(order)
        except Exception as e:
            self.logger.error("Error in submit_order", exc_info=True)
            return {}

    def get_current_price(self, symbol: str) -> float:
        """
        Fetch the most recent available price for a symbol on Alpaca.

        Args:
            symbol: The asset ticker.

        Returns:
            The current price as a float.
        """
        try:
            start_time = datetime.utcnow() - timedelta(minutes=5)
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=start_time
            )
            bars = self.data_client.get_crypto_bars(request)
            df = bars.df.reset_index()

            if df.empty:
                self.logger.warning("get_current_price: no bars returned for symbol %s", symbol)
                return 0.0

            return float(df['close'].iloc[-1])
        except Exception as e:
            self.logger.error("Error in get_current_price", exc_info=True)
            return 0.0

    def cancel_all_orders(self) -> None:
        """
        Cancel all open orders on Alpaca.
        """
        try:
            self.trading_client.cancel_orders()
            self.logger.info("Successfully canceled all open orders on Alpaca.")
        except Exception as e:
            self.logger.error("Error in cancel_all_orders", exc_info=True)

    def is_market_open(self) -> bool:
        """
        Check if the crypto market is open (always true).

        Returns:
            True always.
        """
        try:
            self.logger.debug("Crypto markets never close. Always return True.")
            return True
        except Exception as e:
            self.logger.error("Error in is_market_open", exc_info=True)
            return True

    def get_positions(self) -> list[dict[str, Any]]:
        """
        Fetch all currently open positions from Alpaca.

        Returns:
            A list of dictionary containing position details.
        """
        try:
            positions = self.trading_client.get_all_positions()
            return [
                {
                    "symbol": pos.symbol,
                    "qty": float(pos.qty),
                    "market_value": float(pos.market_value),
                    "avg_entry_price": float(pos.avg_entry_price)
                }
                for pos in positions
            ]
        except Exception as e:
            self.logger.error("Error in get_positions", exc_info=True)
            return []
