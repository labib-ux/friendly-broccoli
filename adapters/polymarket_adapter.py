import logging
import requests
import pandas as pd
from datetime import datetime, timezone
from typing import Any
from py_clob_client.client import ClobClient
from py_clob_client.credentials import ApiCreds
from config import POLYMARKET_API_KEY, POLYMARKET_PRIVATE_KEY
from adapters.base_adapter import BaseAdapter


POLYMARKET_HOST  = "https://clob.polymarket.com"
GAMMA_API_BASE   = "https://gamma-api.polymarket.com"
POLYGON_CHAIN_ID = 137
REQUEST_TIMEOUT  = 10
MAX_TRADES_FETCH = 1000


class PolymarketAdapter(BaseAdapter):
    """
    Adapter implementation for the Polymarket API.
    
    Provides methods to fetch historical OHLCV data, check current odds,
    submit orders, and manage portfolio state using the Polymarket CLOB
    and Gamma APIs.
    """

    def __init__(self) -> None:
        """
        Initialize the PolymarketAdapter with credentials and client instances.
        """
        creds = ApiCreds(
            api_key=POLYMARKET_API_KEY,
            api_secret=POLYMARKET_PRIVATE_KEY,
            api_passphrase=""
        )
        self.client = ClobClient(
            host=POLYMARKET_HOST,
            chain_id=POLYGON_CHAIN_ID,
            creds=creds
        )
        self.logger = logging.getLogger(__name__)

    def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a given market from Polymarket.

        Args:
            symbol: The market condition ID.
            timeframe: The time resolution ('1m', '1h', '4h', '1d').
            limit: The maximum number of bars to return.

        Returns:
            A pandas DataFrame with columns: [open, high, low, close, volume]
        """
        try:
            timeframe_map = {"1m": "1min", "1h": "1h", "4h": "4h", "1d": "1D"}
            rule = timeframe_map.get(timeframe, "1h")
            if timeframe not in timeframe_map:
                self.logger.warning("Unrecognized timeframe '%s', defaulting to 1h", timeframe)

            url = f"{GAMMA_API_BASE}/markets/{symbol}"
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            history = response.json().get("history", [])

            history_records = [
                {"timestamp": pd.to_datetime(h["t"], unit='s', utc=True), "p": float(h["p"])}
                for h in history
            ]
            price_df = pd.DataFrame(history_records)
            if not price_df.empty:
                price_df = price_df.set_index("timestamp")
                price_ohlc = price_df['p'].resample(rule).ohlc()
                price_ohlc = price_ohlc.ffill()
            else:
                price_ohlc = pd.DataFrame(columns=['open', 'high', 'low', 'close'])

            trades = self.client.get_trades(market=symbol, limit=MAX_TRADES_FETCH)
            trade_records = []
            for t in trades:
                trade_records.append({
                    "timestamp": pd.to_datetime(t.get("timestamp"), utc=True),
                    "size": float(t.get("size", 0.0))
                })
            
            trades_df = pd.DataFrame(trade_records)
            if not trades_df.empty:
                trades_df = trades_df.set_index("timestamp")
                volume_sum = trades_df['size'].resample(rule).sum().rename('volume')
                volume_sum = volume_sum.fillna(0.0)
            else:
                volume_sum = pd.Series(name='volume', dtype=float)

            final_df = pd.concat([price_ohlc, volume_sum], axis=1).astype(float)
            if 'volume' not in final_df.columns:
                final_df['volume'] = 0.0
            else:
                final_df['volume'] = final_df['volume'].fillna(0.0)
                
            final_df = final_df[['open', 'high', 'low', 'close', 'volume']]
            
            return final_df.tail(limit)

        except Exception as e:
            self.logger.error("Error in get_ohlcv", exc_info=True)
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    def get_portfolio(self) -> dict[str, Any]:
        """
        Fetch current portfolio balances and value from Polymarket.

        Returns:
            A dictionary with keys: 'cash', 'positions', 'total_value'
        """
        try:
            # Safely attempt to fetch USDC balance. 
            usdc_balance = 0.0
            try:
                # If there's an actual method in py-clob-client, use it, else default 0.0
                if hasattr(self.client, 'get_balance'):
                    bal = self.client.get_balance()
                    if hasattr(bal, 'balance'):
                        usdc_balance = float(bal.balance)
                    else:
                        usdc_balance = float(bal)
            except Exception:
                pass
                
            positions = self.get_positions()
            position_value = sum(float(pos.get("size", 0.0)) * float(pos.get("price", 0.0)) for pos in positions)
            return {
                "cash": usdc_balance,
                "positions": positions,
                "total_value": usdc_balance + position_value
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
        Submit a new trading order to Polymarket.

        Args:
            symbol: The condition ID.
            side: The order side ('buy' or 'sell').
            qty: The quantity to trade.
            order_type: The type of order.

        Returns:
            A dictionary representing the serialized order response.
        """
        try:
            side_str = "BUY" if side.lower() == "buy" else "SELL"
            price = self.get_current_price(symbol)
            
            order_args = {
                "token_id": symbol,
                "price": float(price),
                "side": side_str,
                "size": float(qty),
                "fee_rate_bps": 0
            }
            order = self.client.create_order(**order_args)
            resp = self.client.post_order(order)
            return dict(resp) if isinstance(resp, dict) else {"response": resp}
        except Exception as e:
            self.logger.error("Error in submit_order", exc_info=True)
            return {}

    def get_current_price(self, symbol: str) -> float:
        """
        Fetch the most recent available price (odds) for a condition_id on Polymarket.

        Args:
            symbol: The condition ID.

        Returns:
            The current price as a float.
        """
        try:
            return float(self.get_market_odds(symbol))
        except Exception as e:
            self.logger.error("Error in get_current_price", exc_info=True)
            return 0.0

    def cancel_all_orders(self) -> None:
        """
        Cancel all open orders on Polymarket.
        """
        try:
            self.client.cancel_all()
            self.logger.info("Successfully canceled all open orders.")
        except Exception as e:
            self.logger.error("Error in cancel_all_orders", exc_info=True)

    def is_market_open(self) -> bool:
        """
        Check if the Polymarket is open (always true).

        Returns:
            True always.
        """
        try:
            self.logger.debug("Polymarket operates 24/7 on-chain. Market is open.")
            return True
        except Exception as e:
            self.logger.error("Error in is_market_open", exc_info=True)
            return True

    def get_positions(self) -> list[dict[str, Any]]:
        """
        Fetch all currently open positions from Polymarket.

        Returns:
            A list of dictionary containing position details.
        """
        try:
            orders = self.client.get_open_orders()
            positions = []
            for order in orders:
                positions.append({
                    "condition_id": getattr(order, 'asset_id', ''),
                    "side": getattr(order, 'side', ''),
                    "size": float(getattr(order, 'size', 0.0)),
                    "price": float(getattr(order, 'price', 0.0))
                })
            return positions
        except Exception as e:
            self.logger.error("Error in get_positions", exc_info=True)
            return []

    def get_market_odds(self, condition_id: str) -> float:
        """
        Fetch the best ask price for a specific condition on Polymarket.

        Args:
            condition_id: The condition ID to query.

        Returns:
            The best ask price clamped between 0.0 and 1.0.
        """
        try:
            order_book = self.client.get_order_book(condition_id)
            if not order_book.asks:
                self.logger.warning("get_market_odds: empty order book asks for %s", condition_id)
                return 0.0
            
            best_ask = min(float(ask.price) for ask in order_book.asks)
            return float(max(0.0, min(1.0, best_ask)))
        except Exception as e:
            self.logger.error("Error in get_market_odds", exc_info=True)
            return 0.0

    def get_active_markets(self, keyword: str) -> list[dict[str, Any]]:
        """
        Get active markets matching a specific keyword.

        Args:
            keyword: The keyword to search for in market questions.

        Returns:
            A list of matches, each as a dictionary.
        """
        try:
            url = f"{GAMMA_API_BASE}/markets"
            params = {"active": "true", "closed": "false", "limit": 50}
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            markets = response.json()
            
            result = []
            keyword_lower = keyword.lower()
            for market in markets:
                if isinstance(market, dict):
                    question = market.get("question", "")
                    if keyword_lower in question.lower():
                        result.append({
                            "condition_id": market.get("conditionId", ""),
                            "question": question,
                            "end_date_iso": market.get("endDate", ""),
                            "current_yes_price": float(market.get("bestAsk", 0.0) or 0.0),
                            "volume_24h": float(market.get("volume24hr", 0.0) or 0.0)
                        })
            return result
        except Exception as e:
            self.logger.error("Error in get_active_markets", exc_info=True)
            return []
