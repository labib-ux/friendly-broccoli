"""
Module for fetching historical market data via adapters.
"""

import logging
import pandas as pd
from typing import Any

logger = logging.getLogger(__name__)

def fetch_historical_data(adapter: Any, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """
    Fetches historical OHLCV data using the provided adapter.

    Args:
        adapter (Any): The exchange adapter instance.
        symbol (str): The trading pair symbol.
        timeframe (str): The interval/timeframe (e.g., '1h', '1d').
        limit (int): The number of data points to fetch.

    Returns:
        pd.DataFrame: A DataFrame containing historical data, or an empty DataFrame on failure.
    """
    try:
        df = adapter.get_ohlcv(symbol, timeframe, limit)
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        logger.warning(f"Adapter returned an empty DataFrame for {symbol}.")
        return pd.DataFrame()

    required_columns = {'open', 'high', 'low', 'close', 'volume'}
    df_cols_lower = set(df.columns.str.lower())
    
    if not required_columns.issubset(df_cols_lower):
        logger.warning(
            f"Returned DataFrame for {symbol} missing required OHLCV columns. "
            f"Found: {df.columns.tolist()}"
        )
        return pd.DataFrame()

    return df
