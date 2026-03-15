"""
Module for computing technical indicators and engineering features for the RL agent.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
import ta
from typing import Tuple, Optional
from sklearn.preprocessing import MinMaxScaler

from config import MODEL_SAVE_PATH

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    'rsi', 'macd_line', 'macd_hist', 'macd_signal',
    'bb_lower', 'bb_mid', 'bb_upper', 'ema9', 'ema21',
    'volume_sma20', 'price_change_pct', 'high_low_range_pct'
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical indicators and features from an OHLCV DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing OHLCV data.

    Returns:
        pd.DataFrame: A DataFrame with the engineered features.
    """
    logger.info("Starting feature engineering...")
    
    # Optional: ensure columns are easily accessible
    # Assumes standard open, high, low, close, volume names
    
    # STEP 1: Compute indicators using ta
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd_line']   = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist']   = macd.macd_diff()
    
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_mid']   = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    
    df['ema9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
    
    df['volume_sma20'] = df['volume'].rolling(window=20).mean()
    
    # Custom percentage changes
    df['price_change_pct']   = df['close'].pct_change()
    df['high_low_range_pct'] = (df['high'] - df['low']) / df['low']

    # STEP 4: NaN/inf safety (order matters)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # assert df.isnull().sum().sum() == 0, raise ValueError showing which columns still have nulls if assertion fails
    try:
        assert df.isnull().sum().sum() == 0
    except AssertionError:
        null_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"DataFrame still contains null values in columns: {null_cols}")

    return df


def scale_features(df: pd.DataFrame, scaler: Optional[MinMaxScaler] = None) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Scales the feature columns using MinMaxScaler.

    Args:
        df (pd.DataFrame): The DataFrame containing the features.
        scaler (Optional[MinMaxScaler]): Pre-fitted scaler instance. If None, fits a new scaler.

    Returns:
        Tuple[pd.DataFrame, MinMaxScaler]: The DataFrame with scaled features and the active scaler.
    """
    logger.info("Scaling features...")
    
    # Ensure FEATURE_COLUMNS exist
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        logger.error(f"Missing feature columns for scaling: {missing_cols}")
        raise ValueError(f"Missing feature columns for scaling: {missing_cols}")

    if scaler is None:
        scaler = MinMaxScaler()
        df[FEATURE_COLUMNS] = scaler.fit_transform(df[FEATURE_COLUMNS])
        
        # Save fitted scaler to disk at MODEL_SAVE_PATH/feature_scaler.pkl
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        scaler_file = os.path.join(MODEL_SAVE_PATH, "feature_scaler.pkl")
        try:
            joblib.dump(scaler, scaler_file)
            logger.info(f"Fitted scaler successfully saved to {scaler_file}")
        except Exception as e:
            logger.error(f"Failed to save scaler at {scaler_file}: {e}")
            raise
    else:
        df[FEATURE_COLUMNS] = scaler.transform(df[FEATURE_COLUMNS])

    # Log a warning if any scaled value is outside [0, 1]
    if (df[FEATURE_COLUMNS].max().max() > 1.0 + 1e-7) or (df[FEATURE_COLUMNS].min().min() < -1e-7):
        logger.warning("Some scaled feature values are outside the [0, 1] range.")

    return df, scaler
