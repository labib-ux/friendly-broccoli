"""
Module for computing technical indicators and engineering features for the RL agent.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
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
    
    # STEP 1: Compute indicators using pandas-ta
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2.0, append=True)
    df.ta.ema(length=9, append=True)
    df.ta.ema(length=21, append=True)
    
    # SMA of volume column(length=20)
    # Using pandas-ta sma on the volume column
    if 'volume' in df.columns:
        df['volume_sma20'] = ta.sma(df['volume'], length=20)
    elif 'Volume' in df.columns:
        df['volume_sma20'] = ta.sma(df['Volume'], length=20)
    
    # Custom percentage changes
    if 'close' in df.columns:
        df['price_change_pct'] = df['close'].pct_change()
        df['high_low_range_pct'] = (df['high'] - df['low']) / df['low']
    elif 'Close' in df.columns:
        df['price_change_pct'] = df['Close'].pct_change()
        df['high_low_range_pct'] = (df['High'] - df['Low']) / df['Low']

    # STEP 2: Rename all pandas-ta output columns to clean names
    rename_cols = {
        'RSI_14': 'rsi',
        'MACD_12_26_9': 'macd_line',
        'MACDh_12_26_9': 'macd_hist',
        'MACDs_12_26_9': 'macd_signal',
        'BBL_20_2.0': 'bb_lower',
        'BBM_20_2.0': 'bb_mid',
        'BBU_20_2.0': 'bb_upper',
        'EMA_9': 'ema9',
        'EMA_21': 'ema21'
    }
    df.rename(columns=rename_cols, inplace=True)
    
    # Drop BBB_20_2.0 and BBP_20_2.0 columns
    drop_cols = ['BBB_20_2.0', 'BBP_20_2.0']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

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
