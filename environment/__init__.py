"""
RL Simulation Environment Module.

This module provides the TradingEnvironment class which implements a
Gymnasium-compatible reinforcement learning environment for trading
simulation. It handles the processing of historical data, action
resolution, account portfolio management, and reward calculation.
"""

from .trading_env import TradingEnvironment

__all__ = ["TradingEnvironment"]
