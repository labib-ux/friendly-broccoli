"""
Live Execution and Scheduling Module.

Provides the primary trading engine cycle function `run_trading_cycle`
designed to be reliably triggered by the APScheduler for real-time market
data processing, agent prediction, and adapter execution.
"""

from .jobs import run_trading_cycle

__all__ = ["run_trading_cycle"]
