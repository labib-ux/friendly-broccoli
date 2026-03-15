"""
Database Module for Trading Bot.

Handles persistent storage of trades and agent decisions, using SQLAlchemy.
Provides models and functions to interact with the sqlite database safely.
"""

from .logger import (
    init_db,
    log_trade,
    log_decision,
    get_trade_history,
    get_pnl_summary,
    Trade,
    AgentDecision
)

__all__ = [
    "init_db",
    "log_trade",
    "log_decision",
    "get_trade_history",
    "get_pnl_summary",
    "Trade",
    "AgentDecision"
]
