"""
Risk Management Module.

Provides the RiskManager class which enforces risk constraints, calculates
dynamic position sizes including Kelly criteria, and manages drawdown limits
and circuit breakers.
"""

from .manager import RiskManager

__all__ = ["RiskManager"]
