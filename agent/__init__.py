"""
RL Agent Module.

Provides the reinforcement learning agent implementations, including the custom
policy feature extractor and the primary TradingAgent trainer using PPO.
"""

from .policy import CustomFeatureExtractor
from .trainer import TradingAgent

__all__ = ["CustomFeatureExtractor", "TradingAgent"]
