import logging
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from data.features import FEATURE_COLUMNS

COMMISSION_RATE = 0.001
DRAWDOWN_THRESHOLD = 0.10
PENALTY_MULTIPLIER = 10.0
SHARPE_WINDOW = 50
RUIN_THRESHOLD = 0.10


class TradingEnvironment(gym.Env):
    """
    A reinforcement learning simulation environment for trading.

    This environment simulates trading operations using historical
    market data, accounting for commission rates, portfolio drawdown,
    and returns dynamically scaled rewards based on Sharpe-like metrics.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 10000.0,
        lookback_window: int = 30,
        render_mode: str | None = None
    ) -> None:
        """
        Initialize the TradingEnvironment.

        Args:
            df: Historical market data containing features and 'close' prices.
            initial_capital: Starting capital for the account.
            lookback_window: Number of historical steps to include in observations.
            render_mode: Rendering mode (e.g., 'human' or None).

        Raises:
            ValueError: If the input data contains NaN or infinite values in feature columns.
        """
        # Input validation MUST run before anything else
        assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"
        is_na = df[FEATURE_COLUMNS].isna().any()
        is_inf = df[FEATURE_COLUMNS].isin([np.inf, -np.inf]).any()
        if is_na.any() or is_inf.any():
            bad_cols = [col for col in FEATURE_COLUMNS if is_na[col] or is_inf[col]]
            raise ValueError(f"DataFrame contains NaN or infinite values in columns: {bad_cols}")

        self.df = df.reset_index(drop=True)
        self.initial_capital = float(initial_capital)
        self.lookback_window = lookback_window
        self.render_mode = render_mode

        self.feature_columns = FEATURE_COLUMNS
        n_features = len(self.feature_columns)

        # Action Space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = gym.spaces.Discrete(3)
        
        # Observation Space: Infinite bounds to prevent clipping
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(lookback_window * n_features + 3,),
            dtype=np.float32
        )

        self.logger = logging.getLogger(__name__)

        # Initialize all stateful variables to satisfy type checkers
        self.current_capital: float = self.initial_capital
        self.position_size: float = 0.0
        self.entry_price: float = 0.0
        self.trade_count: int = 0
        self.current_step: int = self.lookback_window
        self.peak_value: float = self.initial_capital
        self.portfolio_value: float = self.initial_capital
        self.reward_history: list[float] = []

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options for environment reset.

        Returns:
            A tuple of the initial observation and an info dictionary.
        """
        super().reset(seed=seed)

        self.current_capital = self.initial_capital
        self.position_size = 0.0
        self.entry_price = 0.0
        self.trade_count = 0
        self.current_step = self.lookback_window
        self.peak_value = self.initial_capital
        self.portfolio_value = self.initial_capital
        self.reward_history = []

        observation = self._get_observation()

        return observation.astype(np.float32), {"portfolio_value": self.portfolio_value}

    def _get_observation(self) -> np.ndarray:
        """
        Generate the current environment observation.

        Returns:
            The current observation array including the historical window
            and account state variables.

        Raises:
            AssertionError: If generated observation contains NaN or infinite values.
        """
        safe_step = min(self.current_step, len(self.df) - 1)
        
        window = self.df[self.feature_columns].iloc[
            max(0, safe_step - self.lookback_window) : safe_step
        ].values.flatten()
        
        # Pad with zeros if we don't have enough history
        # (This usually only happens on the very first few steps after reset)
        expected_size = self.lookback_window * len(self.feature_columns)
        if len(window) < expected_size:
            window = np.pad(window, (expected_size - len(window), 0), 'constant')
            
        capital_ratio = self.current_capital / self.initial_capital
        position_held = 1.0 if self.position_size > 0 else 0.0

        if self.position_size > 0:
            current_price = float(self.df['close'].iloc[safe_step])
            unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            unrealized_pnl_pct = 0.0

        obs = np.concatenate((window, [capital_ratio, position_held, unrealized_pnl_pct]))

        assert not np.isnan(obs).any() and not np.isinf(obs).any(), \
            f"Observation contains NaN or Inf values at indices: {np.where(np.isnan(obs) | np.isinf(obs))[0]}"

        return obs.astype(np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one step within the trading environment.

        Args:
            action: Integer representing the chosen action (0=HOLD, 1=BUY, 2=SELL).

        Returns:
            A tuple containing:
            - observation: Current environment state.
            - reward: Immediate reward for the taken action.
            - terminated: Whether the episode has reached a terminal state.
            - truncated: Whether the episode was artificially truncated (always False).
            - info: Additional dictionary with portfolio metadata.
        """
        current_price = float(self.df['close'].iloc[self.current_step])
        step_reward = 0.0

        if action == 1 and self.position_size == 0:
            # BUY
            self.position_size = (self.current_capital * (1 - COMMISSION_RATE)) / current_price
            self.entry_price = current_price
            self.current_capital = 0.0
            step_reward = -COMMISSION_RATE  # cost of entry
        elif action == 2 and self.position_size > 0:
            # SELL
            sale_value = self.position_size * current_price * (1 - COMMISSION_RATE)
            realized_pnl = sale_value - (self.position_size * self.entry_price)
            self.current_capital = sale_value
            step_reward = realized_pnl
            self.position_size = 0.0
            self.entry_price = 0.0
            self.trade_count += 1
        else:
            # HOLD (or BUY attempted while in position, or SELL attempted while flat)
            step_reward = 0.0

        # Portfolio update (runs after every action)
        self.portfolio_value = self.current_capital + (self.position_size * current_price)
        self.peak_value = max(self.peak_value, self.portfolio_value)

        # Drawdown penalty
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        if drawdown > DRAWDOWN_THRESHOLD:
            step_reward -= drawdown * PENALTY_MULTIPLIER

        # Sharpe-style reward scaling
        self.reward_history.append(step_reward)
        recent_rewards = self.reward_history[-SHARPE_WINDOW:]
        reward = float(step_reward / (np.std(recent_rewards) + 1e-9))

        # Termination conditions
        terminated = bool(
            self.current_step >= len(self.df) - 1
            or self.portfolio_value < self.initial_capital * RUIN_THRESHOLD
        )

        # Advance step
        self.current_step += 1

        info = {
            "portfolio_value": self.portfolio_value,
            "position_size": self.position_size,
            "trade_count": self.trade_count,
            "drawdown": drawdown,
            "current_capital": self.current_capital
        }

        return (self._get_observation(), reward, terminated, False, info)

    def render(self) -> None:
        """
        Render the environment state logging current step information.
        """
        if self.render_mode == "human":
            self.logger.info(
                "Step %d | Portfolio: $%.2f | Shares: %.4f | "
                "Capital: $%.2f | Trades: %d",
                self.current_step,
                self.portfolio_value,
                self.position_size,
                self.current_capital,
                self.trade_count,
            )

    def close(self) -> None:
        """
        Close the trading environment, performing any necessary cleanup.
        """
        self.logger.info("Closing TradingEnvironment.")
