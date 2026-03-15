import logging


class RiskManager:
    """
    RiskManager handles risk constraints, position sizing, and stop-loss logic.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_position_pct: float = 0.2,
        max_daily_drawdown: float = 0.05
    ) -> None:
        """
        Initialize the RiskManager.

        Args:
            initial_capital: The starting capital of the account.
            max_position_pct: Maximum percentage of capital to risk per trade.
            max_daily_drawdown: Maximum allowed daily drawdown percentage.
        """
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.max_daily_drawdown = max_daily_drawdown
        self.logger = logging.getLogger(__name__)

    def calculate_position_size(
        self, capital: float, price: float, confidence: float = 1.0
    ) -> float:
        """
        Calculate fixed fractional position size.

        Args:
            capital: Current available capital.
            price: Current entry price.
            confidence: Confidence multiplier (default 1.0).

        Returns:
            The calculated position size as a float.
        """
        if price <= 0:
            self.logger.warning("calculate_position_size: invalid price %s, returning 0.0", price)
            return 0.0

        size = (capital * self.max_position_pct * confidence) / price
        return max(0.0, float(size))

    def kelly_size(
        self,
        win_prob: float,
        win_return: float,
        loss_return: float,
        capital: float,
        price: float
    ) -> float:
        """
        Calculate position size using the Half-Kelly criterion.

        Args:
            win_prob: Probability of a winning trade [0.0 - 1.0].
            win_return: Expected positive return percentage.
            loss_return: Expected negative return percentage.
            capital: Current available capital.
            price: Current entry price.

        Returns:
            The calculated Half-Kelly position size as a float.
        """
        if win_return <= 0 or price <= 0:
            self.logger.warning(
                "kelly_size: invalid inputs win_return=%s price=%s, returning 0.0",
                win_return, price
            )
            return 0.0

        kelly = (win_prob * win_return - (1 - win_prob) * loss_return) / win_return
        
        # Clamp to [0, 1] and halve
        half_kelly = max(0.0, min(1.0, kelly * 0.5))
        
        size = (capital * half_kelly) / price
        return max(0.0, float(size))

    def check_circuit_breaker(
        self, current_value: float, day_start_value: float
    ) -> bool:
        """
        Check if the daily drawdown exceeds the maximum allowed threshold.

        Args:
            current_value: Current portfolio value.
            day_start_value: Portfolio value at the start of the day.

        Returns:
            True if trading should be halted, False otherwise.
        """
        if day_start_value <= 0:
            self.logger.warning("check_circuit_breaker: invalid day_start_value %s", day_start_value)
            return False

        drawdown = (day_start_value - current_value) / day_start_value

        if drawdown > self.max_daily_drawdown:
            self.logger.critical(
                "CIRCUIT BREAKER TRIGGERED! Drawdown: %.2f%% > Threshold: %.2f%%",
                drawdown * 100,
                self.max_daily_drawdown * 100
            )
            return True

        return False

    def should_stop_loss(
        self,
        entry_price: float,
        current_price: float,
        side: str,
        stop_pct: float = 0.02
    ) -> bool:
        """
        Check if a stop-loss condition has been met for an open position.

        Args:
            entry_price: The price at which the position was entered.
            current_price: The current market price.
            side: 'long' or 'short'.
            stop_pct: Maximum allowed loss percentage (default 2%).

        Returns:
            True if stop loss is triggered, False otherwise.
        """
        if side == "long":
            return current_price < entry_price * (1 - stop_pct)
        elif side == "short":
            return current_price > entry_price * (1 + stop_pct)
        else:
            self.logger.warning("should_stop_loss: unrecognized side value '%s'", side)
            return False
