import logging
import statistics
from datetime import datetime, timezone

from sqlalchemy import create_engine, String, Float, Integer, DateTime
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, sessionmaker, Session
)

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class Trade(Base):
    """
    SQLAlchemy Model representing a single executed trade log.
    """
    __tablename__ = 'trades'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )
    adapter: Mapped[str] = mapped_column(String)
    symbol: Mapped[str] = mapped_column(String)
    side: Mapped[str] = mapped_column(String)
    qty: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    portfolio_value: Mapped[float] = mapped_column(Float)


class AgentDecision(Base):
    """
    SQLAlchemy Model representing an agent strategy decision log.
    """
    __tablename__ = 'agent_decisions'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )
    state_hash: Mapped[str] = mapped_column(String)
    action: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[float] = mapped_column(Float)
    reward: Mapped[float | None] = mapped_column(Float, nullable=True)


def init_db(db_path: str) -> sessionmaker:
    """
    Initialize the database engine and create tables.

    Args:
        db_path: Path to database (e.g. "./trading_bot.db" or "sqlite:///./trading_bot.db").

    Returns:
        A configured SQLAlchemy sessionmaker instance.
    """
    if db_path.startswith("sqlite:///"):
        connection_url = db_path
    else:
        connection_url = f"sqlite:///{db_path}"

    engine = create_engine(
        connection_url,
        connect_args={"check_same_thread": False},
        echo=False
    )
    
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(
        bind=engine, autocommit=False, autoflush=False
    )
    
    logger.info("Database initialized with connection_url: %s", connection_url)
    return SessionLocal


def log_trade(session: Session, trade_data: dict) -> None:
    """
    Safely log a trade into the database.

    Args:
        session: Active SQLAlchemy session.
        trade_data: Dictionary containing trade parameters.
    """
    trade = Trade(
        adapter=trade_data.get("adapter", ""),
        symbol=trade_data.get("symbol", ""),
        side=trade_data.get("side", ""),
        qty=float(trade_data.get("qty", 0.0)),
        price=float(trade_data.get("price", 0.0)),
        pnl=trade_data.get("pnl"),
        portfolio_value=float(trade_data.get("portfolio_value", 0.0))
    )
    try:
        session.add(trade)
        session.commit()
        logger.debug("Trade logged: %s %s", trade.symbol, trade.side)
    except Exception as e:
        logger.error("log_trade failed: %s", e, exc_info=True)
        session.rollback()


def log_decision(session: Session, decision_data: dict) -> None:
    """
    Safely log an agent decision into the database.

    Args:
        session: Active SQLAlchemy session.
        decision_data: Dictionary containing decision parameters.
    """
    decision = AgentDecision(
        state_hash=decision_data.get("state_hash", ""),
        action=int(decision_data.get("action", 0)),
        confidence=float(decision_data.get("confidence", 0.0)),
        reward=decision_data.get("reward")
    )
    try:
        session.add(decision)
        session.commit()
        logger.debug("Decision logged: action=%d", decision.action)
    except Exception as e:
        logger.error("log_decision failed: %s", e, exc_info=True)
        session.rollback()


def get_trade_history(session: Session, limit: int = 100) -> list[dict]:
    """
    Fetch the most recent trade history.

    Args:
        session: Active SQLAlchemy session.
        limit: Maximum number of rows to return.

    Returns:
        A list of parsed dictionary objects representing trades.
    """
    try:
        results = session.query(Trade).order_by(Trade.timestamp.desc()).limit(limit).all()
        return [
            {
                "id": trade.id,
                "timestamp": trade.timestamp.isoformat(),
                "adapter": trade.adapter,
                "symbol": trade.symbol,
                "side": trade.side,
                "qty": trade.qty,
                "price": trade.price,
                "pnl": trade.pnl,
                "portfolio_value": trade.portfolio_value,
            }
            for trade in results
        ]
    except Exception as e:
        logger.error("get_trade_history failed", exc_info=True)
        return []


def get_pnl_summary(session: Session) -> dict[str, float]:
    """
    Calculate and return a portfolio performance summary.

    Args:
        session: Active SQLAlchemy session.

    Returns:
        A dictionary containing total_pnl, win_rate, total_trades, avg_pnl, sharpe_ratio.
    """
    try:
        trades = session.query(Trade).all()
        total_trades = len(trades)
        
        if total_trades == 0:
            return {
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "total_trades": 0.0,
                "avg_pnl": 0.0,
                "sharpe_ratio": 0.0
            }

        winning_trades = sum(1 for t in trades if t.pnl is not None and t.pnl > 0)
        win_rate = winning_trades / total_trades
        
        total_pnl = sum(t.pnl for t in trades if t.pnl is not None)
        avg_pnl = total_pnl / total_trades

        pnl_values = [t.pnl for t in trades if t.pnl is not None]
        if len(pnl_values) > 1:
            mean_pnl = statistics.mean(pnl_values)
            std_pnl = statistics.stdev(pnl_values)
            sharpe = mean_pnl / (std_pnl + 1e-9)
        else:
            sharpe = 0.0

        return {
            "total_pnl": float(total_pnl),
            "win_rate": float(win_rate),
            "total_trades": float(total_trades),
            "avg_pnl": float(avg_pnl),
            "sharpe_ratio": float(sharpe)
        }
    except Exception as e:
        logger.error("get_pnl_summary failed", exc_info=True)
        return {
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "total_trades": 0.0,
            "avg_pnl": 0.0,
            "sharpe_ratio": 0.0
        }
