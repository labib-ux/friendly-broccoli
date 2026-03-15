import time
import logging
import sys
import argparse
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED

from config import (
    DB_PATH,
    MODEL_SAVE_PATH,
    INITIAL_CAPITAL,
    LOG_LEVEL,
    LOOKBACK_WINDOW
)
from database.logger import init_db
from adapters.alpaca_adapter import AlpacaAdapter
from risk.manager import RiskManager
from agent.trainer import TradingAgent
from scheduler.jobs import run_trading_cycle
from data.fetcher import fetch_historical_data
from data.features import compute_features, scale_features
from environment.trading_env import TradingEnvironment

# Setup global application logger unconditionally first
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Algorithmic Trading Engine")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["train", "paper", "live"],
        default="paper",
        help="Execution mode: train, paper, or live"
    )
    args = parser.parse_args()

    # STEP 1 — Initialize database
    SessionLocal = init_db(DB_PATH)
    logger.info("Database initialized successfully.")

    adapter = AlpacaAdapter()
    risk_manager = RiskManager(initial_capital=INITIAL_CAPITAL)

    if args.mode == "train":
        # STEP 1 — Fetch historical training data
        logger.info("Fetching historical data for training...")
        df = fetch_historical_data(adapter, "BTC/USD", "1h", limit=500)
        
        if df.empty:
            logger.critical(
                "Could not fetch training data. "
                "Check your Alpaca API keys and connection."
            )
            sys.exit(1)

        # STEP 2 — Compute and scale features
        df = compute_features(df)
        
        if df.empty:
            logger.critical("Feature engineering returned empty DataFrame.")
            sys.exit(1)
        
        df_scaled, scaler = scale_features(df)
        logger.info(
            "Training data ready: %d rows, %d features",
            len(df_scaled), len(df_scaled.columns)
        )

        # STEP 3 — Build the environment
        env = TradingEnvironment(
            df=df_scaled,
            initial_capital=INITIAL_CAPITAL,
            lookback_window=LOOKBACK_WINDOW,
            render_mode=None
        )
        logger.info("Training environment created.")

        # STEP 4 — Initialize agent WITH the environment
        agent = TradingAgent(env=env)

        # STEP 5 — Train
        logger.info("Starting training...")
        agent.train(total_timesteps=100000, reset_num_timesteps=True)
        logger.info("Training complete. Model saved to: %s", MODEL_SAVE_PATH)
        
        # Exit immediately after training completes
        sys.exit(0)

    else:
        # For paper/live modes, agent loads the inference models from disk 
        agent = TradingAgent(env=None)

    logger.info("All trading components initialized.")

    # STEP 3 — Daily state tracking for circuit breaker
    daily_state = {"start_value": INITIAL_CAPITAL}

    # STEP 4 — Build the scheduler with listeners
    scheduler = BackgroundScheduler()

    def on_job_error(event):
        logger.critical(
            "Trading cycle job FAILED with exception: %s",
            event.exception, exc_info=True
        )

    def on_job_success(event):
        logger.info(
            "Trading cycle job completed successfully at %s",
            event.scheduled_run_time
        )

    scheduler.add_listener(on_job_error, EVENT_JOB_ERROR)
    scheduler.add_listener(on_job_success, EVENT_JOB_EXECUTED)

    # STEP 5 — Register the trading job
    scheduler.add_job(
        run_trading_cycle,
        trigger='cron',
        minute='*/15',
        args=[
            adapter,
            agent,
            risk_manager,
            SessionLocal,
            "BTC/USD",
            "15m",
            30,
            INITIAL_CAPITAL,
            daily_state["start_value"]
        ],
        id="live_trading_cycle",
        name="Live Trading Cycle — BTC/USD 15m",
        misfire_grace_time=60
    )

    # STEP 6 — Start scheduler and keep main thread alive
    scheduler.start()
    
    logger.info("="*60)
    logger.info("Live trading engine ONLINE")
    logger.info("Mode: %s", getattr(args, 'mode', 'paper').upper())
    logger.info("Symbol: BTC/USD | Interval: every 15 minutes")
    logger.info("Initial capital: $%.2f", INITIAL_CAPITAL)
    logger.info("Press Ctrl+C to stop.")
    logger.info("="*60)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutdown signal (Ctrl+C) received.")
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped. Live trading engine OFFLINE.")
        sys.exit(0)
