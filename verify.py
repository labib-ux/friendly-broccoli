"""
System Verification Script

Runs a comprehensive suite of startup checks validating Python runtime bounds,
module dependency structures, credentials payload existence, SQLite DB instantiation,
exchange API endpoints configurations, and complex state modeling tensor scaling loops.
"""

import sys
import traceback

def print_result(check_name: str, passed: bool, error: str = "") -> None:
    if passed:
        print(f"PASS \u2014 {check_name}")
    else:
        print(f"FAIL \u2014 {check_name}: {error}")

def run_checks() -> tuple[int, int]:
    passed_checks = 0
    total_checks = 0

    print("Running system verification...\n")

    # CHECK 1: Python version
    total_checks += 1
    if sys.version_info >= (3, 11):
        print_result("Python 3.11+ found", True)
        passed_checks += 1
    else:
        print_result(
            "Python 3.11+ required",
            False,
            f"Current version is {sys.version_info.major}.{sys.version_info.minor}"
        )

    # CHECK 2: All imports
    imports_to_test = [
        ("config", ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_BASE_URL", "DB_PATH", "MODEL_SAVE_PATH", "INITIAL_CAPITAL", "LOG_LEVEL"]),
        ("data.features", ["FEATURE_COLUMNS", "compute_features"]),
        ("data.fetcher", ["fetch_historical_data"]),
        ("environment.trading_env", ["TradingEnvironment"]),
        ("adapters.alpaca_adapter", ["AlpacaAdapter"]),
        ("adapters.base_adapter", ["BaseAdapter"]),
        ("risk.manager", ["RiskManager"]),
        ("database.logger", ["init_db", "log_trade", "log_decision"]),
        ("agent.trainer", ["TradingAgent"]),
        ("agent.policy", ["CustomFeatureExtractor"]),
        ("scheduler.jobs", ["run_trading_cycle"])
    ]

    all_imports_passed = True
    for module_name, symbols in imports_to_test:
        total_checks += 1
        try:
            module = __import__(module_name, fromlist=symbols)
            for symbol in symbols:
                getattr(module, symbol)
            print_result(f"imported {module_name}", True)
            passed_checks += 1
        except Exception as e:
            print_result(f"{module_name}", False, str(e))
            all_imports_passed = False

    # CHECK 3: API keys present
    total_checks += 1
    keys_present = False
    try:
        from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
        if ALPACA_API_KEY and ALPACA_SECRET_KEY:
            print_result("Alpaca API keys present", True)
            passed_checks += 1
            keys_present = True
        else:
            print_result(
                "Alpaca API keys present", 
                False, 
                "Alpaca API keys not set. Open .env and add your keys."
            )
    except Exception as e:
        print_result("Alpaca API keys present", False, str(e))

    # CHECK 4: Database initialization
    total_checks += 1
    db_initialized = False
    try:
        from database.logger import init_db
        from config import DB_PATH
        init_db(DB_PATH)
        print_result("Database initialized", True)
        passed_checks += 1
        db_initialized = True
    except Exception as e:
        print_result("Database initialized", False, str(e))

    df_scaled = None
    df_features = None

    # CHECK 5: Data pipeline (requires valid API keys)
    if keys_present:
        total_checks += 1
        pipeline_passed = False
        try:
            from adapters.alpaca_adapter import AlpacaAdapter
            from data.fetcher import fetch_historical_data
            
            adapter = AlpacaAdapter()
            df = fetch_historical_data(adapter, "BTC/USD", "1h", limit=100)
            
            required_cols = ["open", "high", "low", "close", "volume"]
            if not df.empty and all(col in df.columns for col in required_cols):
                print_result("Data fetcher returned valid OHLCV", True)
                passed_checks += 1
                pipeline_passed = True
            else:
                print_result(
                    "Data fetcher returned valid OHLCV", 
                    False, 
                    f"Empty: {df.empty}, Columns: {df.columns.tolist() if not df.empty else '[]'}"
                )
        except Exception as e:
            print_result("Data fetcher returned valid OHLCV", False, str(e))

        if pipeline_passed:
            total_checks += 1
            try:
                from data.features import compute_features, FEATURE_COLUMNS
                df_features = compute_features(df)
                
                if all(col in df_features.columns for col in FEATURE_COLUMNS):
                    if df_features.isnull().sum().sum() == 0:
                        print_result("Feature engineering succeeded without NaNs", True)
                        passed_checks += 1
                    else:
                        print_result(
                            "Feature engineering succeeded without NaNs",
                            False,
                            "NaN values detected in features."
                        )
                else:
                    missing = [c for c in FEATURE_COLUMNS if c not in df_features.columns]
                    print_result(
                        "Feature engineering succeeded without NaNs",
                        False,
                        f"Missing required features: {missing}"
                    )
            except Exception as e:
                print_result("Feature engineering succeeded without NaNs", False, str(e))

    # CHECK 6: Environment construction
    if df_features is not None:
        total_checks += 1
        try:
            import numpy as np
            from data.features import scale_features
            from environment.trading_env import TradingEnvironment
            
            df_scaled, scaler = scale_features(df_features)
            
            env = TradingEnvironment(
                df=df_scaled,
                initial_capital=10000.0,
                lookback_window=30,
                render_mode=None
            )
            obs, info = env.reset()
            
            if obs is not None and obs.dtype == np.float32:
                if obs.shape == env.observation_space.shape:
                    print_result("Environment instantiated and reset correctly", True)
                    passed_checks += 1
                else:
                    print_result(
                        "Environment instantiated and reset correctly",
                        False,
                        f"Shape mismatch: {obs.shape} != {env.observation_space.shape}"
                    )
            else:
                print_result(
                    "Environment instantiated and reset correctly",
                    False,
                    f"Invalid obs type/value: {type(obs)}"
                )
        except Exception as e:
            print_result("Environment instantiated and reset correctly", False, str(e))

    return passed_checks, total_checks

if __name__ == "__main__":
    passed, total = run_checks()
    
    print("\n==============================")
    print("VERIFICATION SUMMARY")
    print("==============================")
    print(f"{passed} of {total} checks passed")
    
    if passed == total:
        print("✅ System is ready. Run: python main.py --mode train")
    else:
        print("❌ Fix the failing checks above before running the system.")
