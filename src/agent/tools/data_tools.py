import pandas as pd
import yfinance as yf
from src.utils.cache_manager import load_cache, save_cache, is_cache_valid
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger()

def fetch_from_yfinance(symbols, period="6mo", interval="1d", cache=True):
    """
    Fetch OHLCV (Open, High, Low, Close, Volume) data for given symbols from Yahoo Finance.
    Automatically flattens MultiIndex columns (if present) and keeps only essential fields.
    Handles caching, logging, and data freshness checks.
    """

    config = load_config()
    cache_dir = config["cache"]["path"]
    freshness = config["cache"]["freshness_days"]

    all_data = {}

    for symbol in symbols:
        try:
            # ---------------------- CACHE HANDLING ----------------------
            if cache and is_cache_valid(symbol, cache_dir, freshness):
                df = load_cache(symbol, cache_dir)
                logger.info(f"Loaded {symbol} from cache ✅")
            else:
                logger.info(f"Fetching {symbol} from Yahoo Finance ...")
                df = yf.download(symbol, period=period, interval=interval)

                # ---------------------- FLATTEN MULTIINDEX ----------------------
                if isinstance(df.columns, pd.MultiIndex):
                    # Extract data for the symbol level (e.g., 'RELIANCE.NS')
                    try:
                        df = df.xs(symbol, axis=1, level=1)
                        logger.info(f"Flattened MultiIndex columns for {symbol}.")
                    except Exception as e:
                        logger.warning(f"Could not flatten MultiIndex for {symbol}: {e}")

                # ---------------------- KEEP ONLY REQUIRED COLUMNS ----------------------
                expected_cols = ["Open", "High", "Low", "Close", "Volume"]
                available_cols = [col for col in expected_cols if col in df.columns]
                df = df[available_cols]

                # Reset and sort by Date
                df.reset_index(inplace=True)
                df = df.sort_values(by="Date")

                # ---------------------- SAVE TO CACHE ----------------------
                if cache:
                    save_cache(symbol, df, cache_dir)
                    logger.info(f"Fetched and cached data for {symbol} ({len(df)} rows).")

            all_data[symbol] = df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")

    return all_data
