"""
Fetch OHLCV market data using yFinance and integrate with cache + logging.
"""

import yfinance as yf
import pandas as pd
from src.utils.cache_manager import (load_cache, save_cache, is_cache_valid)
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger()

def fetch_from_yfinance(symbols, period="6mo", interval="1d", cache=True):
    """Fetch OHLCV data for given symbols."""
    config = load_config()
    cache_dir = config["cache"]["path"]
    freshness = config["cache"]["freshness_days"]

    all_data = {}

    for symbol in symbols:
        try:
            if cache and is_cache_valid(symbol, cache_dir, freshness):
                df = load_cache(symbol, cache_dir)
                logger.info(f"Loaded {symbol} from cache ✅")
            else:
                df = yf.download(symbol, period=period, interval=interval)
                df.reset_index(inplace=True)
                if cache:
                    save_cache(symbol, df, cache_dir)
                    logger.info(f"Fetched {symbol} and saved to cache.")
            all_data[symbol] = df
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
    return all_data
