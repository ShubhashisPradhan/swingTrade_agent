"""
Handles reading/writing of cached market data (.parquet).
"""

import os
import pandas as pd
from datetime import datetime, timedelta

def get_cache_path(symbol, cache_dir):
    """Return full cache path for a symbol."""
    return os.path.join(cache_dir, f"{symbol}.parquet")

def is_cache_valid(symbol, cache_dir, freshness_days=1):
    """Check if cache exists and is recent enough."""
    path = get_cache_path(symbol, cache_dir)
    if not os.path.exists(path):
        return False
    file_time = datetime.fromtimestamp(os.path.getmtime(path))
    return (datetime.now() - file_time) < timedelta(days=freshness_days)

def load_cache(symbol, cache_dir):
    """Load cached data from Parquet."""
    return pd.read_parquet(get_cache_path(symbol, cache_dir))

def save_cache(symbol, df, cache_dir):
    """Save data to Parquet cache."""
    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(get_cache_path(symbol, cache_dir), compression="snappy")
