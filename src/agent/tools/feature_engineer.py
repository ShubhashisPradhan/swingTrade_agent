"""
feature_engineer.py
--------------------
Transforms raw indicator outputs (from indicator_tools.py)
into composite, model-friendly swing trading features.

Author: Shubhashis Pradhan
Phase: 3.2 (Feature Engineering)
"""

import pandas as pd
import numpy as np


# ============================================================
# 1️⃣ VOLATILITY FEATURES
# ============================================================

def compute_volatility_features(df, period=20):
    """
    Enhanced volatility measure combining:
    - ATR (volatility amplitude)
    - Rolling std (price variability)
    - Bollinger Band Width (squeeze/expansion behavior)
    """
    if "ATR" not in df.columns:
        raise ValueError("ATR must be computed before volatility features.")
    
    if "std" not in df.columns:
        df["std"] = df["Close"].rolling(window=period).std()

    # Add BB_Width if available (upper - lower) / MA
    if all(x in df.columns for x in ["upper", "lower", "MA"]):
        df["BB_Width"] = (df["upper"] - df["lower"]) / df["MA"]
    else:
        df["BB_Width"] = df["std"] / df["Close"]

    # Rank normalization
    atr_rank = df["ATR"].rank(pct=True)
    std_rank = df["std"].rank(pct=True)
    bb_rank = df["BB_Width"].rank(pct=True)

    # Weighted composite volatility score
    df["Volatility_Score"] = (
        0.5 * atr_rank + 0.3 * std_rank + 0.2 * bb_rank
    ).fillna(0)

    # Volatility condition (signal)
    df["Volatility_Signal"] = np.where(df["Volatility_Score"] > 0.7, 1, 0)
    return df



# ============================================================
# 2️⃣ VOLUME-BASED FEATURES
# ============================================================

def compute_volume_features(df, period=20):
    """Volume-based metrics.
    - Vol_MA: 20-day moving average of volume.
    - Vol_Surge: Volume / Vol_MA → shows sudden spikes.
    """
    df["Vol_MA"] = df["Volume"].rolling(window=period).mean()
    df["Vol_Surge"] = (df["Volume"] / df["Vol_MA"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df["Vol_Surge_Signal"] = np.where(df["Vol_Surge"] > 1.5, 1, 0)  # Volume surge signal
    return df


# ============================================================
# 3️⃣ TREND STRENGTH FEATURES
# ============================================================

def compute_trend_strength(df):
    """Combine trend indicators into a composite trend score.
    Includes EMA, MACD, and Heikin-Ashi directional signals.
    Range: roughly -3 to +3.
    """
    trend_signals = [
        "EMA_20_signal", "MACD_signal", "HA_signal",
        "KC_signal", "RSI_signal", "CCI_signal"
    ]
    available = [col for col in trend_signals if col in df.columns]
    df["Trend_Strength"] = df[available].sum(axis=1)
    return df


# ============================================================
# 4️⃣ COMPOSITE SWING SCORE
# ============================================================

def compute_swing_score(df):
    """Compute an overall swing score from all *_signal columns.
    +ve → bullish bias, -ve → bearish bias.
    """
    signal_cols = [col for col in df.columns if col.endswith("_signal")]
    df["Swing_Score"] = df[signal_cols].sum(axis=1)
    df["Swing_Sentiment"] = np.select(
        [df["Swing_Score"] > 2, df["Swing_Score"] < -2],
        ["Strong Bullish", "Strong Bearish"],
        default="Neutral"
    )
    return df


# ============================================================
# 5️⃣ ORCHESTRATOR
# ============================================================

def build_feature_set(df):
    """
    Full feature-engineering pipeline:
    1. Compute volatility, volume, and trend metrics.
    2. Combine into a final swing score.
    """
    df = compute_volatility_features(df)
    df = compute_volume_features(df)
    df = compute_trend_strength(df)
    df = compute_swing_score(df)

    # Optional normalization
    df["Normalized_Swing_Score"] = (
        (df["Swing_Score"] - df["Swing_Score"].mean()) /
        df["Swing_Score"].std()
    ).fillna(0)

    # Rank-based signal for easy interpretation
    df["Swing_Rank"] = df["Swing_Score"].rank(pct=True)
    return df
