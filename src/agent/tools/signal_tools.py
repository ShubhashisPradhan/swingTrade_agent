# src/agent/tools/signal_tools.py
"""
Extended signal_tools.py — Phase 4++ (Full Indicator Coverage)

This file implements an extended signal layer that maps the
20+ indicators produced by indicator_tools.py into 10
signal detectors (archetypes). Each detector produces:
  - binary signal (0/1)
  - normalized strength [0..1]

generate_combined_signals() runs all detectors and returns:
  - list of (signal_name, aggregated_strength) sorted by strength desc

It also provides save_signals_to_state() to persist daily signals.

Expectations:
  - Input df contains columns from indicator_tools.py such as:
      Close, Open, High, Low, Volume,
      SMA_20, EMA_20, EMA_50 (if computed), TEMA_20, DEMA_20,
      RSI, STOCHk_14_3_3, STOCHd_14_3_3,
      MA, upper, lower, BB_signal, BB_Width (optional),
      DCU_20_20, DCL_20_20 (Donchian),
      ATR, KC_mid, KC_upper, KC_lower, KC_signal,
      MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9,
      PSAR, HA_close, HA_open, CCI, MOM, ROC, WILLIAMS_%R
  - Missing columns are handled safely (detector returns no signal).
"""

import os
import json
from datetime import datetime
from typing import List, Tuple, Dict

import yaml
import numpy as np
import pandas as pd


# ---------------------------
# Config loader & defaults
# ---------------------------
DEFAULT_CONFIG = {
    "pullback": {"ema_fast": 20, "ema_slow": 50, "rsi_low": 40, "rsi_high": 55, "near_pct": 0.02},
    "breakout": {"donchian_period": 20, "volume_ratio": 1.3},
    "mean_reversion": {"rsi_oversold": 30, "macdh_window": 3},
    "bollinger": {"period": 20, "n_std": 2},
    "macd": {"min_diff": 0.0},
    "rsi_divergence": {"lookback": 14, "price_change_min": 0.01},
    "psar": {"use_psar": True},
    "heikin_ashi": {"use_ha": True},
    "cci": {"upper": 100, "lower": -100},
    "volatility": {"atr_period": 14, "kc_multiplier": 2.0},
    "aggregation": {
        # weights for detectors when computing final ranking (sum should be 1.0 but doesn't have to be)
        "weights": {
            "PULLBACK": 0.15,
            "BREAKOUT": 0.20,
            "MEAN_REV": 0.10,
            "BOLLINGER_BREAKOUT": 0.10,
            "MACD_CROSS": 0.10,
            "RSI_DIVERGENCE": 0.07,
            "PSAR_FLIP": 0.06,
            "HA_TREND": 0.06,
            "CCI_EXTREME": 0.06,
            "VOL_EXPANSION": 0.10
        }
    }
}


def load_signal_config(path: str = "src/agent/config/signal_definitions.yaml") -> Dict:
    """Load YAML config, merge with defaults."""
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                user_cfg = yaml.safe_load(f) or {}
            # shallow merge for top-level keys
            for k, v in user_cfg.items():
                if isinstance(v, dict):
                    cfg.setdefault(k, {}).update(v)
                else:
                    cfg[k] = v
        except Exception:
            # fallback to defaults on parse error
            pass
    return cfg


# ---------------------------
# Utility helpers
# ---------------------------
def _safe_rank_pct(series: pd.Series) -> pd.Series:
    """Return pct rank [0..1] with NaN -> 0."""
    if series is None:
        return pd.Series(0, index=series.index if hasattr(series, "index") else None)
    return series.rank(pct=True).fillna(0)


def _normalize_series(series: pd.Series) -> pd.Series:
    """Scale numeric series to 0..1 by min-max; fallback to rank pct if constant."""
    if series is None:
        return series
    s = series.copy().astype(float)
    if s.isna().all():
        return s.fillna(0)
    minv, maxv = s.min(), s.max()
    if pd.isna(minv) or pd.isna(maxv) or minv == maxv:
        return _safe_rank_pct(s)
    return ((s - minv) / (maxv - minv)).clip(0, 1).fillna(0)


def _get_latest(df: pd.DataFrame) -> pd.Series:
    return df.iloc[-1] if len(df) > 0 else pd.Series()


# ---------------------------
# Individual detectors
# ---------------------------

def detect_ema_pullback(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """EMA Pullback detector: trend continuation on pullback to fast EMA."""
    ema_fast = cfg["ema_fast"]
    ema_slow = cfg["ema_slow"]
    near_pct = cfg.get("near_pct", 0.02)
    rsi_low = cfg["rsi_low"]
    rsi_high = cfg["rsi_high"]

    sig_col = "EMA_Pullback_Signal"
    strength_col = "EMA_Pullback_Strength"

    # default safe values
    df[sig_col] = 0
    df[strength_col] = 0.0

    required = {f"EMA_{ema_fast}", f"EMA_{ema_slow}", "RSI", "MOM", "Close"}
    if not required.issubset(set(df.columns)):
        return df

    # condition checks
    near_mask = df["Close"].between(df[f"EMA_{ema_fast}"] * (1 - near_pct), df[f"EMA_{ema_fast}"] * (1 + near_pct))
    uptrend_mask = df["Close"] > df[f"EMA_{ema_slow}"]
    rsi_mask = df["RSI"].between(rsi_low, rsi_high)
    mom_mask = df["MOM"] > 0

    cond = uptrend_mask & near_mask & rsi_mask & mom_mask
    df[sig_col] = cond.astype(int)

    # strength composite: closeness to EMA_fast, RSI closeness to 50, MOM percentile
    closeness = (1 - (abs(df["Close"] - df[f"EMA_{ema_fast}"]) / df[f"EMA_{ema_fast}"])).clip(lower=0)
    rsi_score = 1 - (abs(df["RSI"] - 50) / 50)
    mom_rank = _safe_rank_pct(df["MOM"])

    raw_strength = 0.4 * _normalize_series(closeness) + 0.3 * _normalize_series(rsi_score) + 0.3 * mom_rank
    df[strength_col] = (raw_strength * df[sig_col]).round(3)
    return df


def detect_breakout(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Breakout detector using Donchian upper band + volume + MACD momentum."""
    period = cfg.get("donchian_period", 20)
    vr = cfg.get("volume_ratio", 1.3)

    sig_col = "Breakout_Signal"
    strength_col = "Breakout_Strength"
    df[sig_col] = 0
    df[strength_col] = 0.0

    dcu_col = f"DCU_{period}_{period}"
    macd_col = "MACD_12_26_9"
    macds_col = "MACDs_12_26_9"

    if not set([dcu_col, "Volume", macd_col, macds_col]).issubset(set(df.columns)):
        return df

    avg_vol = df["Volume"].rolling(window=period, min_periods=1).mean().replace(0, np.nan)
    cond = (df["Close"] > df[dcu_col]) & (df["Volume"] > avg_vol * vr) & (df[macd_col] > df[macds_col])
    df[sig_col] = cond.astype(int)

    vol_ratio = (df["Volume"] / avg_vol).fillna(1).clip(0, 5)
    macd_diff = (df[macd_col] - df[macds_col]).fillna(0)
    price_ratio = (df["Close"] / df[dcu_col]).fillna(1)

    raw_strength = 0.5 * _normalize_series(vol_ratio) + 0.3 * _normalize_series(macd_diff) + 0.2 * _normalize_series(price_ratio)
    df[strength_col] = (raw_strength * df[sig_col]).round(3)
    return df


def detect_mean_reversion(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Mean reversion — oversold bounce using RSI and MACD histogram."""
    rsi_oversold = cfg.get("rsi_oversold", 30)
    macdh_col = "MACDh_12_26_9"

    sig_col = "MeanRev_Signal"
    strength_col = "MeanRev_Strength"
    df[sig_col] = 0
    df[strength_col] = 0.0

    if not set(["RSI", "Close", macdh_col]).issubset(set(df.columns)):
        return df

    # condition: RSI below threshold and price has started to recover + MACDh rising
    cond = (df["RSI"] < rsi_oversold) & (df["Close"] > df["Close"].shift(1)) & (df[macdh_col].diff() > 0)
    df[sig_col] = cond.astype(int)

    oversold = ((rsi_oversold - df["RSI"]) / rsi_oversold).clip(lower=0)
    macdh_gain = df[macdh_col].diff().clip(lower=0).fillna(0)
    recovery = (df["Close"] / df["Close"].shift(1) - 1).clip(lower=0).fillna(0)

    raw_strength = 0.5 * _normalize_series(oversold) + 0.3 * _normalize_series(macdh_gain) + 0.2 * _normalize_series(recovery)
    df[strength_col] = (raw_strength * df[sig_col]).round(3)
    return df


def detect_bollinger_breakout(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Bollinger breakout detector: close beyond upper or lower band."""
    sig_col = "BOLLINGER_BREAKOUT_Signal"
    strength_col = "BOLLINGER_BREAKOUT_Strength"
    df[sig_col] = 0
    df[strength_col] = 0.0

    required = {"MA", "upper", "lower", "Close"}
    if not required.issubset(set(df.columns)):
        return df

    up = df["Close"] > df["upper"]
    down = df["Close"] < df["lower"]
    df[sig_col] = (up | down).astype(int)

    # strength: distance from band normalized
    dist = np.where(up, (df["Close"] - df["upper"]) / df["MA"], np.where(down, (df["lower"] - df["Close"]) / df["MA"], 0))
    df[strength_col] = (_normalize_series(pd.Series(dist, index=df.index)) * df[sig_col]).round(3)
    return df


def detect_macd_crossover(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """MACD crossover detector (signal line crossing) — detects recent cross events."""
    sig_col = "MACD_CROSS_Signal"
    strength_col = "MACD_CROSS_Strength"
    df[sig_col] = 0
    df[strength_col] = 0.0

    macd_col = "MACD_12_26_9"
    macds_col = "MACDs_12_26_9"
    macdh_col = "MACDh_12_26_9"

    if not set([macd_col, macds_col, macdh_col]).issubset(set(df.columns)):
        return df

    # crossover: macd crosses above signal -> bullish; vice versa -> bearish
    macd = df[macd_col]
    sig = df[macds_col]
    cross_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))
    cross_down = (macd < sig) & (macd.shift(1) >= sig.shift(1))
    df[sig_col] = (cross_up | cross_down).astype(int)

    # strength: absolute histogram magnitude normalized
    hist = df[macdh_col].abs().fillna(0)
    df[strength_col] = (_normalize_series(hist) * df[sig_col]).round(3)
    return df


def detect_rsi_divergence(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """
    Simple RSI divergence heuristic:
    - Lookback peaks/troughs in price and RSI over window;
    - If price makes higher high while RSI makes lower high => bearish divergence (pre-reversal)
    - If price makes lower low while RSI makes higher low => bullish divergence
    Note: This is a heuristic and not exhaustive divergence detection.
    """
    lookback = cfg.get("lookback", 14)
    sig_col = "RSI_DIVERGENCE_Signal"
    strength_col = "RSI_DIVERGENCE_Strength"
    df[sig_col] = 0
    df[strength_col] = 0.0

    if not set(["Close", "RSI"]).issubset(set(df.columns)):
        return df

    # compute rolling highs/lows
    price_h = df["Close"].rolling(window=lookback, min_periods=2).max()
    price_l = df["Close"].rolling(window=lookback, min_periods=2).min()
    rsi_h = df["RSI"].rolling(window=lookback, min_periods=2).max()
    rsi_l = df["RSI"].rolling(window=lookback, min_periods=2).min()

    # heuristic signals at latest index: compare latest two pivots
    # find last two local highs/lows by simple index approach (approx)
    # Use simple check across window: if in last lookback price increased but rsi decreased => divergence
    recent = df.iloc[-lookback:]
    if len(recent) < 3:
        return df

    # compute simple slopes
    price_slope = recent["Close"].iloc[-1] - recent["Close"].iloc[0]
    rsi_slope = recent["RSI"].iloc[-1] - recent["RSI"].iloc[0]

    bullish_div = (price_slope < 0) and (rsi_slope > 0)
    bearish_div = (price_slope > 0) and (rsi_slope < 0)

    if bullish_div or bearish_div:
        df.at[df.index[-1], sig_col] = 1
        # strength based on opposite slopes magnitude
        raw = abs(price_slope) * abs(rsi_slope)
        df.at[df.index[-1], strength_col] = float(min(1.0, raw / (abs(recent["Close"]).mean() * 0.1 + 1e-9)))
    return df


def detect_psar_flip(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Parabolic SAR flip: when PSAR location flips relative to price."""
    sig_col = "PSAR_FLIP_Signal"
    strength_col = "PSAR_FLIP_Strength"
    df[sig_col] = 0
    df[strength_col] = 0.0

    if "PSAR" not in df.columns or "Close" not in df.columns:
        return df

    psar = df["PSAR"]
    close = df["Close"]
    # detect flip: psar moves from above price to below price or vice versa
    flip_up = (psar.shift(1) > close.shift(1)) & (psar < close)
    flip_down = (psar.shift(1) < close.shift(1)) & (psar > close)
    flip = flip_up | flip_down
    df[sig_col] = flip.astype(int)

    # strength: distance between PSAR and price normalized
    dist = (abs(close - psar) / close).fillna(0)
    df[strength_col] = (_normalize_series(dist) * df[sig_col]).round(3)
    return df


def detect_heikin_ashi_trend(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Heikin-Ashi trend: HA_close > HA_open indicates bullish smoothing trend."""
    sig_col = "HA_TREND_Signal"
    strength_col = "HA_TREND_Strength"
    df[sig_col] = 0
    df[strength_col] = 0.0

    if not set(["HA_close", "HA_open"]).issubset(set(df.columns)):
        return df

    bullish = df["HA_close"] > df["HA_open"]
    df[sig_col] = bullish.astype(int)
    # strength: size of HA candle normalized by HA range
    ha_size = (abs(df["HA_close"] - df["HA_open"]) / (df["HA_high"] - df["HA_low"] + 1e-9)).fillna(0)
    df[strength_col] = (_normalize_series(ha_size) * df[sig_col]).round(3)
    return df


def detect_cci_extreme(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """CCI extremes: detect > +100 or < -100 signals."""
    sig_col = "CCI_EXTREME_Signal"
    strength_col = "CCI_EXTREME_Strength"
    df[sig_col] = 0
    df[strength_col] = 0.0

    upper = cfg.get("upper", 100)
    lower = cfg.get("lower", -100)
    if "CCI" not in df.columns:
        return df

    over = df["CCI"] > upper
    under = df["CCI"] < lower
    df[sig_col] = (over | under).astype(int)

    # strength: abs distance normalized
    dist = (df["CCI"].abs() - min(abs(lower), upper)) / (df["CCI"].abs().max() + 1e-9)
    df[strength_col] = (_normalize_series(dist.fillna(0)) * df[sig_col]).round(3)
    return df


def detect_volatility_expansion(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """
    Volatility expansion: ATR increase + KC breakout or BB width spike.
    Useful to flag that a squeeze has released.
    """
    sig_col = "VOL_EXPANSION_Signal"
    strength_col = "VOL_EXPANSION_Strength"
    df[sig_col] = 0
    df[strength_col] = 0.0

    if "ATR" not in df.columns and not set(["KC_upper", "KC_lower"]).issubset(set(df.columns)):
        return df

    # ATR surge relative to rolling mean
    atr = df["ATR"] if "ATR" in df.columns else pd.Series(0, index=df.index)
    atr_avg = atr.rolling(window=20, min_periods=1).mean().replace(0, np.nan)
    atr_surge = (atr / atr_avg).fillna(1).clip(0, 10)

    # KC breakout: close outside KC bands
    kc_break = False
    if set(["KC_upper", "KC_lower", "Close"]).issubset(set(df.columns)):
        kc_break = (df["Close"] > df["KC_upper"]) | (df["Close"] < df["KC_lower"])
    else:
        kc_break = pd.Series(False, index=df.index)

    # BB width if available
    bbw = df["BB_Width"] if "BB_Width" in df.columns else ( (df["upper"] - df["lower"]) / (df["MA"] + 1e-9) if set(["upper", "lower", "MA"]).issubset(df.columns) else pd.Series(0, index=df.index) )

    # combine
    cond = (atr_surge > 1.25) | kc_break | (bbw > bbw.rolling(window=20, min_periods=1).quantile(0.8))
    df[sig_col] = cond.astype(int)

    # strength combination
    raw_strength = 0.5 * _normalize_series(atr_surge) + 0.3 * _normalize_series(bbw) + 0.2 * _normalize_series(kc_break.astype(int))
    df[strength_col] = (_normalize_series(raw_strength) * df[sig_col]).round(3)
    return df


# ---------------------------
# Aggregation & persistence
# ---------------------------

DETECTOR_FUNCS = {
    "PULLBACK": detect_ema_pullback,
    "BREAKOUT": detect_breakout,
    "MEAN_REV": detect_mean_reversion,
    "BOLLINGER_BREAKOUT": detect_bollinger_breakout,
    "MACD_CROSS": detect_macd_crossover,
    "RSI_DIVERGENCE": detect_rsi_divergence,
    "PSAR_FLIP": detect_psar_flip,
    "HA_TREND": detect_heikin_ashi_trend,
    "CCI_EXTREME": detect_cci_extreme,
    "VOL_EXPANSION": detect_volatility_expansion,
}


def generate_combined_signals(df: pd.DataFrame, symbol: str = None, config: Dict = None) -> List[Tuple[str, float]]:
    """
    Run all detectors, aggregate weighted strengths, and return an ordered list of (signal_name, score).
    Score is in [0..1] (aggregation of detector strength * configured weight).
    """
    if config is None:
        config = load_signal_config()
    agg_weights = config.get("aggregation", {}).get("weights", DEFAULT_CONFIG["aggregation"]["weights"])

    df_local = df.copy()

    # run each detector to populate *_Signal and *_Strength columns
    for name, func in DETECTOR_FUNCS.items():
        try:
            func_cfg = config.get(name.lower(), {}) if isinstance(config.get(name.lower(), {}), dict) else {}
            # pass the whole detector-specific config fallback using merged keys
            func(df_local, func_cfg or config.get(name.lower(), {}) or config)
        except TypeError:
            # some detectors expect (df, cfg) others only (df, cfg)
            try:
                DETECTOR_FUNCS[name](df_local, config.get(name.lower(), {}) or config)
            except Exception:
                # fail-safe: ignore detector on error
                continue
        except Exception:
            continue

    latest = _get_latest(df_local)
    if latest.empty:
        return [("NEUTRAL", 0.0)]

    # collect detector outputs and compute weighted aggregate for each signal NAME
    aggregate_scores = {}
    per_detector_info = {}

    for det_name in DETECTOR_FUNCS.keys():
        sig_col_candidates = [c for c in df_local.columns if c.startswith(det_name.replace("BOLLINGER_BREAKOUT", "BOLLINGER_BREAKOUT").split()[0]) or det_name in c]
        # find expected columns
        # detectors wrote columns like "<DETECTOR>_Signal" and "<DETECTOR>_Strength" (or named differently)
        # we'll attempt to find exact columns if present
        # mapping:
        # PULLBACK -> EMA_Pullback_Signal / Strength
        if det_name == "PULLBACK":
            sig_col = "EMA_Pullback_Signal"
            str_col = "EMA_Pullback_Strength"
        elif det_name == "BREAKOUT":
            sig_col = "Breakout_Signal"
            str_col = "Breakout_Strength"
        elif det_name == "MEAN_REV":
            sig_col = "MeanRev_Signal"
            str_col = "MeanRev_Strength"
        elif det_name == "BOLLINGER_BREAKOUT":
            sig_col = "BOLLINGER_BREAKOUT_Signal"
            str_col = "BOLLINGER_BREAKOUT_Strength"
        elif det_name == "MACD_CROSS":
            sig_col = "MACD_CROSS_Signal"
            str_col = "MACD_CROSS_Strength"
        elif det_name == "RSI_DIVERGENCE":
            sig_col = "RSI_DIVERGENCE_Signal"
            str_col = "RSI_DIVERGENCE_Strength"
        elif det_name == "PSAR_FLIP":
            sig_col = "PSAR_FLIP_Signal"
            str_col = "PSAR_FLIP_Strength"
        elif det_name == "HA_TREND":
            sig_col = "HA_TREND_Signal"
            str_col = "HA_TREND_Strength"
        elif det_name == "CCI_EXTREME":
            sig_col = "CCI_EXTREME_Signal"
            str_col = "CCI_EXTREME_Strength"
        elif det_name == "VOL_EXPANSION":
            sig_col = "VOL_EXPANSION_Signal"
            str_col = "VOL_EXPANSION_Strength"
        else:
            sig_col = None
            str_col = None

        # read values safely
        sig_val = float(latest.get(sig_col, 0)) if sig_col else 0.0
        strength_val = float(latest.get(str_col, 0.0)) if str_col else 0.0

        weight = float(agg_weights.get(det_name, 0.0))
        aggregate_scores[det_name] = weight * strength_val * sig_val
        per_detector_info[det_name] = {"sig": sig_val, "strength": strength_val, "weight": weight}

    # final scoring: sort by aggregated weight
    scored = [(name, float(score)) for name, score in aggregate_scores.items() if score > 0]
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)

    # if none found, return neutral
    if not scored_sorted:
        return [("NEUTRAL", 0.0)]

    # also return top N (all but sorted) — but function returns list
    return scored_sorted


def save_signals_to_state(symbol: str, signals: List[Tuple[str, float]], path: str = "src/agent/states/signals.json") -> str:
    """
    Append today's signals for 'symbol' into a simple json state file.
    Structure:
      { 'YYYY-MM-DD': { 'SYMBOL': [ {'signal':'PULLBACK','score':0.82}, ... ] } }
    """
    date_key = datetime.utcnow().strftime("%Y-%m-%d")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                state = json.load(f)
        except Exception:
            state = {}
    else:
        state = {}

    state.setdefault(date_key, {})
    state[date_key][symbol] = [{"signal": s[0], "score": float(s[1])} for s in signals]

    with open(path, "w") as f:
        json.dump(state, f, indent=2)

    return path


# ---------------------------
# Convenience runner for single symbol
# ---------------------------
def run_signal_detection_for_df(df: pd.DataFrame, symbol: str = None, config_path: str = None) -> List[Tuple[str, float]]:
    """
    Convenience function: load config, run detectors & aggregation, return signal list.
    """
    cfg = load_signal_config(config_path) if config_path else load_signal_config()
    signals = generate_combined_signals(df, symbol=symbol, config=cfg)
    if symbol:
        try:
            save_signals_to_state(symbol, signals)
        except Exception:
            pass
    return signals




