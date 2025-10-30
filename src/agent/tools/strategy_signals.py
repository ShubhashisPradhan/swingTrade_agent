import yaml
import numpy as np
from datetime import datetime

def load_signal_config(config_path="src/agent/config/signal_config.yaml"):
    """Load signal weight & direction config."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg.get("signals", {})


def aggregate_signals(signal_list, df=None, config_path="src/agent/config/signal_config.yaml"):
    """
    Enhanced signal aggregator:
    Combines indicator-level signals into a final actionable decision.

    Args:
        signal_list : list[tuple(str, float)]
            Example: [("VOL_EXPANSION", 0.0988), ("HA_TREND", 0.0469)]
        df : pd.DataFrame, optional
            Used for volatility-based confidence scaling.
        config_path : str
            Path to signal YAML config.

    Returns:
        dict : final_action, confidence, total_score, reasoning details
    """
    signal_cfg = load_signal_config(config_path)
    total_score = 0.0
    active_signals = []
    bullish_score, bearish_score = 0.0, 0.0

    # --- Dynamic volatility adjustment ---
    vol_adj_factor = 1.0
    if df is not None and "Volatility_Score" in df.columns:
        vol_adj_factor = float(df["Volatility_Score"].iloc[-1])
        vol_adj_factor = np.clip(0.8 + (vol_adj_factor * 0.4), 0.8, 1.2)  # normalize to 0.8–1.2

    # --- Aggregate weighted signal scores ---
    for signal, score in signal_list:
        if signal not in signal_cfg:
            continue

        weight = signal_cfg[signal].get("weight", 0.2)
        direction = signal_cfg[signal].get("direction", "neutral")
        polarity = 1 if direction == "bullish" else -1 if direction == "bearish" else 0

        weighted_score = polarity * weight * score
        total_score += weighted_score

        if polarity == 1:
            bullish_score += weighted_score
        elif polarity == -1:
            bearish_score += abs(weighted_score)

        active_signals.append({
            "name": signal,
            "score": round(score, 4),
            "weight": weight,
            "direction": direction,
            "weighted_contribution": round(weighted_score, 4)
        })

    # --- Normalize & adjust confidence dynamically ---
    normalized = np.tanh(total_score)  # ensures bounded [-1, 1]
    confidence = abs(normalized) * vol_adj_factor
    confidence = float(np.clip(confidence, 0, 1))

    # --- Dynamic thresholds (tunable) ---
    BUY_THRESHOLD = 0.1
    SELL_THRESHOLD = -0.1
    HOLD_THRESHOLD = 0.05
    MIN_CONFIDENCE = 0.03

    # --- Decision logic ---
    if confidence < MIN_CONFIDENCE:
        action = "NEUTRAL"
    elif normalized > BUY_THRESHOLD:
        action = "BUY"
    elif normalized < SELL_THRESHOLD:
        action = "SELL"
    elif abs(normalized) <= HOLD_THRESHOLD:
        action = "WATCH"
    else:
        action = "HOLD"

    # --- Construct reasoning dictionary ---
    reasoning = {
        "bullish_score": round(bullish_score, 4),
        "bearish_score": round(bearish_score, 4),
        "volatility_adjustment": round(vol_adj_factor, 3),
        "net_direction": (
            "bullish" if total_score > 0
            else "bearish" if total_score < 0
            else "neutral"
        ),
        "explanation": f"{action} chosen: bullish={bullish_score:.3f}, bearish={bearish_score:.3f}, "
                       f"vol_adj={vol_adj_factor:.2f}, conf={confidence:.2f}"
    }

    return {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "final_action": action,
        "confidence": round(confidence, 4),
        "total_score": round(total_score, 4),
        "signals_used": active_signals,
        "reasoning": reasoning
    }



