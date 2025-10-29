"""
indicator_tools.py
--------------------
Compute all 20 technical indicators for Swing Trading feature engineering.

✔ Includes both continuous and discretized (signal) features.
✔ Each indicator is clearly documented.
✔ Outputs standardized trading signals: +1 (Bullish), 0 (Neutral), -1 (Bearish).
"""

import pandas as pd
import numpy as np
import pandas_ta as ta


# ============================================================
# 1️⃣ TREND / MOVING AVERAGE INDICATORS
# ============================================================

def compute_sma(df, period=20):
    """Simple Moving Average (SMA)
    - Formula: SMA = mean(Close[N])
    - Close > SMA → Bullish (+1)
    - Close < SMA → Bearish (–1)
    """
    df[f"SMA_{period}"] = ta.sma(df["Close"], length=period)
    df[f"SMA_{period}_signal"] = np.where(df["Close"] > df[f"SMA_{period}"], 1, -1)
    return df


def compute_ema(df, period=20):
    """Exponential Moving Average (EMA)
    - Formula: EMA_t = (Close_t * α) + EMA_(t−1)*(1−α)
    - Close > EMA → Bullish (+1)
    """
    df[f"EMA_{period}"] = ta.ema(df["Close"], length=period)
    df[f"EMA_{period}_signal"] = np.where(df["Close"] > df[f"EMA_{period}"], 1, -1)
    return df


def compute_tema(df, period=20):
    """Triple Exponential Moving Average (TEMA)
    - Reduces lag and helps identify sustained trends.
    """
    df[f"TEMA_{period}"] = ta.tema(df["Close"], length=period)
    return df


def compute_dema(df, period=20):
    """Double Exponential Moving Average (DEMA)
    - Reacts faster to price changes than EMA.
    """
    df[f"DEMA_{period}"] = ta.dema(df["Close"], length=period)
    return df


def compute_triple_ma_crossover(df):
    """Triple Moving Average Crossover
    - Bullish: short > mid > long
    - Bearish: short < mid < long
    """
    df["TMA_short"] = ta.sma(df["Close"], length=10)
    df["TMA_mid"] = ta.sma(df["Close"], length=50)
    df["TMA_long"] = ta.sma(df["Close"], length=100)
    df["TMA_signal"] = np.where(
        (df["TMA_short"] > df["TMA_mid"]) & (df["TMA_mid"] > df["TMA_long"]), 1,
        np.where((df["TMA_short"] < df["TMA_mid"]) & (df["TMA_mid"] < df["TMA_long"]), -1, 0)
    )
    return df


def compute_ma_envelope(df, period=20, percent=2.5):
    """Moving Average Envelope
    - Bands = SMA ± (percent%)
    - Price > Upper → Overbought (–1)
    - Price < Lower → Oversold (+1)
    """
    sma = df["Close"].rolling(window=period).mean()
    df[f"MAENV_{period}_MID"] = sma
    df[f"MAENV_{period}_UPPER"] = sma * (1 + percent / 100)
    df[f"MAENV_{period}_LOWER"] = sma * (1 - percent / 100)
    df[f"MAENV_{period}_signal"] = np.where(df["Close"] > df[f"MAENV_{period}_UPPER"], -1,
                                    np.where(df["Close"] < df[f"MAENV_{period}_LOWER"], 1, 0))
    return df


# ============================================================
# 2️⃣ MOMENTUM / OSCILLATOR INDICATORS
# ============================================================

def compute_rsi(df, period=14):
    """Relative Strength Index (RSI)
    - RSI > 70 → Overbought (Bearish = –1)
    - RSI < 30 → Oversold (Bullish = +1)
    """
    df["RSI"] = ta.rsi(df["Close"], length=period)
    df["RSI_signal"] = np.where(df["RSI"] > 70, -1, np.where(df["RSI"] < 30, 1, 0))
    return df


def compute_stochastic(df):
    """Stochastic Oscillator (%K, %D)
    - %K > 80 → Overbought (–1)
    - %K < 20 → Oversold (+1)
    """
    stoch = ta.stoch(df["High"], df["Low"], df["Close"])
    df = pd.concat([df, stoch], axis=1)
    df["STOCH_signal"] = np.where(df["STOCHk_14_3_3"] > 80, -1,
                           np.where(df["STOCHk_14_3_3"] < 20, 1, 0))
    return df


def compute_momentum(df, period=10):
    """Momentum Indicator
    - MOM > 0 → Bullish
    - MOM < 0 → Bearish
    """
    df["MOM"] = ta.mom(df["Close"], length=period)
    df["MOM_signal"] = np.sign(df["MOM"])
    return df


def compute_roc(df, period=10):
    """Rate of Change (ROC)
    - Positive → Bullish
    - Negative → Bearish
    """
    df["ROC"] = ta.roc(df["Close"], length=period)
    df["ROC_signal"] = np.sign(df["ROC"])
    return df


def compute_williams_r(df, period=14):
    """Williams %R
    - > -20 → Overbought (–1)
    - < -80 → Oversold (+1)
    """
    df["WILLIAMS_%R"] = ta.willr(df["High"], df["Low"], df["Close"], length=period)
    df["WILLIAMS_signal"] = np.where(df["WILLIAMS_%R"] > -20, -1,
                              np.where(df["WILLIAMS_%R"] < -80, 1, 0))
    return df


def compute_cci(df, period=20):
    """Commodity Channel Index (CCI)
    - > +100 → Bullish (+1)
    - < -100 → Bearish (–1)
    """
    df["CCI"] = ta.cci(df["High"], df["Low"], df["Close"], length=period)
    df["CCI_signal"] = np.where(df["CCI"] > 100, 1, np.where(df["CCI"] < -100, -1, 0))
    return df


# ============================================================
# 3️⃣ VOLATILITY / RANGE-BASED INDICATORS
# ============================================================

def compute_bollinger_breakout(df, period=20, n_std=2):
    """Bollinger Bands
    - Price > Upper → Bullish (+1)
    - Price < Lower → Bearish (–1)
    """
    ma = df["Close"].rolling(window=period, min_periods=period).mean()
    sd = df["Close"].rolling(window=period, min_periods=period).std()

    df["MA"] = ma
    df["upper"] = ma + n_std * sd
    df["lower"] = ma - n_std * sd
    df["BB_signal"] = np.where(df["Close"] > df["upper"], 1,
                        np.where(df["Close"] < df["lower"], -1, 0))
    return df


def compute_donchian_channel(df, period=20):
    """Donchian Channel
    - Upper = N-day highest high
    - Lower = N-day lowest low
    - Price > Upper → Bullish
    - Price < Lower → Bearish
    """
    don = ta.donchian(df["High"], df["Low"], lower_length=period)
    df = pd.concat([df, don], axis=1)
    df["DON_signal"] = np.where(df["Close"] > df["DCU_20_20"], 1,
                         np.where(df["Close"] < df["DCL_20_20"], -1, 0))
    return df


def compute_atr(df, period=14):
    """Average True Range (ATR)
    - Measures volatility (higher ATR = larger price swings)
    - Signal not directional but used to gauge risk.
    """
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=period)
    df["ATR_signal"] = np.where(df["ATR"] > df["ATR"].rolling(20).mean(), 1, 0)  # Above avg → volatile
    return df


def compute_keltner_breakout(df, ema_period=20, atr_period=10, multiplier=2):
    """Keltner Channel
    - Close > Upper Band → Bullish (+1)
    - Close < Lower Band → Bearish (–1)
    """
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift(1)).abs()
    lc = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean()
    ema = df["Close"].ewm(span=ema_period, adjust=False).mean()

    df["KC_mid"] = ema
    df["KC_upper"] = ema + multiplier * atr
    df["KC_lower"] = ema - multiplier * atr
    df["KC_signal"] = np.where(df["Close"] > df["KC_upper"], 1,
                        np.where(df["Close"] < df["KC_lower"], -1, 0))
    return df


def compute_price_channel_breakout(df, period=20):
    """Price Channel Breakout
    - Close > rolling High → Bullish
    - Close < rolling Low → Bearish
    """
    df["PC_High"] = df["High"].rolling(window=period).max()
    df["PC_Low"] = df["Low"].rolling(window=period).min()
    df["PC_signal"] = np.where(df["Close"] > df["PC_High"], 1,
                        np.where(df["Close"] < df["PC_Low"], -1, 0))
    return df


# ============================================================
# 4️⃣ DIRECTIONAL / COMPOSITE INDICATORS
# ============================================================

def compute_macd(df):
    """MACD
    - MACD > Signal → Bullish
    - MACD < Signal → Bearish
    """
    macd = ta.macd(df["Close"])
    df = pd.concat([df, macd], axis=1)
    df["MACD_signal"] = np.where(df["MACD_12_26_9"] > df["MACDs_12_26_9"], 1, -1)
    return df


def compute_parabolic_sar(df):
    """Parabolic SAR
    - Close > PSAR → Bullish
    - Close < PSAR → Bearish
    """
    psar = ta.psar(df["High"], df["Low"], df["Close"])
    df["PSAR"] = psar["PSARl_0.02_0.2"]
    df["PSAR_signal"] = np.where(df["Close"] > df["PSAR"], 1, -1)
    return df


def compute_heikin_ashi_trend(df):
    """Heikin-Ashi Trend
    - HA_close > HA_open → Bullish
    - HA_close < HA_open → Bearish
    """
    ha = ta.ha(df["Open"], df["High"], df["Low"], df["Close"])
    df = pd.concat([df, ha], axis=1)
    df["HA_signal"] = np.where(df["HA_close"] > df["HA_open"], 1, -1)
    return df


# ============================================================
# 5️⃣ MASTER FUNCTION
# ============================================================

def add_all_indicators(df):
    """Compute all 20 indicators and their discrete trading signals."""
    df = compute_sma(df)
    df = compute_ema(df)
    df = compute_tema(df)
    df = compute_dema(df)
    df = compute_triple_ma_crossover(df)
    df = compute_ma_envelope(df)

    df = compute_rsi(df)
    df = compute_stochastic(df)
    df = compute_momentum(df)
    df = compute_roc(df)
    df = compute_williams_r(df)
    df = compute_cci(df)

    df = compute_bollinger_breakout(df)
    df = compute_donchian_channel(df)
    df = compute_atr(df)
    df = compute_keltner_breakout(df)
    df = compute_price_channel_breakout(df)

    df = compute_macd(df)
    df = compute_parabolic_sar(df)
    df = compute_heikin_ashi_trend(df)

    #print(df.iloc[100])
    return df
