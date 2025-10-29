from src.agent.tools.data_tools import fetch_from_yfinance
import os
from src.agent.tools.feature_engineer import build_feature_set

symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
data = fetch_from_yfinance(symbols)
print({sym: df.shape for sym, df in data.items()})



import pandas as pd
df = pd.read_parquet("data/processed/RELIANCE.NS.parquet")
print("before",df.tail())


from src.agent.tools.indicator_tools import add_all_indicators
import pandas as pd

df = pd.read_parquet("data/processed/RELIANCE.NS.parquet")
#print(df.columns)
df = add_all_indicators(df)
print("after",df.tail())
df = build_feature_set(df)
os.makedirs("data/processed/indicators", exist_ok=True)
df.to_parquet("data/processed/indicators/RELIANCE.NS.parquet", compression="snappy")
print(df[["Close", "Swing_Score", "Swing_Sentiment", "Volatility_Score", "Trend_Strength", "Vol_Surge","BB_Width"]].tail(20))
#print(df.columns[-20:])"""
