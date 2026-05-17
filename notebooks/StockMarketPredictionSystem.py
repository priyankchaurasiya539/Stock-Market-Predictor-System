import pandas as pd 

df = pd.read_csv("Master_data.csv")

print(df)

df["NIFTY_return_1d"] = df["NIFTY"].pct_change(periods = 1)
df["NIFTY_return_7d"] = df["NIFTY"].pct_change(periods= 7)
df["NIFTY_return_30d"] = df["NIFTY"].pct_change(periods= 30)


print(df[["NIFTY_return_1d" , "NIFTY_return_7d" , "NIFTY_return_30d"]].head(35))
print("==============================================================================")

df["NIFTY_MA_50"] = df["NIFTY"].rolling(window = 50).mean()
df["NIFTY_MA_200"] = df["NIFTY"].rolling(window = 200).mean()

print(df[["NIFTY_MA_50" , "NIFTY_MA_200"]].head(205))


#RSI formula 
def calculate_rsi(prices , period = 14):
    delta = prices.diff()
    gain = delta.clip(lower = 0 )
    loss = -delta.clip(upper = 0 )
    avg_gain = gain.rolling(window = period).mean()
    avg_loss = loss.rolling(window = period ).mean()
    rs = avg_gain/avg_loss
    rsi = 100 - (100/(1 + rs))
    return rsi
    
df["RSI"] = calculate_rsi(df["NIFTY"])
print(df["RSI"].head(30))
print("\nRSI min:", df["RSI"].min())
print("RSI max:", df["RSI"].max())

df["VIX_lag1"] = df["VIX"].shift(1)
df["VIX_lag7"] = df["VIX"].shift(7)
# df["VIX_lag30"] = df["VIX"].shift(30)

# df["CRUDE_lag1"] = df["CRUDE"].shift(1)
df["CRUDE_lag7"] = df["CRUDE"].shift(7)
# df["CRUDE_lag30"] = df["CRUDE"].shift(30)


print(df[["VIX_lag1" , "VIX_lag7" , "CRUDE_lag7"]].head(10))
print("==============================================================================")

df["TARGET"] = (df["NIFTY"].shift(-30) > df["NIFTY"]).astype(int)
df.dropna(inplace=True)

print("Shape after dropna:", df.shape)
print("\nTARGET distribution:")
print(df["TARGET"].value_counts())


df.to_csv("master_data_features.csv", index=True)
print("File saved successfully")
print("Final columns:", list(df.columns))