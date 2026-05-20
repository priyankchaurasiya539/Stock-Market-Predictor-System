import streamlit as st
import pandas as pd 
import joblib
from PIL import Image

import yfinance as yf
from datetime import date, timedelta

st.set_page_config(page_title= "Stock Market Predictor system" , layout="wide")
page = st.sidebar.radio("Pick The page : " , ["EDA Dashboard" , "Backtesting Results" , "Predict"])

if page == "EDA Dashboard":
    st.title("EDA Dashboard")
    img1 = Image.open("Correlation_heatmap.png")
    st.image(img1 , caption="Correlation Heatmap" , use_container_width=True)

    img2 = Image.open("rsi_distribution.png")
    st.image(img2 , caption="RSI Distribution" , use_container_width=True)

    img3 = Image.open("vix_vs_return.png")
    st.image(img3 , caption="VIX vs Return" , use_container_width=True)

elif page == "Backtesting Results":
    st.title("Bactesting Results")
    df_backtest = pd.DataFrame({
        "Event" : ["Rate Hike 2022" , "Bull Run 2023"],
        "Accuracy" :[ 0.55 , 0.75 ], 
        "Class 0 Recall" : [0.31 , 0.00 ], 
        "Class 1 Recall" : [0.85 , 0.99] 
    })
    st.dataframe(df_backtest)

# elif page == "Predict":
#     st.info("Tip: See the values from TradingView and NSE website , then , put here for correct prediction.")
#     NIFTY_return_1d = st.slider("NIFTY_return_1d" , min_value=-0.4 , max_value=0.4 , value=0.0)
#     st.caption("See the return of NIFTY today , See from NSE.")


#     NIFTY_return_7d = st.slider("NIFTY_return_7d", min_value= -0.6 , max_value=0.6 , value=0.0)
#     st.caption("See the return of NIFTY in last 7 days , See from NSE")


#     NIFTY_return_30d = st.slider( "NIFTY_return_30d", min_value=-0.7 , max_value=0.7 , value=0.0)
#     st.caption("See the return of NIFTY in last 30 days , See from NSE.")


#     RSI = st.slider("RSI", min_value=0 , max_value=100 , value = 50)


#     VIX_lag1 = st.slider("VIX_lag1", min_value=10 , max_value= 80 , value=20)


#     VIX_lag7 = st.slider("VIX_lag7", min_value=10 , max_value=80 , value=20)

#     NIFTY_MA_50 = st.number_input("NIFTY_MA_50" , min_value=5000 , max_value=25000 , value=15000)
#     NIFTY_MA_200 = st.number_input("NIFTY_MA_200" , min_value=5000 , max_value=25000 , value=15000)
#     CRUDE_lag7 = st.number_input("CRUDE_lag7" , min_value=30 , max_value=130 , value= 70)

#     if st.button("Predict"):
#         model = joblib.load("models/rf_model.pkl")

#         df_stocks = pd.DataFrame({
#     "NIFTY_return_1d": [NIFTY_return_1d],
#     "NIFTY_return_7d": [NIFTY_return_7d],
#     "NIFTY_return_30d": [NIFTY_return_30d],
#     "NIFTY_MA_50": [NIFTY_MA_50],
#     "NIFTY_MA_200": [NIFTY_MA_200],
#     "RSI": [RSI],
#     "VIX_lag1": [VIX_lag1],
#     "VIX_lag7": [VIX_lag7],
#     "CRUDE_lag7": [CRUDE_lag7]
# })
        

#         prediction = model.predict(df_stocks)
#         probability = model.predict_proba(df_stocks)

#         if prediction[0] == 1 :
#             st.success("NIFTY will go UP ↑ ")

#         else:
#             st.error("NIFTY will go DOWN ↓ ")

#         # st.write("Probability : " , probability)
#         st.write(f"Confidence: {probability[0][1]*100:.1f}% chance of going UP")
# streamlit run streamlit_app.py


elif page == "Predict":
    st.title("Predict")
    st.info("Click the button to fetch live data and predict NIFTY movement for next 30 days.")
    
    end_date = date.today()
    start_date = end_date - timedelta(days=1100)
    
    if st.button("Fetch Live Data & Predict"):
        with st.spinner("Fetching live data..."):
            nifty = yf.download("^NSEI", start=start_date, end=end_date)["Close"].squeeze()
            vix = yf.download("^INDIAVIX", start=start_date, end=end_date)["Close"].squeeze()
            crude = yf.download("CL=F", start=start_date, end=end_date)["Close"].squeeze()
            sbi =yf.download("SBIN.NS" , start=start_date , end = end_date)["Close"].squeeze()
            usdinr = yf.download("INR=X" , start=start_date ,end = end_date)["Close"].squeeze()

            df = pd.DataFrame({
    "NIFTY": nifty,
    "VIX": vix,
    "CRUDE": crude,
    "SBI": sbi,
    "USDINR": usdinr
})
            df.dropna(inplace=True)


            df["NIFTY_return_1d"] = df["NIFTY"].pct_change(1)
            df["NIFTY_return_7d"] = df["NIFTY"].pct_change(7)
            df["NIFTY_return_30d"] = df["NIFTY"].pct_change(30)


            df["NIFTY_MA_50"] = df["NIFTY"].rolling(50).mean()
            df["NIFTY_MA_200"] = df["NIFTY"].rolling(200).mean()

            df["VIX_lag1"] = df["VIX"].shift(1)
            df["VIX_lag7"] = df["VIX"].shift(7)
            df["CRUDE_lag7"] = df["CRUDE"].shift(7)


            delta = df["NIFTY"].diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            df["RSI"] = 100 - (100 / (1 + rs))

            df.dropna(inplace=True)

            features = ["NIFTY_return_1d", "NIFTY_return_7d", "NIFTY_return_30d",
            "NIFTY_MA_50", "NIFTY_MA_200", "RSI",
            "VIX_lag1", "VIX_lag7", "CRUDE_lag7"]

            if df.empty or len(df) < 10:
                st.error("Not enough data fetched. Please try again later.")
                st.stop()

            st.write(f"Data fetched: {len(df)} rows")
            latest = df[features].iloc[[-1]]

            model = joblib.load("models/rf_model.pkl")
            prediction = model.predict(latest)
            probability = model.predict_proba(latest)


            if prediction[0] == 1:
                st.success("NIFTY will go UP in next 30 days ↑")
            else:
                st.error("NIFTY will go DOWN in next 30 days ↓")

            st.write(f"Confidence: {probability[0][1]*100:.1f}% chance of going UP")

