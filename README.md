STOCK PREDICTION MOVEMENT PREDICTOR USING MACHINE LEARNING 
About Model :   It is a model which predict the behavior of market for the upcoming days .It does not predict the actual future price of any stock but it can predict the market ups and downs by seeing the last 1 day , 7 day , 50 day and 200 days  market trend.

It uses some concepts of Stock/Share Market that are listed below  :- 
•	RSI (Relative Strength Index)
•	MA (Moving averages)
  o	What is Golden cross?
  o	What is Death cross?
•	Daily/Weekly/Monthly Returns


It also uses some time series concepts like -
•	What is lag Feature?
•	Why random train/test split is wrong here why we use chronological split?
•	Rolling window logic
•	What is data leakage.

I have divided my model making into five different phases.

Phase 1 : Data Collection 
I have collected the from yfinance(yahoo finance) by running python code in vs code in its terminal. I take data from 5 sources (NIFTY 50 , SBI , USD/INR ,  CRUDEOIL  , VIX(Fear Index_)) . 

Phase 2: Data Verification 
I see the dip of market in the market crash year 2008 (Global Economic crisis) , 2020 (Covid -19 Pandemic Crisis ) and 2022 (Rate Hike Crisis).


Phase 3 : Feature Engineering 
STEP 1 : Returns – 1d , 7d , 30d using  pct_change()
STEP 2 : Moving averages – 50MA and 200MA
STEP 3 : RSI – 14 Day momentum Indicator
STEP 4 : Lag features – VIX_lag1 , VIX_lag7 , CRUDE_lag7
STEP 5 : Target Variable – shift(-30) , binary classification.

Phase 4 : EDA (On-Going)



