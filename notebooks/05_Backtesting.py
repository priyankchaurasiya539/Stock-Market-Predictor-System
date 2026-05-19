import pandas as pd 
import joblib
import os
from sklearn.metrics import confusion_matrix , classification_report ,accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "master_data_features.csv"))
model = joblib.load(os.path.join(BASE_DIR, "..", "models", "rf_model.pkl"))


cols_to_drop = ["Unnamed: 0" , "Date" , "NIFTY" , "SBI" , "CRUDE" , "VIX" , "USDINR"]



#--------------Rate Hike Crash------------------
rate_df = df[(df["Date"] >= "2022-01-01") & (df["Date"] <= "2022-12-31")]
rate_df = rate_df.drop(columns=cols_to_drop , errors="ignore")
X_rate = rate_df.drop(columns=["TARGET"])
y_rate = rate_df["TARGET"]
rate_pred = model.predict(X_rate)
print("Rate Hike Crash 2022")
print("Accuracy :" , accuracy_score(y_rate , rate_pred))
print(classification_report(y_rate , rate_pred))
print(confusion_matrix(y_rate , rate_pred))


#------------------Bull Run 2023-----------------------
bull_df = df[(df["Date"] >= "2023-01-01") & (df["Date"] <= "2023-12-31")].copy()
bull_df = bull_df.drop(columns=cols_to_drop , errors= "ignore")
X_bull_run = bull_df.drop(columns=["TARGET"])
y_bull_run = bull_df["TARGET"]
bull_pred = model.predict(X_bull_run)
print("Bull Run 2023")
print("Accuracy : " , accuracy_score(y_bull_run , bull_pred))
print(classification_report(y_bull_run , bull_pred))
print(confusion_matrix(y_bull_run , bull_pred))


