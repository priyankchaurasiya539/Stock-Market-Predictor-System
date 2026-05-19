import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "master_data_features.csv"))
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

#---------------------------Heatmap----------------------- 
plt.figure(figsize=(14 , 10))
sns.heatmap(df.drop(columns = ["Date"]).corr() , annot=True , cmap = "coolwarm" , fmt = ".2f" , linewidths= 0.5 , linecolor="green")
plt.savefig(os.path.join(BASE_DIR, "..", "Correlation_heatmap.png"))
plt.show()


#-------------------RSI Distribution------------------
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="RSI", bins=50, kde=True)
plt.axvline(x=30, color="red", linestyle="--", label="Oversold (30)")
plt.axvline(x=70, color="green", linestyle="--", label="Overbought (70)")
plt.legend()
plt.title("RSI Distribution")
plt.savefig(os.path.join(BASE_DIR, "..", "rsi_distribution.png"))
plt.show()



#-------------------Scatter Plot ----------------------
plt.figure(figsize=(10 , 6))
sns.scatterplot(data = df , x = "VIX" , y = "NIFTY_return_1d" , hue = "TARGET"  )
plt.title("VIX vs NIFTY Daily Return")
plt.savefig(os.path.join(BASE_DIR, "..", "vix_vs_return.png"))
plt.show()