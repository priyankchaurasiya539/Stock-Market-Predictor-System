import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Scikit Learn\Stock Market Predictor System\data\master_data_features.csv")


#---------------------------Heatmap----------------------- 
# plt.figure(figsize=(14 , 10))
# sns.heatmap(df.drop(columns = ["Date"]).corr() , annot=True , cmap = "coolwarm" , fmt = ".2f" , linewidths= 0.5 , linecolor="green")
# df = df.drop(columns=["Unnamed: 0"], errors="ignore")
# plt.savefig("Correlation_heatmap.png")
# plt.show()


#-------------------RSI Distribution------------------
# plt.figure(figsize=(10, 5))
# sns.histplot(data=df, x="RSI", bins=50, kde=True)
# plt.axvline(x=30, color="red", linestyle="--", label="Oversold (30)")
# plt.axvline(x=70, color="green", linestyle="--", label="Overbought (70)")
# plt.legend()
# plt.title("RSI Distribution")
# plt.savefig("rsi_distribution.png")
# plt.show()



#-------------------Scatter Plot ----------------------
plt.figure(figsize=(10 , 6))
sns.scatterplot(data = df , x = "VIX" , y = "NIFTY_return_1d" , hue = "TARGET"  )
plt.title("VIX vs NIFTY Daily Return")
plt.savefig("vix_vs_return.png")
plt.show()