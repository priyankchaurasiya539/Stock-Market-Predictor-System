import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score , precision_score , accuracy_score , classification_report , confusion_matrix , f1_score
from xgboost import XGBClassifier


df = pd.read_csv(r"C:\Scikit Learn\Stock Market Predictor System\data\master_data_features.csv")
df = df.drop(columns=[ "Unnamed: 0" ], errors="ignore")
X = df.drop(columns=["TARGET"])
y = df["TARGET"]

print("X shape :" , X.shape)
print("Y shape :" , y.shape)
print("Columns in X : " , (X.columns))


#Split first using date 
train_df = df[df["Date"] < "2022-01-01"]
test_df = df [df["Date"] >= "2022-01-01"]

#Now drop date columns from both 
train_df = train_df.drop(columns=["Date"])
test_df = test_df.drop(columns=["Date"])

#Create x and y 
X_train = train_df.drop(columns=["TARGET"])
X_test = test_df.drop(columns=["TARGET"])

y_train = train_df["TARGET"]
y_test = test_df["TARGET"]

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train:", y_train.value_counts())
print("y_test:", y_test.value_counts())


print("Training done.")

# Remove raw prices — keep only derived features
cols_to_drop = ["NIFTY", "SBI", "CRUDE", 
                "VIX", "USDINR", "TARGET"]

# Step 1 - correct X_train with 9 columns
X_train = train_df.drop(columns=cols_to_drop)
X_test  = test_df.drop(columns=cols_to_drop)

# Step 2 - train model AFTER X_train is ready
model_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)
model_rf.fit(X_train, y_train)  # ← fresh training
y_pred = model_rf.predict(X_test)

# Step 3 - now feature importance
feature_names = list(X_train.columns)
importances   = model_rf.feature_importances_

print(len(feature_names))  # should be 9
print(len(importances))    # should be 9


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
feature_names = X_train.columns
importances = model_rf.feature_importances_

feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(feat_df["Feature"], feat_df["Importance"])
plt.title("Feature Importance — Random Forest")
plt.savefig("feature_importance.png")
plt.show()

xgb_model = XGBClassifier(
    n_estimators = 100 ,
    randome_state = 42,
    eval_metric = 'logloss'
)
xgb_model.fit(X_train , y_train)
y_pred_xgb = xgb_model.predict(X_test)

print("\nXGBoost Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("\nClassification Report:\n", 
      classification_report(y_test, y_pred_xgb))


joblib.dump(model_rf, r"C:\Scikit Learn\Stock Market Predictor System\models\rf_model.pkl")
print("Model saved successfully.")