import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# =========================
# 1. LOAD DATA
# =========================
features = pd.read_csv("elliptic_txs_features.csv", header=None)
classes = pd.read_csv("elliptic_txs_classes.csv")

classes.columns = ["txId", "label"]
classes["label"] = classes["label"].map({"1": 1, "2": 0})
classes = classes.dropna()

data = pd.concat([features, classes["label"]], axis=1)
data = data.dropna()

X = data.drop("label", axis=1)
y = data["label"].astype(int)

# =========================
# 2. TRAIN/TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("✅ Data loaded successfully!")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

# =========================
# 3. XGBOOST MODEL
# =========================
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

print("\n=== XGBOOST RESULTS ===")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))