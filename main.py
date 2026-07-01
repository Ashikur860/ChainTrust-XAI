import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

# LOAD DATA
features = pd.read_csv("elliptic_txs_features.csv", header=None)
classes = pd.read_csv("elliptic_txs_classes.csv")

classes.columns = ["txId", "label"]
classes["label"] = classes["label"].map({"1": 1, "2": 0})
classes = classes.dropna()

data = pd.concat([features, classes["label"]], axis=1)
data = data.dropna()

X = data.drop("label", axis=1)
y = data["label"].astype(int)

# TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# MODELS
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

ensemble = VotingClassifier(
    estimators=[("rf", rf), ("xgb", xgb)],
    voting="soft"
)
ensemble.fit(X_train, y_train)
y_pred_ens = ensemble.predict(X_test)

print("=== RESULTS ===")
print("RF accuracy:", accuracy_score(y_test, y_pred_rf))
print("XGBoost accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Ensemble accuracy:", accuracy_score(y_test, y_pred_ens))

# SAVE FIGURES
cm = confusion_matrix(y_test, y_pred_xgb)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("xgboost_confusion_matrix.png", dpi=300)
plt.close()

y_prob = xgb.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.title("ROC Curve - XGBoost")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("roc_curve.png", dpi=300)
plt.close()

importances = xgb.feature_importances_
indices = np.argsort(importances)[-10:]

plt.figure(figsize=(8, 5))
plt.barh(range(len(indices)), importances[indices], color="skyblue")
plt.yticks(range(len(indices)), [f"F{i}" for i in indices])
plt.title("Top 10 Feature Importance - XGBoost")
plt.xlabel("Importance")
plt.savefig("feature_importance.png", dpi=300)
plt.close()