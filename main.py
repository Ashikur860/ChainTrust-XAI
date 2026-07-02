# ============================================
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ============================================
# 1. LOAD DATA
# ============================================
DATA_DIR = Path(__file__).resolve().parent
features_path = DATA_DIR / "elliptic_txs_features.csv"
classes_path = DATA_DIR / "elliptic_txs_classes.csv"

print(f"Loading features from {features_path.name}...", flush=True)
feature_usecols = [0] + list(range(2, 167))
features = pd.read_csv(
    features_path,
    header=None,
    usecols=feature_usecols,
    dtype={0: np.int64},
    low_memory=False,
)
features.rename(columns={0: "txId"}, inplace=True)

print(f"Loading labels from {classes_path.name}...", flush=True)
classes = pd.read_csv(classes_path, dtype={"txId": np.int64, "class": str})
classes.columns = ["txId", "label"]
classes["label"] = classes["label"].map({"1": 1, "2": 0})
classes = classes.dropna(subset=["label"]).astype({"label": int})

# Join by txId to keep data aligned and drop transaction ids before training.
data = features.merge(classes[["txId", "label"]], on="txId", how="inner")
data = data.drop(columns=["txId"]).dropna()

X = data.drop("label", axis=1)
y = data["label"].astype(int)

# ============================================
# 2. TRAIN/TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Data loaded successfully!")
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# ============================================
# 3. MODELS (6 models)
# ============================================

# 3.1 Logistic Regression
print("Training Logistic Regression...", flush=True)
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression finished.", flush=True)

# 3.2 Decision Tree
print("Training Decision Tree...", flush=True)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree finished.", flush=True)

# 3.3 SVM (Linear Kernel)
print("Training SVM...", flush=True)
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("SVM finished.", flush=True)

# 3.4 Random Forest
print("Training Random Forest...", flush=True)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest finished.", flush=True)

# 3.5 XGBoost
print("Training XGBoost...", flush=True)
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print("XGBoost finished.", flush=True)

# 3.6 Ensemble (Soft Voting)
print("Training ensemble model...", flush=True)
ensemble = VotingClassifier(
    estimators=[("rf", rf), ("xgb", xgb)],
    voting="soft"
)
ensemble.fit(X_train, y_train)
y_pred_ens = ensemble.predict(X_test)
print("Ensemble model finished.", flush=True)

# ============================================
# 4. RESULTS TABLE
# ============================================
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "SVM", "Random Forest", "XGBoost", "Ensemble"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_dt),
        accuracy_score(y_test, y_pred_svm),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_xgb),
        accuracy_score(y_test, y_pred_ens)
    ]
})

print("\n=== MODEL COMPARISON (6 MODELS) ===")
print(results.to_string(index=False))

# ============================================
# 5. DETAILED REPORTS FOR BEST 2 MODELS
# ============================================
print("\n=== XGBOOST DETAILED REPORT ===")
print(classification_report(y_test, y_pred_xgb, target_names=["Legit", "Illicit"]))

print("\n=== ENSEMBLE DETAILED REPORT ===")
print(classification_report(y_test, y_pred_ens, target_names=["Legit", "Illicit"]))

# ============================================
# 6. CONFUSION MATRIX (XGBoost)
# ============================================
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legit", "Illicit"],
            yticklabels=["Legit", "Illicit"])
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("xgboost_confusion_matrix.png", dpi=300)
print("✅ Saved: xgboost_confusion_matrix.png")
plt.close()

# ============================================
# 7. ROC CURVE (XGBoost)
# ============================================
y_prob = xgb.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300)
print("✅ Saved: roc_curve.png")
plt.close()

# ============================================
# 8. FEATURE IMPORTANCE (XGBoost)
# ============================================
importances = xgb.feature_importances_
indices = np.argsort(importances)[-10:]

plt.figure(figsize=(8, 5))
plt.barh(range(len(indices)), importances[indices], color="steelblue")
plt.yticks(range(len(indices)), [f"F{i}" for i in indices])
plt.xlabel("Importance")
plt.title("Top 10 Feature Importance - XGBoost")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
print("✅ Saved: feature_importance.png")
plt.close()

# ============================================
# 9. CONFUSION MATRIX (Ensemble - Best 2nd model)
# ============================================
cm_ens = confusion_matrix(y_test, y_pred_ens)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_ens, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Legit", "Illicit"],
            yticklabels=["Legit", "Illicit"])
plt.title("Confusion Matrix - Ensemble Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("ensemble_confusion_matrix.png", dpi=300)
print("✅ Saved: ensemble_confusion_matrix.png")
plt.close()

print("\n🎉 All models trained and figures saved successfully!")
