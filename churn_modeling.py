# Telco Customer Churn Prediction - 
# Author: Melek Ikiz
# Description: Machine Learning models (Random Forest & Logistic Regression) to predict customer churn with ROC comparison

# -------------------------------------
# 1. IMPORT LIBRARIES
# -------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")

# -------------------------------------
# 2. LOAD DATASET
# -------------------------------------

df = pd.read_csv("Churn_Modelling.csv")

# -------------------------------------
# 3. DATA CLEANING & PREPROCESSING
# -------------------------------------
df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)
df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})
df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

# Features and Target
X = df.drop("Exited", axis=1)
y = df["Exited"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -------------------------------------
# 4. BALANCING DATA WITH SMOTE
# -------------------------------------
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Original Class Distribution:\n", y_train.value_counts())
print("After SMOTE:\n", y_train_smote.value_counts())

# -------------------------------------
# 5. RANDOM FOREST MODEL & GRID SEARCH
# -------------------------------------
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_smote, y_train_smote)
y_pred = rf_model.predict(X_test)

print("\nInitial Random Forest Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()

# GridSearch for Best Hyperparameters
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), 
                            param_grid=param_grid, 
                            cv=3, 
                            n_jobs=-1, 
                            verbose=2)

grid_search.fit(X_train_smote, y_train_smote)
print("\nBest Parameters:", grid_search.best_params_)

# Best model prediction
y_pred_best = grid_search.best_estimator_.predict(X_test)
print("\nTuned Random Forest Classification Report:")
print(classification_report(y_test, y_pred_best))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_best)

# -------------------------------------
# 6. LOGISTIC REGRESSION MODEL
# -------------------------------------
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_smote, y_train_smote)

# -------------------------------------
# 7. ROC & AUC COMPARISON
# -------------------------------------
y_probs_logreg = logreg.predict_proba(X_test)[:, 1]
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_probs_logreg)
auc_logreg = roc_auc_score(y_test, y_probs_logreg)

# Random Forest ROC
best_model = grid_search.best_estimator_
y_probs_rf = best_model.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_probs_rf)
auc_rf = roc_auc_score(y_test, y_probs_rf)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={auc_rf:.4f})")
plt.plot(fpr_logreg, tpr_logreg, label=f"Logistic Regression (AUC={auc_logreg:.4f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid()
plt.show()

print(f"\nRandom Forest AUC Score: {auc_rf:.4f}")
print(f"Logistic Regression AUC Score: {auc_logreg:.4f}")
