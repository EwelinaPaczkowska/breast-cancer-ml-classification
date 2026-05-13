from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = pd.Series(
    data.target,
    name="diagnosis"
)

print("\n=== WISCONSIN BREAST CANCER - KLASYFIKACJA ===\n")

print(f"Wczytano dane: {len(X)} próbek, {X.shape[1]} cech\n")

print("Rozkład klas w pełnym datasecie:")

print(
    f"  Benign (1): {sum(y == 1)} "
    f"({sum(y == 1) / len(y) * 100:.1f}%)"
)

print(
    f"  Malignant (0): {sum(y == 0)} "
    f"({sum(y == 0) / len(y) * 100:.1f}%)"
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"\nPodział danych: {len(X_train)} treningowych, {len(X_test)} testowych")

print("\nRozkład klas w zbiorze treningowym:")

print(
    f"  Benign: {sum(y_train == 1) / len(y_train) * 100:.1f}%"
)

print(
    f"  Malignant: {sum(y_train == 0) / len(y_train) * 100:.1f}%"
)

print("\nRozkład klas w zbiorze testowym:")

print(
    f"  Benign: {sum(y_test == 1) / len(y_test) * 100:.1f}%"
)

print(
    f"  Malignant: {sum(y_test == 0) / len(y_test) * 100:.1f}%"
)

print("\n=== LOGISTIC REGRESSION ===")

lr_model = LogisticRegression(max_iter=10000)

lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_precision = precision_score(y_test, y_pred_lr)
lr_recall = recall_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr)

print(f"Accuracy: {lr_accuracy:.2f}")
print(f"Precision: {lr_precision:.2f}")
print(f"Recall: {lr_recall:.2f}")
print(f"F1-score: {lr_f1:.2f}")

cm_lr = confusion_matrix(y_test, y_pred_lr)

disp_lr = ConfusionMatrixDisplay(
    confusion_matrix=cm_lr,
    display_labels=data.target_names
)

disp_lr.plot(cmap="Blues")

plt.title("Confusion Matrix - Logistic Regression")

plt.savefig(
    "confusion_matrix_lr.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Confusion matrix zapisana: confusion_matrix_lr.png")

print("\n=== RANDOM FOREST ===")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)

print(f"Accuracy: {rf_accuracy:.2f}")
print(f"Precision: {rf_precision:.2f}")
print(f"Recall: {rf_recall:.2f}")
print(f"F1-score: {rf_f1:.2f}")

cm_rf = confusion_matrix(y_test, y_pred_rf)

disp_rf = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf,
    display_labels=data.target_names
)

disp_rf.plot(cmap="Greens")

plt.title("Confusion Matrix - Random Forest")

plt.savefig(
    "confusion_matrix_rf.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Confusion matrix zapisana: confusion_matrix_rf.png")

print("\nAnaliza zakończona.")