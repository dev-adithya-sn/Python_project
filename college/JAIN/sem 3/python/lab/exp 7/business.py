"""
Business Analytics — Customer Satisfaction Prediction
========================================================
Goal: Predict whether a customer gives HIGH satisfaction (4 or 5)
vs LOW satisfaction (1, 2, or 3), using order/sales features.

Why binary instead of 5-class:
  Raw target distribution was 5:483, 4:397, 3:119, 2:1, 1:0.
  Class "2" has a single row — a 5-class model can't be trained or
  evaluated properly on that. Collapsing to High/Low gives a real,
  usable classification problem (880 High vs 120 Low).

Model: Random Forest Classifier
Output: Accuracy, precision/recall/F1, confusion matrix, feature importance.

test size 0.2,0.4,0.6 and use
top 5 best accuracy in the reccord

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
df = pd.read_csv("business_analytics_1000_rows.csv")

# ---------------------------------------------------------
# 2. Build target: High (1) if satisfaction >= 4, else Low (0)
# ---------------------------------------------------------
df["satisfaction_class"] = (df["customer_satisfaction_1_to_5"] >= 4).astype(int)

# ---------------------------------------------------------
# 3. Select features
#    Dropped: order_id, customer_id (unique identifiers, no predictive value)
#    Dropped: order_date (raw date string, not usable directly)
#    Dropped: currency (constant column, all "USD")
#    Dropped: customer_satisfaction_1_to_5 (this is the source of the target — leakage)
# ---------------------------------------------------------
drop_cols = ["order_id", "customer_id", "order_date", "currency",
             "customer_satisfaction_1_to_5", "satisfaction_class"]
X = df.drop(columns=drop_cols)
y = df["satisfaction_class"]

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

print(f"Categorical features: {categorical_cols}")
print(f"Numeric features: {numeric_cols}")
print(f"Target balance -> High: {y.sum()}, Low: {len(y) - y.sum()}\n")

# ---------------------------------------------------------
# 4. Encode categorical columns
# ---------------------------------------------------------
X_encoded = X.copy()
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])
    encoders[col] = le

# ---------------------------------------------------------
# 5. Train/test split (stratified to preserve class ratio)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 6. Train model
# ---------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    class_weight="balanced"  # compensates for 880:120 imbalance
)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")

print("Classification report:")
print(classification_report(y_test, y_pred, target_names=["Low", "High"]))

print("Confusion matrix (rows=actual, cols=predicted, order=[Low, High]):")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------
# 8. Feature importance
# ---------------------------------------------------------
importances = pd.Series(model.feature_importances_, index=X_encoded.columns)
importances = importances.sort_values(ascending=False)
print("\nTop 10 most important features:")
print(importances.head(10))

# ---------------------------------------------------------
# 9. Baseline comparison (majority-class guess)
# ---------------------------------------------------------
majority_class_accuracy = max(y_test.mean(), 1 - y_test.mean())
print(f"\nBaseline (always predict majority class) accuracy: {majority_class_accuracy:.4f}")
print(f"Model improvement over baseline: {(accuracy - majority_class_accuracy)*100:.2f} percentage points")