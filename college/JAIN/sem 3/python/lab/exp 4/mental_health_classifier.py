import numpy as np
import pandas as pd

##################################################
# Import dataset
##################################################
dataset = pd.read_csv("/Users/adithya/college/JAIN/sem 3/python/exp 4/mental_health_disorder_dataset_1000_rows.csv")

# Drop the ID column - it's just a label, carries no predictive signal
dataset = dataset.drop(columns=["Patient_ID"])

# Substance_Use has ~611 missing values out of 1000. In a survey field like this,
# a blank almost always means "did not report using any substance", so we treat
# missing as its own category rather than imputing a statistic into it.
dataset["Substance_Use"] = dataset["Substance_Use"].fillna("None")

##################################################
# Merge extremely rare target classes
##################################################
# Bipolar Disorder (2 rows) and Schizophrenia (1 row) are too rare to model or
# even stratify-split reliably. Folding them into "Other Disorder" keeps every
# class learnable and lets us stratify the train/test split safely.
rare_classes = ["Bipolar Disorder", "Schizophrenia"]
dataset["Target_Disorder"] = dataset["Target_Disorder"].replace(
    {c: "Other Disorder" for c in rare_classes}
)

print("Target class distribution after merging rare classes:")
print(dataset["Target_Disorder"].value_counts())

##################################################
# Independent / dependent variables
##################################################
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Column index map (after dropping Patient_ID):
# 0 Age, 1 Gender, 2 Sleep_Hours, 3 Stress_Level_0_10, 4 Anxiety_Score_0_10,
# 5 Depression_Score_0_10, 6 Social_Support_0_10, 7 Exercise_Hours_Per_Week,
# 8 Work_Study_Hours_Per_Day, 9 Substance_Use, 10 Family_History,
# 11 Currently_In_Therapy
categorical_idx = [1, 9, 10, 11]   # nominal categorical columns
numeric_idx = [0, 2, 3, 4, 5, 6, 7, 8]

##################################################
# Encoding categorical columns
##################################################
from sklearn.preprocessing import LabelEncoder

encoders = {}
for idx in categorical_idx:
    le = LabelEncoder()
    X[:, idx] = le.fit_transform(X[:, idx])
    encoders[idx] = le

# Encode target class
le_class = LabelEncoder()
y = le_class.fit_transform(y)

##################################################
# Handle Missing Values (numeric columns only)
##################################################
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="mean")
X[:, numeric_idx] = imputer.fit_transform(X[:, numeric_idx])

##################################################
# Train Test Split
##################################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,          # smaller test size leaves more data to learn the rarer classes
    random_state=42,
    stratify=y
)

##################################################
# Feature Scaling
##################################################
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

##################################################
# Random Forest Classifier
##################################################
# A single Decision Tree overfits easily and is unstable on imbalanced, noisy
# tabular data like this. A Random Forest averages many trees, which raises
# accuracy and generalizes much better. class_weight='balanced' compensates
# for the uneven class sizes (Depression=270 vs Panic Disorder=15, etc.)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(
    n_estimators=300,
    criterion="entropy",
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

##################################################
# Evaluation
##################################################
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

print("\nAccuracy : ", accuracy_score(y_test, y_pred))
print("\nClassification Report\n")
print(classification_report(y_test, y_pred, target_names=le_class.classes_, zero_division=0))
print("\nConfusion Matrix\n")
cm = confusion_matrix(y_test, y_pred)
print(cm)

##################################################
# Multi-class Metrics
##################################################
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\n----------------------------")
print("Performance Metrics")
print("----------------------------")
print("Accuracy  :", accuracy)
print("Precision :", precision)
print("Recall    :", recall)
print("F1 Score  :", f1)

##################################################
# Per-Class Accuracy
##################################################
class_accuracy = cm.diagonal() / cm.sum(axis=1)
print("\nPer-Class Accuracy\n")
for disorder, acc in zip(le_class.classes_, class_accuracy):
    print(f"{disorder:32s} : {acc:.4f}")

##################################################
# Feature Importance
##################################################
feature_names = ["Age", "Gender", "Sleep_Hours", "Stress_Level_0_10", "Anxiety_Score_0_10",
                  "Depression_Score_0_10", "Social_Support_0_10", "Exercise_Hours_Per_Week",
                  "Work_Study_Hours_Per_Day", "Substance_Use", "Family_History", "Currently_In_Therapy"]
importances = classifier.feature_importances_
print("\nFeature Importance (most to least influential)\n")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"{name:28s} : {imp:.4f}")
