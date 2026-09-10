import numpy as np
import pandas as pd

# Import dataset
dataset = pd.read_csv("agri_soil_crop_dataset_1000.csv")

# Independent variables
X = dataset.iloc[:, :-1].values

# Dependent variable
y = dataset.iloc[:, -1].values

##################################################
# Encoding categorical columns
##################################################

from sklearn.preprocessing import LabelEncoder

le_texture = LabelEncoder()
X[:, 11] = le_texture.fit_transform(X[:, 11])

le_soil = LabelEncoder()
X[:, 12] = le_soil.fit_transform(X[:, 12])

# Encode target class
le_class = LabelEncoder()
y = le_class.fit_transform(y)

##################################################
# Handle Missing Values
##################################################

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')

# Numeric columns (excluding categorical columns)
X[:,1:11] = imputer.fit_transform(X[:,1:11])

##################################################
# Train Test Split
##################################################

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.40,
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
# CatBoost Classifier
##################################################
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

y_pred=classifier.predict(X_test)

##################################################
# Evaluation
##################################################

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("Accuracy : ", accuracy_score(y_test, y_pred))

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix\n")
cm = confusion_matrix(y_test, y_pred)
print(cm)

##################################################
# Multi-class Metrics
##################################################

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average='weighted'
)

recall = recall_score(
    y_test,
    y_pred,
    average='weighted'
)

f1 = f1_score(
    y_test,
    y_pred,
    average='weighted'
)

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

for crop, acc in zip(le_class.classes_, class_accuracy):
    print(f"{crop:12s} : {acc:.4f}")
