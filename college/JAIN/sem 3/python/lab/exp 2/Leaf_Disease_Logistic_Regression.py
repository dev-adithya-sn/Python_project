import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load Dataset
dataset = pd.read_csv("Leaf_Disease_Dataset_1000.csv")

# Remove ID column
dataset = dataset.drop(columns=["Sample_ID"])

# One-hot encode categorical feature
dataset = pd.get_dummies(dataset, columns=["Plant_Type"], drop_first=True)

# Features and Target
X = dataset.drop(columns=["Disease_Class", "Disease_Severity", "Target_Label"])
y = dataset["Target_Label"]

# Missing value handling
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
model = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("="*60)
print("Leaf Disease Detection - Logistic Regression")
print("="*60)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

print("\nConfusion Matrix")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nPer-Class Accuracy")
for i in range(len(cm)):
    print(f"Class {i}: {cm[i,i]/cm[i].sum():.4f}")

overall = np.trace(cm)/np.sum(cm)
print(f"\nOverall Accuracy: {overall:.4f}")
