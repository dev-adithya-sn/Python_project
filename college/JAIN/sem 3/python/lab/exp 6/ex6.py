

import numpy as np
import pandas as pd


# =====================================================================
# STEP 1: LOAD THE DATASET
# =====================================================================

dataset = pd.read_csv('financial_fraud_dataset.csv')

# Drop the ID column - it's just a label, not a predictive feature
dataset = dataset.drop(columns=['transaction_id'])

print("Dataset shape:", dataset.shape)
print("Fraud cases:", dataset['is_fraud'].sum(), "out of", len(dataset),
      "(%.1f%%)" % (dataset['is_fraud'].mean() * 100))
print(dataset.head(), "\n")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Identify which columns are categorical (text) vs numeric, by position
categorical_cols = dataset.iloc[:, :-1].select_dtypes(exclude=[np.number]).columns.tolist()
categorical_idx = [dataset.columns.get_loc(c) for c in categorical_cols]
print("Categorical columns:", categorical_cols, "at indices", categorical_idx)


# =====================================================================
# STEP 2: ENCODE CATEGORICAL VARIABLES
# =====================================================================
# merchant_category and channel are text -> convert to One-Hot columns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'), categorical_idx)
    ],
    remainder='passthrough'
)

X = ct.fit_transform(X)
X = X.toarray() if hasattr(X, "toarray") else X


# =====================================================================
# STEP 3: SPLIT INTO TRAINING AND TEST SETS
# =====================================================================
# stratify=y keeps the same fraud percentage in both train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)


# =====================================================================
# STEP 4: FEATURE SCALING
# =====================================================================

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# =====================================================================
# STEP 5: ALGORITHMS
# =====================================================================

#QDA #11
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
classifier = QuadraticDiscriminantAnalysis()
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)

# =====================================================================
# STEP 6: EVALUATE THE MODEL
# =====================================================================

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print('==============================================')
#print('Financial Fraud Detection -', best_name)
print('==============================================')
print('Accuracy  : %.2f %%' % (accuracy * 100))
print('Precision : %.2f %%   (of flagged transactions, how many were really fraud)' % (precision * 100))
print('Recall    : %.2f %%   (of real fraud cases, how many did we catch)' % (recall * 100))
print('F1 Score  : %.4f' % f1)

print('\nConfusion Matrix:')
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('  [ [True Negative   False Positive] ')
print('    [False Negative  True Positive ] ]')

print('\nFull Classification Report:')
print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'], zero_division=0))


# =====================================================================
# STEP 8: ACTUAL VS PREDICTED TABLE (first 20 rows)
# =====================================================================

result = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
result['Actual'] = result['Actual'].map({0: 'Legit', 1: 'Fraud'})
result['Predicted'] = result['Predicted'].map({0: 'Legit', 1: 'Fraud'})

print('\n==============================================')
print('Actual vs Predicted (first 20 rows)')
print('==============================================')
print(result.head(20))

result.to_csv('Fraud_Actual_vs_Predicted.csv', index=False)


# =====================================================================
# STEP 9: SAVE THE TRAINED MODEL
# =====================================================================

#import joblib

#joblib.dump({"model": classifier, "scaler": sc, "encoder": ct, "model_name": best_name},
 #           "fraud_detection_model.pkl")
#print("\nModel saved as fraud_detection_model.pkl")