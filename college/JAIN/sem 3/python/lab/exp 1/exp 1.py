import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('network_intrusion_dataset_label.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X[:, 1:6])
X[:, 1:6] = imputer.transform(X[:, 1:6])

# Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
classifier = LinearDiscriminantAnalysis()
classifier.fit(X_train, y_train)

# Predicting test Results
y_pred = classifier.predict(X_test)

# Making confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# True positive (TP): We predict a label of 1 (positive) and true label is 1
tp = np.sum(np.logical_and(y_test == 1, y_pred == 1))

# True Negative (TN): We predict label of 0 (Negative) and true is 0
tn = np.sum(np.logical_and(y_test == 0, y_pred == 0))

# False positive (FP): We predict a label of 1, but true is 0
fp = np.sum(np.logical_and(y_test == 0, y_pred == 1))

# False negative (FN): We predict a label of 0 (Negative) but true label is 1
fn = np.sum(np.logical_and(y_test == 1, y_pred == 0))

# Sensitivity, hit rate, recall, or true positive rate
tpr = tp / (tp + fn)

# Specificity or true negative rate
tnr = tn / (tn + fp)

# Precision or positive predictive value
ppv = tp / (tp + fp)

# Negative predictive value
npv = tn / (tn + fn)

# Fall out or false positive rate
fpr = fp / (fp + tn)

# False negative rate
fnr = fn / (tp + fn)

# False discovery rate
fdr = fp / (tp + fp)

# Overall accuracy
acc = (tp + tn) / (tp + fp + fn + tn)

if tp > 0:
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)

print('\nConfusion matrix for Logistic Regression Classifier:\n', confusion_matrix(y_test, y_pred))
print('\nTrue Positive: %d' % (tp))
print('\nTrue Negative: %d' % (tn))
print('\nFalse Positive: %d' % (fp))
print('\nFalse Negative: %d' % (fn))
print('\nSensitivity, hit rate, recall, or true positive rate: %f' % (tpr))
print('\nSpecificity or true negative rate: %f' % (tnr))
print('\nPrecision or positive predictive value: %f' % (ppv))
print('\nNegative predictive value: %f' % (npv))
print('\nFall out or false positive rate: %f' % (fpr))
print('\nFalse negative rate: %f' % (fnr))
print('\nFalse discovery rate: %f' % (fdr))
print('\nPrecision: %f' % (precision))
print('\nRecall: %f' % (recall))
print('\nOverall accuracy for classifier: %f' % (acc))