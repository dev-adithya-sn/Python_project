import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Agricultural_Price_Prediction_1000_Rows.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# ---------------------------------------------------------
# Encoding Categorical Variables
# ---------------------------------------------------------

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Crop = 1
# State = 2
# Season = 3
# Soil_Type = 5
# Market_Type = 6

ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'),
         [1, 2, 3, 5, 6])
    ],
    remainder='passthrough'
)

X = ct.fit_transform(X)

X = X.toarray() if hasattr(X, "toarray") else X


# ---------------------------------------------------------
# Splitting the dataset into Training set and Test set
# ---------------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0
)


# ---------------------------------------------------------
# Feature Scaling
# ---------------------------------------------------------

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)


# ---------------------------------------------------------
# Random Forest Regression
# ---------------------------------------------------------

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(
    n_estimators=500,
    random_state=0,
    max_features='sqrt'
)

regressor.fit(X_train, y_train)


# ---------------------------------------------------------
# Predicting the Test Set results
# ---------------------------------------------------------

y_pred = regressor.predict(X_test)


# ---------------------------------------------------------
# Regression Evaluation
# ---------------------------------------------------------

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

mape = mean_absolute_percentage_error(
    y_test,
    y_pred
) * 100


# ---------------------------------------------------------
# Printing Results
# ---------------------------------------------------------

print('\n==============================================')

print('\nAgricultural Price Prediction')

print('\nRandom Forest Regression')

print('\n==============================================')

print('\nMean Absolute Error : %f' % (mae))

print('\nRoot Mean Square Error : %f' % (rmse))

print('\nR2 Score : %f' % (r2))

print('\nMean Absolute Percentage Error : %f %%' % (mape))

print('\nR2 Based Prediction Performance : %f %%'
      % (r2 * 100))


# ---------------------------------------------------------
# Actual vs Predicted Price
# ---------------------------------------------------------

result = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': y_pred
})

print('\n==============================================')

print('\nActual vs Predicted Price')

print('\n==============================================')

print(result.head(20))


# ---------------------------------------------------------
# Save Prediction Results
# ---------------------------------------------------------

result.to_csv(
    'Agricultural_Actual_vs_Predicted.csv',
    index=False
)


# ---------------------------------------------------------
# Actual vs Predicted Graph
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

plt.xlabel(
    'Actual Market Price (INR/Quintal)'
)

plt.ylabel(
    'Predicted Market Price (INR/Quintal)'
)

plt.title(
    'Actual vs Predicted Agricultural Market Price'
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle='--'
)

plt.tight_layout()

plt.show()