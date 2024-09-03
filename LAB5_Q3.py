#A3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np




target_column = 'LABEL'

features = ['0', '1', '2']


X = df[features]
y = df[target_column]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


reg = LinearRegression().fit(X_train, y_train)


y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)


mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# Calculate Metrics for Test Set
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
mape_test = mean_absolute_percentage_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

# Print the metrics
print(f"Train MSE: {mse_train}, Test MSE: {mse_test}")
print(f"Train RMSE: {rmse_train}, Test RMSE: {rmse_test}")
print(f"Train MAPE: {mape_train}, Test MAPE: {mape_test}")
print(f"Train R²: {r2_train}, Test R²: {r2_test}")