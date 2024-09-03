#A2
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np


mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)


rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)


mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
mape_test = mean_absolute_percentage_error(y_test, y_test_pred)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Train MSE: {mse_train}, Test MSE: {mse_test}")
print(f"Train RMSE: {rmse_train}, Test RMSE: {rmse_test}")
print(f"Train MAPE: {mape_train}, Test MAPE: {mape_test}")
print(f"Train R²: {r2_train}, Test R²: {r2_test}")