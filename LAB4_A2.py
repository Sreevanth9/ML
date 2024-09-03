import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Use forward slashes in the file path
s = "C:/Users/SREEVANTH/Downloads/Lab Session Data.xlsx"

# Load the correct sheet name
data = pd.read_excel(s, sheet_name='IRCTC Stock Price')

# Extract relevant columns for price
price = data['Price'].to_numpy()

# Calculate the mean price (as a simple model prediction)
mean_price = np.mean(price)

# Create predictions as the mean price for each entry
predictions = np.full_like(price, mean_price)

# Calculate MSE
mse = mean_squared_error(price, predictions)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate RMSE
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Calculate MAPE
mape = np.mean(np.abs((price - predictions) / price)) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")

# Calculate R² Score
r2 = r2_score(price, predictions)
print(f"R-squared (R²): {r2}")

# Analyze the results
print("\n--- Analysis of the Results ---")
print(f"The MSE is {mse:.2f}, which indicates the average squared difference between the actual prices and the predicted prices.")
print(f"The RMSE is {rmse:.2f}, which gives us an idea of the prediction error in the same units as the target variable (price).")
print(f"The MAPE is {mape:.2f}%, showing the average percentage error between the predicted and actual values.")
print(f"The R² score is {r2:.2f}, representing how well the mean price explains the variance in the actual stock prices. An R² value close to 0 indicates that the mean price is not a good predictor.")