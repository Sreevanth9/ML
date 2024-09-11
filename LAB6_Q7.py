
import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Load the provided dataset into a DataFrame
data = {
    'Customer': ['C_1', 'C_2', 'C_3', 'C_4', 'C_5', 'C_6', 'C_7', 'C_8', 'C_9', 'C_10'],
    'Candies': [20, 16, 27, 19, 24, 22, 15, 18, 21, 16],
    'Mangoes': [6, 3, 6, 1, 4, 1, 4, 4, 1, 2],
    'Milk Packets': [2, 6, 2, 2, 2, 5, 2, 2, 4, 4],
    'Payment': [386, 289, 393, 110, 280, 167, 271, 274, 148, 198],
    'High Value Tx?': ['Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No']
}

# Create a DataFrame from the dataset
df = pd.DataFrame(data)

# Convert 'High Value Tx?' to binary labels: 'Yes' -> 1, 'No' -> 0
df['High Value Tx?'] = df['High Value Tx?'].apply(lambda x: 1 if x == 'Yes' else 0)

# Features and target variable
features = df[['Candies', 'Mangoes', 'Milk Packets']].values
target = df['High Value Tx?'].values

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Function to calculate weights using the pseudo-inverse method
def pseudo_inverse_method(X, y):
    X_bias = np.hstack([np.ones((X.shape[0], 1)), X])  # Add a bias term
    weights_pseudo_inverse = np.linalg.pinv(X_bias).dot(y)
    return weights_pseudo_inverse

# Calculate weights using pseudo-inverse method
weights_pseudo_inverse = pseudo_inverse_method(features_scaled, target)
print(f"Weights obtained using pseudo-inverse method: {weights_pseudo_inverse}")
# Using Perceptron model for comparison
perceptron_model = Perceptron(max_iter=1000, random_state=42)
perceptron_model.fit(features_scaled, target)
perceptron_weights = np.hstack([perceptron_model.intercept_, perceptron_model.coef_[0]])
print(f"Perceptron Weights: {perceptron_weights}")

# Predict and evaluate using perceptron model
predictions = perceptron_model.predict(features_scaled)
print("\nConfusion Matrix (Perceptron):")
print(confusion_matrix(target, predictions))
print("\nClassification Report (Perceptron):")
print(classification_report(target, predictions))
