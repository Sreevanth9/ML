import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# A6: Load the provided dataset
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
X = df[['Candies', 'Mangoes', 'Milk Packets']]
y = df['High Value Tx?']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', solver='adam', learning_rate_init=0.01, max_iter=1000, random_state=42)

# Train the model
mlp.fit(X_train_scaled, y_train)

# Predict on the test data
y_pred = mlp.predict(X_test_scaled)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display the model weights
print("\nMLPClassifier Weights:")
for i, layer_weights in enumerate(mlp.coefs_):
    print(f"Layer {i + 1} weights:")
    print(layer_weights)
