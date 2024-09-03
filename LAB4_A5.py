### A5 ###
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Prepare the training data
data = {
    'Candies': [20, 16, 27, 19, 24, 22, 15, 18, 21, 16],
    'Mangoes': [6, 3, 6, 1, 4, 1, 4, 4, 1, 2],
    'Milk Packets': [2, 6, 2, 2, 2, 5, 2, 2, 4, 4],
    'Payment': [386, 289, 393, 110, 280, 167, 271, 274, 148, 198],
    'ANV': [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]  # Assuming ANV is a binary classification column
}

df = pd.DataFrame(data)

# Define features (X) and target (y)
X = df[['Candies', 'Mangoes']].values
y = df['ANV'].values  # Assuming 'ANV' is the target variable for classification

# Standardizing the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# List of k values to evaluate
k_values = [1, 3, 5, 7, 9]

# Loop through each k value, train the model, and calculate metrics
results = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    y_pred = knn.predict(X)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    results.append({
        'k': k,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })

# Print the results
print("k | Accuracy | Precision | Recall | F1 Score")
print("--|----------|-----------|--------|----------")
for result in results:
    print(
        f"{result['k']} | {result['accuracy']:.2f}     | {result['precision']:.2f}     | {result['recall']:.2f}  | {result['f1_score']:.2f}")