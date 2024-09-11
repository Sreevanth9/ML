
from sklearn.neural_network import MLPClassifier

# A11: Using MLPClassifier for AND Gate
mlp_and = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', learning_rate_init=0.05, max_iter=1000)
mlp_and.fit(inputs, outputs)
print(f"MLPClassifier Weights for AND gate: {mlp_and.coefs_}")

# A11: Using MLPClassifier for XOR Gate
mlp_xor = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', learning_rate_init=0.05, max_iter=1000)
mlp_xor.fit(inputs_xor, outputs_xor)
print(f"MLPClassifier Weights for XOR gate: {mlp_xor.coefs_}")
MLPClassifier Weights for AND gate: [array([[-4.66439058, -4.70971052],
       [-4.70187354, -4.6722469 ]]), array([[-5.52916835],
       [-4.94161041]])]
MLPClassifier Weights for XOR gate: [array([[-7.21792739, -6.93529282],
       [-4.67000142,  4.70247249]]), array([[-5.92964435],
       [ 6.08290793]])]
C:\Users\NISHANTH\AppData\Roaming\Python\Python312\site-packages\sklearn\neural_network\_multilayer_perceptron.py:1105: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
#A12

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
file_path = "D:/DCT_withoutduplicate 3 (1).csv"
data = pd.read_csv(file_path)

# Assuming the last column is the target and the rest are features
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for MLP)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, activation='relu', solver='adam', random_state=42)

# Train the model
mlp.fit(X_train, y_train)

# Make predictions
y_pred = mlp.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
