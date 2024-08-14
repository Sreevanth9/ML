import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
file_path = "C:/Users/SREEVANTH/Downloads/DCT_withoutduplicate 4.csv"
df = pd.read_csv(file_path)

# Extracting features and target variables
X = df.iloc[:, :-1]  # All columns except the last one are features
y = df.iloc[:, -1]  # The last column is the target

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# List to store accuracies
accuracies = []

# Loop through k values from 1 to 11
for k in range(1, 12):
    # Initialize the k-NN classifier
    neigh = KNeighborsClassifier(n_neighbors=k)

    # Train the classifier
    neigh.fit(X_train, y_train)

    # Predict the classes for the test set
    y_pred = neigh.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f'Accuracy for k={k}: {accuracy:.4f}')

# Plot the accuracy as a function of k
plt.figure(figsize=(10, 6))
plt.plot(range(1, 12), accuracies, marker='o', linestyle='--', color='b')
plt.title('k-NN Accuracy for Different Values of k')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.xticks(range(1, 12))
plt.grid(True)
plt.show()