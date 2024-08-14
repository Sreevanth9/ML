import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the CSV file
file_path =  "C:/Users/SREEVANTH/Downloads/DCT_withoutduplicate 4.csv"
data = pd.read_csv(file_path)

# Step 2: Preprocess the data (if necessary)

X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # The last column

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the kNN model
neigh = KNeighborsClassifier(n_neighbors=5)  # Default is 5 neighbors
neigh.fit(X_train, y_train)

# Step 5: Evaluate the model using the test set
accuracy = neigh.score(X_test, y_test)
print(f"Accuracy of the kNN model: {accuracy:.2f}")