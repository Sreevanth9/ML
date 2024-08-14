import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path =  "C:/Users/SREEVANTH/Downloads/DCT_withoutduplicate 4.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
print(df.head())

# Assuming the last column is the target variable and the rest are features
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features (k-NN works better with scaled data)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the k-NN classifier with k=3 (as an example)
neigh = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
neigh.fit(X_train, y_train)

# Perform prediction on a sample test vector
test_vect = X_test[0].reshape(1, -1)  # Selecting the first test vector and reshaping it
predicted_class = neigh.predict(test_vect)

# Output the predicted class
print(f'Predicted class for the test vector: {predicted_class[0]}')