import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset
file_path = "C:/Users/SREEVANTH/Downloads/DCT_withoutduplicate 4.csv"
df = pd.read_csv(file_path)

# Extracting features and target variables
X = df.iloc[:, :-1]  # All columns except the last one are features
y = df.iloc[:, -1]   # The last column is the target

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the k-NN classifier with k=3
neigh = KNeighborsClassifier(n_neighbors=3)

# Train the classifier on the training data
neigh.fit(X_train, y_train)

# Predictions on training and test data
y_train_pred = neigh.predict(X_train)
y_test_pred = neigh.predict(X_test)

# Confusion Matrix for training data
conf_matrix_train = confusion_matrix(y_train, y_train_pred)
print('Confusion Matrix - Training Data:')
print(conf_matrix_train)

# Confusion Matrix for test data
conf_matrix_test = confusion_matrix(y_test, y_test_pred)
print('\nConfusion Matrix - Test Data:')
print(conf_matrix_test)

# Classification Report for training data with zero_division=1 to handle undefined metrics
class_report_train = classification_report(y_train, y_train_pred, zero_division=1)
print('\nClassification Report - Training Data:')
print(class_report_train)

# Classification Report for test data with zero_division=1 to handle undefined metrics
class_report_test = classification_report(y_test, y_test_pred, zero_division=1)
print('\nClassification Report - Test Data:')
print(class_report_test)