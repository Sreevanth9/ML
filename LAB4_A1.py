import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset
file_path = "C:/Users/SREEVANTH/Downloads/DCT_withoutduplicate 4.csv"
data = pd.read_csv(file_path)

# Display first few rows of the data
print(data.head())

# Assuming the last column is the target variable
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data (if necessary)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the classifier
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train, y_train)

# Predictions on the training data
y_train_pred = model.predict(X_train)

# Predictions on the test data
y_test_pred = model.predict(X_test)

# Confusion matrix for training data
conf_matrix_train = confusion_matrix(y_train, y_train_pred)
print("Confusion Matrix for Training Data:\n", conf_matrix_train)

# Confusion matrix for test data
conf_matrix_test = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix for Test Data:\n", conf_matrix_test)

# Classification report for training data (includes precision, recall, F1-Score)
print("\nClassification Report for Training Data:\n", classification_report(y_train, y_train_pred))

# Classification report for test data (includes precision, recall, F1-Score)
print("\nClassification Report for Test Data:\n", classification_report(y_test, y_test_pred))

# Observations based on confusion matrix and classification report
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print("\nTraining Accuracy: {:.2f}%".format(train_accuracy * 100))
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))

if train_accuracy > test_accuracy:
    print("\nThe model might be overfitting.")
elif train_accuracy < test_accuracy:
    print("\nThe model might be underfitting.")
else:
    print("\nThe model has a good fit.")