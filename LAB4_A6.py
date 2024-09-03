import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = "D:/DCT_withoutduplicate 3 (1).csv"
data = pd.read_csv(file_path)

# Assume the last column is the target and the rest are features
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
knn = KNeighborsClassifier()

# Define the parameter grid for GridSearchCV or RandomizedSearchCV
param_grid = {'n_neighbors': list(range(1, 31))}

# Option 1: Use GridSearchCV
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_k_grid = grid_search.best_params_['n_neighbors']

# Option 2: Use RandomizedSearchCV
random_search = RandomizedSearchCV(knn, param_distributions=param_grid, n_iter=10, cv=5, scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)
best_k_random = random_search.best_params_['n_neighbors']

# Evaluate the best models on the test set
knn_best_grid = KNeighborsClassifier(n_neighbors=best_k_grid)
knn_best_grid.fit(X_train, y_train)
y_pred_grid = knn_best_grid.predict(X_test)
accuracy_grid = accuracy_score(y_test, y_pred_grid)

knn_best_random = KNeighborsClassifier(n_neighbors=best_k_random)
knn_best_random.fit(X_train, y_train)
y_pred_random = knn_best_random.predict(X_test)
accuracy_random = accuracy_score(y_test, y_pred_random)

print(f"Best k value from GridSearchCV: {best_k_grid} with accuracy: {accuracy_grid:.4f}")
print(f"Best k value from RandomizedSearchCV: {best_k_random} with accuracy: {accuracy_random:.4f}")