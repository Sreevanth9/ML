import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv( "C:/Users/SREEVANTH/Downloads/DCT_withoutduplicate 4.csv")


selected_classes = [3333, 3334]


filtered_data = data[data['LABEL'].isin(selected_classes)]


X = filtered_data.drop(columns=['LABEL'])
y = filtered_data['LABEL']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")