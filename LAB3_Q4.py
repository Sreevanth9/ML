import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv( "C:/Users/SREEVANTH/Downloads/DCT_withoutduplicate 4.csv")

# Select two classes to work with
selected_classes = [3333, 3334]

# Filter the dataset to include only the selected classes
filtered_data = data[data['LABEL'].isin(selected_classes)]


X = filtered_data.drop(columns=['LABEL'])
y = filtered_data['LABEL']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Check the shapes of the resulting datasets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)