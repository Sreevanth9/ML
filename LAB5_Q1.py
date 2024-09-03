#A1
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


X = df[['0']]
y = df['LABEL']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the linear regression model
reg = LinearRegression().fit(X_train, y_train)

# Making predictions
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)