### A4 ###

from sklearn.neighbors import KNeighborsClassifier


X1_test = np.random.randint(1, 30)
X2_test = np.random.randint(1, 10)
X3_test = np.random.randint(1, 5)
Y_test = calculate_total_bill(X1_test, X2_test, X3_test)
test_point = np.array([[X1_test, X2_test, X3_test]])

k = 3
knn = KNeighborsClassifier(n_neighbors=k)
X_train_full = np.column_stack((X1_train, X2_train, X3_train))
knn.fit(X_train_full, y_train)

y_pred = knn.predict(test_point)

plt.figure(figsize=(8, 6))
plt.scatter(X1_train, Y_train, c=colors, marker='o', edgecolors='k', label=['Poor (Blue)', 'Rich (Red)'])
plt.scatter(X1_test, Y_test, c='green', marker='x', label='Test Point')
plt.title(f'Training Data with Test Point Prediction (k={k})')
plt.xlabel('Candies (#)')
plt.ylabel('Total Bill (Rs Y)')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

if y_pred == 0:
    print(f'Test point is classified as Poor (Blue)')
else:
    print(f'Test point is classified as Rich (Red)')