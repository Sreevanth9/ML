### A3 ###

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Set random seed for reproducibility
np.random.seed(0)

# Given unit prices
price_candies = 55
price_mangoes = 1
price_milk_packets = 18

# Function to calculate total bill and classify as 'rich' or 'poor'
def calculate_total_bill(X1, X2, X3):
    return X1 * price_candies + X2 * price_mangoes + X3 * price_milk_packets

# Task A3: Generate Training Data
n_points = 20
X1_train = np.random.randint(1, 30, size=n_points)  # Candies (#)
X2_train = np.random.randint(1, 10, size=n_points)  # Mangoes (Kg)
X3_train = np.random.randint(1, 5, size=n_points)   # Milk Packets (#)
Y_train = calculate_total_bill(X1_train, X2_train, X3_train)
y_train = np.where(Y_train > 200, 1, 0)  # 1 for 'rich' (Y > 200), 0 for 'poor' (Y <= 200)

# Plot training data
colors = np.where(y_train == 0, 'blue', 'red')
plt.figure(figsize=(8, 6))
plt.scatter(X1_train, Y_train, c=colors, marker='o', edgecolors='k', label=['Poor (Blue)', 'Rich (Red)'])
plt.title('Training Data')
plt.xlabel('Candies (#)')
plt.ylabel('Total Bill (Rs Y)')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()