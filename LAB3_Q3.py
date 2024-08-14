import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski

# Load the dataset
file_path ="C:/Users/SREEVANTH/Downloads/DCT_withoutduplicate 4.csv"
data = pd.read_csv(file_path)


feature_vector_1 = data.iloc[0, :-1]  # Excluding the label
feature_vector_2 = data.iloc[1, :-1]  # Excluding the label


r_values = np.arange(1, 11)
minkowski_distances = [minkowski(feature_vector_1, feature_vector_2, r) for r in r_values]

# Plot the distances
plt.plot(r_values, minkowski_distances, marker='o')
plt.xlabel('r (Minkowski Distance Parameter)')
plt.ylabel('Minkowski Distance')
plt.title('Minkowski Distance between Two Feature Vectors')
plt.grid(True)
plt.show()

# Output the distances
minkowski_distances