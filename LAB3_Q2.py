import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "C:/Users/SREEVANTH/Downloads/DCT_withoutduplicate 4.csv"
data = pd.read_csv(file_path)


feature = data.iloc[:, 0]


hist, bin_edges = np.histogram(feature, bins=10)  # You can adjust the number of bins


plt.hist(feature, bins=10, edgecolor='black')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.title('Histogram of the Chosen Feature')
plt.show()


mean_value = np.mean(feature)
variance_value = np.var(feature)

mean_value, variance_value