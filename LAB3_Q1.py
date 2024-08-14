import numpy as np
import pandas as pd

# Load the dataset
file_path = "C:/Users/SREEVANTH/Downloads/DCT_withoutduplicate 4.csv"
data = pd.read_csv(file_path)


print(data.head())



class_labels = data.iloc[:, -1].unique()


class_centroids = {}
class_spreads = {}

for label in class_labels:
    class_data = data[data.iloc[:, -1] == label].iloc[:, :-1]
    class_centroids[label] = np.mean(class_data, axis=0)
    class_spreads[label] = np.std(class_data, axis=0)



centroid1 = class_centroids[class_labels[0]]
centroid2 = class_centroids[class_labels[1]]
distance_between_centroids = np.linalg.norm(centroid1 - centroid2)

print(f"Centroid of Class {class_labels[0]}: \n{centroid1}")
print(f"Centroid of Class {class_labels[1]}: \n{centroid2}")
print(f"Spread of Class {class_labels[0]}: \n{class_spreads[class_labels[0]]}")
print(f"Spread of Class {class_labels[1]}: \n{class_spreads[class_labels[1]]}")
print(f"Euclidean distance between centroids of Class {class_labels[0]} and Class {class_labels[1]}: {distance_between_centroids}")