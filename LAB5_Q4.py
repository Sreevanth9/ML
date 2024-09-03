
#A4
from sklearn.cluster import KMeans

X = df.drop(columns=['LABEL'])


kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)


labels = kmeans.labels_
centers = kmeans.cluster_centers_

print("Cluster Centers:\n", centers)
