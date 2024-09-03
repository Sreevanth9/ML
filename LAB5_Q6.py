# A6

silhouette_scores = []
ch_scores = []
db_scores = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X)
    labels = kmeans.labels_

    silhouette_scores.append(silhouette_score(X, labels))
    ch_scores.append(calinski_harabasz_score(X, labels))
    db_scores.append(davies_bouldin_score(X, labels))

# Plotting the scores against k values
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Score vs k')

plt.subplot(1, 3, 2)
plt.plot(k_values, ch_scores, marker='o')
plt.title('CH Score vs k')

plt.subplot(1, 3, 3)
plt.plot(k_values, db_scores, marker='o')
plt.title('DB Score vs k')

plt.tight_layout()
plt.show()