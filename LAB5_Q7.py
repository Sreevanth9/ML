#A7
distortions = []
k_values = range(2, 20)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X)
    distortions.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(k_values, distortions, marker='o')
plt.title('Elbow Plot')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Distortion')
plt.show()