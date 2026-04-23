from sklearn.cluster import KMeans

# Train k-means on your vectors
nlist = 100  # Number of clusters
kmeans = KMeans(n_clusters=nlist, random_state=42)
kmeans.fit(all_vectors)

# centroids = kmeans.cluster_centers_
# labels = kmeans.labels_ (which cluster each vector belongs to)