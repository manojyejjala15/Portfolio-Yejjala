# Import libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans


# Load Iris dataset
df = pd.read_csv("E:/workspace/Manoj_Portfolio/data_sets/iris.csv")

# Check missing values
print("Missing values:")
print(df.isna().sum())

# Drop 'Species' column
iris_clustering = df.drop(columns=["Species"])

# Select SepalLengthCm (0) and PetalLengthCm (2) for clustering
X = iris_clustering.iloc[:, [0, 2]].values

# Elbow method to find optimal k
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title("Elbow Point Graph")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# K-means with k=3
kmeans = KMeans(n_clusters=3, init="k-means++", random_state=0)

# Predict cluster labels
y = kmeans.fit_predict(X)
print("Cluster labels:")
print(y)

# Scatter plot of clusters
plt.scatter(X[y == 0, 0], X[y == 0, 1], s=50, c="red", label="Cluster 1")
plt.scatter(X[y == 1, 0], X[y == 1, 1], s=50, c="blue", label="Cluster 2")
plt.scatter(X[y == 2, 0], X[y == 2, 1], s=50, c="green", label="Cluster 3")
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=100,
    c="cyan",
    label="Centroids",
)
plt.title("Iris Flower Clusters")
plt.xlabel("Sepal Length in cm")
plt.ylabel("Petal Length in cm")
plt.legend()
plt.show()
