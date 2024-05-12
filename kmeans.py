import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Visualize the data
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Sample Data')
plt.show()

def kmeans_with_plot(X, k, init='random'):
    # Initialize KMeans object
    kmeans = KMeans(n_clusters=k, init=init, random_state=0)
    
    # Fit KMeans model to data
    kmeans.fit(X)
    
    # Plot MSE after each iteration
    plt.plot(np.arange(1, len(kmeans.inertia_) + 1), kmeans.inertia_, marker='o')
    plt.xlabel('Number of iterations')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title(f'K-means with k={k} and init={init}')
    plt.show()

# Test the function with different parameters
k_values = [2, 3, 4, 5]
init_methods = ['random', 'k-means++']

for k in k_values:
    for init_method in init_methods:
        kmeans_with_plot(X, k, init_method)
