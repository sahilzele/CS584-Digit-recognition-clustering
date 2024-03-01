import numpy as np
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import zscore
from sklearn.decomposition import PCA

import pandas as pd


# Function to initialize centroids using k-means++
def initialize_centroids(data_matrix, num_clusters):
    num_samples, _ = data_matrix.shape
    centroids = [data_matrix[np.random.choice(num_samples)]]
    for _ in range(1, num_clusters):
        distances = np.linalg.norm(data_matrix - np.array(centroids)[:, np.newaxis], axis=2)
        min_distances = np.min(distances, axis=0)
        next_centroid = data_matrix[np.argmax(min_distances)]
        centroids.append(next_centroid)
    return np.array(centroids)


# Function to assign each data point to the nearest centroid
def assign_to_nearest_centroid(data_matrix, cluster_centroids):
    distances = np.linalg.norm(data_matrix[:, np.newaxis] - cluster_centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    return labels


# Function to update centroids based on assigned data points
def update_centroids(data_matrix, data_labels, num_clusters):
    new_centroids = np.array([data_matrix[data_labels == k].mean(axis=0) for k in range(num_clusters)])
    return new_centroids


# Function to calculate silhouette score
def calculate_silhouette(data_matrix, data_labels):
    return silhouette_score(data_matrix, data_labels)


# Custom K-means clustering algorithm
def my_kmeans(data_matrix, num_clusters, max_iterations=300, num_runs=20):
    best_centroids = None
    best_labels = None
    best_silhouette_score = -1

    for _ in range(num_runs):
        centroids = initialize_centroids(data_matrix, num_clusters)
        centroids_previous = np.zeros_like(centroids)

        for _ in range(max_iterations):
            labels = assign_to_nearest_centroid(data_matrix, centroids)
            centroids_previous[:] = centroids
            centroids = update_centroids(data_matrix, labels, num_clusters)

            if np.all(centroids == centroids_previous):
                break

        silhouette = calculate_silhouette(data_matrix, labels)

        if silhouette > best_silhouette_score:
            best_silhouette_score = silhouette
            best_centroids = centroids
            best_labels = labels

    return best_centroids, best_labels, best_silhouette_score


# Load the digits dataset
file_path = "/Users/sahil/Documents/CS584/Ass3/Test.txt"
test_dataframe = pd.read_csv(file_path, engine='python', delimiter='    ', header=None)
test_dataframe.columns = ['pixels']
data_matrix = test_dataframe['pixels'].str.split(',', expand=True)

# Use UMAP for dimensionality reduction
from umap import UMAP
umap_model = UMAP(n_components=2)
data_umap = umap_model.fit_transform(data_matrix)

# Apply Gaussian blur to UMAP components
blurred_data_umap = data_umap
#gaussian_filter(data_umap, sigma=0.2)

# Apply custom K-means clustering
number_of_clusters = 10
best_cluster_centroids, best_data_labels, best_silhouette_score = my_kmeans(blurred_data_umap, number_of_clusters)


# Visualization of the clusters using UMAP components
plt.scatter(blurred_data_umap[:, 0], blurred_data_umap[:, 1], c=best_data_labels, cmap='viridis', edgecolors='k', s=50)
plt.scatter(best_cluster_centroids[:, 0], best_cluster_centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title(f'Custom K-means Clustering for Digit Recognition\nBest Silhouette Score: {best_silhouette_score:.2f}')
plt.xlabel('Blurred UMAP Component 1')
plt.ylabel('Blurred UMAP Component 2')
plt.legend()
plt.show()

print(f"Best Silhouette Score: {best_silhouette_score:.2f}")

# Save the labels to a file
output_file_path = '/Users/sahil/Documents/CS584/format.dat'
with open(output_file_path, 'w') as file:
    for label in best_data_labels:
        file.write(f'{label}\n')
