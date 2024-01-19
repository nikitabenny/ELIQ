import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from skimage import io, color

def calculate_snr(image):
    # Function to calculate Signal-to-Noise Ratio (SNR) of an MRI image
    # Assuming image is a 2D numpy array representing an MRI image
    signal = np.mean(image)
    noise = np.std(image)
    snr = signal / noise
    return snr

def preprocess_images(image_paths):
    # Function to preprocess MRI images and extract SNR values
    # image_paths: List of file paths of MRI images

    # Read each image and convert to grayscale
    images = [io.imread(path, as_gray=True) for path in image_paths]

    # Calculate SNR for each image
    snr_values = [calculate_snr(image) for image in images]

    # Convert the list of SNR values into a numpy array
    return np.array(snr_values).reshape(-1, 1)

def kmeans_clustering(image_paths, num_clusters):
    # Function to perform K-means clustering on MRI images based on SNR values
    # image_paths: List of file paths of MRI images
    # num_clusters: Number of clusters to form

    # Preprocess images and extract SNR values
    snr_data = preprocess_images(image_paths)

    # Applying K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(snr_data)

    # Assigning cluster labels to each image
    cluster_labels = kmeans.labels_

    return cluster_labels

def plot_clusters(snr_data, cluster_labels):
    # Function to plot the clustering results
    # snr_data: Array of SNR values
    # cluster_labels: Cluster assignments for each data point

    # Apply PCA for visualization in 2D
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(snr_data)

    # Plot the clusters
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='viridis', edgecolor='k')
    plt.title('K-means Clustering of MRI Images based on SNR')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

# Example usage:
image_paths = ["path/to/image1.jpg", "path/to/image2.jpg", "path/to/image3.jpg"]
num_clusters = 3  # Set the desired number of clusters

# Perform K-means clustering
cluster_labels = kmeans_clustering(image_paths, num_clusters)

# Plot the clustering results
snr_data = preprocess_images(image_paths)
plot_clusters(snr_data, cluster_labels)
