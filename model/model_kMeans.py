import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, confusion_matrix, accuracy_score
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter

# Paths to datasets
train_dir = r"C:\Users\Acer\Desktop\X-Ray\data\train\train"
val_dir = r"C:\Users\Acer\Desktop\X-Ray\data\val"
test_dir = r"C:\Users\Acer\Desktop\X-Ray\data\test"

# Data preprocessing for training, validation, and test datasets
datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False
)
val_generator = datagen.flow_from_directory(
    val_dir, target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False
)
test_generator = datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False
)

# Function to extract features using a pre-trained model
def extract_features(generator):
    model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    features = model.predict(generator, verbose=1)
    flattened_features = features.reshape(features.shape[0], -1)
    return flattened_features, generator.classes

# Extract features from train, validation, and test datasets
print("Extracting features using VGG16...")
x_train_features, y_train = extract_features(train_generator)
x_val_features, y_val = extract_features(val_generator)
x_test_features, y_test = extract_features(test_generator)

# Combine train, validation, and test data for clustering
x_combined = np.concatenate([x_train_features, x_val_features, x_test_features], axis=0)
y_combined = np.concatenate([y_train, y_val, y_test], axis=0)

# Dimensionality reduction using PCA
print("Applying PCA for dimensionality reduction...")
pca = PCA(n_components=50)  # Reduce to 50 principal components
x_combined_pca = pca.fit_transform(x_combined)

# Silhouette analysis to find the optimal number of clusters
print("Finding optimal number of clusters...")
silhouette_scores = []
cluster_range = range(2, 10)
for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(x_combined_pca)
    silhouette_avg = silhouette_score(x_combined_pca, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"Number of Clusters: {n_clusters}, Silhouette Score: {silhouette_avg:.4f}")

# Plot silhouette scores
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, silhouette_scores, marker='o')
plt.title("Silhouette Scores for Different Numbers of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()

# Apply K-Means with optimal number of clusters (based on silhouette scores)
optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters: {optimal_clusters}")

print("Applying K-Means Clustering with optimal clusters...")
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(x_combined_pca)

# Map clusters to actual labels using majority voting
label_mapping = {}
for cluster in range(optimal_clusters):
    cluster_indices = np.where(cluster_labels == cluster)[0]
    true_labels = y_combined[cluster_indices]
    most_common_label = Counter(true_labels).most_common(1)[0][0]
    label_mapping[cluster] = most_common_label

# Map predicted cluster labels to actual labels
predicted_labels = np.array([label_mapping[cluster] for cluster in cluster_labels])

# Evaluate clustering performance
accuracy = accuracy_score(y_combined, predicted_labels)
silhouette_avg = silhouette_score(x_combined_pca, cluster_labels)

print(f"\nClustering Accuracy: {accuracy * 100:.2f}%")
print(f"Silhouette Score: {silhouette_avg:.4f}")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_combined, predicted_labels))

# Visualize clustering results using PCA (for dimensionality reduction to 2D)
x_combined_2d = PCA(n_components=2).fit_transform(x_combined_pca)
plt.figure(figsize=(8, 6))
for cluster in range(optimal_clusters):
    cluster_points = x_combined_2d[cluster_labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')
plt.title("K-Means Clustering Results (2D PCA Reduced Data)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()
