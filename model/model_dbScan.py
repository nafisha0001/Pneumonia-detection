import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
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
pca = PCA(n_components=50)
x_combined_pca = pca.fit_transform(x_combined)

# Apply DBSCAN for clustering
print("Applying DBSCAN...")
dbscan = DBSCAN(eps=0.5, min_samples=5)  # You can adjust `eps` and `min_samples`
cluster_labels = dbscan.fit_predict(x_combined_pca)

# Evaluate clustering performance
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)  # Exclude noise
print(f"Number of Clusters (excluding noise): {n_clusters}")

# Evaluate clustering performance
silhouette_avg = silhouette_score(x_combined_pca, cluster_labels) if n_clusters > 1 else -1
print(f"Silhouette Score: {silhouette_avg:.4f}")

# Confusion Matrix
# For DBSCAN, we need to handle noise points (labeled -1)
valid_labels = cluster_labels != -1  # Get only valid (non-noise) data points
valid_y_combined = y_combined[valid_labels]
valid_cluster_labels = cluster_labels[valid_labels]

# Map clusters to actual labels using majority voting
label_mapping = {}
for cluster in set(valid_cluster_labels):
    cluster_indices = np.where(valid_cluster_labels == cluster)[0]
    true_labels = valid_y_combined[cluster_indices]
    most_common_label = Counter(true_labels).most_common(1)[0][0]
    label_mapping[cluster] = most_common_label

# Map predicted cluster labels to actual labels
predicted_labels = np.array([label_mapping.get(cluster, -1) for cluster in cluster_labels])

# Evaluate accuracy and confusion matrix
accuracy = accuracy_score(y_combined, predicted_labels)
print(f"Clustering Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_combined, predicted_labels))

# Visualize clustering results using PCA (for dimensionality reduction to 2D)
x_combined_2d = PCA(n_components=2).fit_transform(x_combined_pca)
plt.figure(figsize=(8, 6))
unique_labels = set(cluster_labels)
for label in unique_labels:
    cluster_points = x_combined_2d[cluster_labels == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}' if label != -1 else 'Noise', marker='x' if label == -1 else 'o')

plt.title("DBSCAN Clustering Results (2D PCA Reduced Data)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()
