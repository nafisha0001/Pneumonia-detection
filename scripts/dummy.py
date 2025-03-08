import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA

# Paths
training_data_path = r"C:\Users\Acer\Desktop\X-Ray\data\train\train"

# Image data generator
datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load training data
train_data = datagen.flow_from_directory(
    training_data_path,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Extract features and labels
images, labels = [], []
for _ in range(len(train_data)):
    batch_images, batch_labels = next(train_data)
    images.extend(batch_images)
    labels.extend(batch_labels)

images = np.array(images)
labels = np.array(labels)

# Flatten images for dimensionality reduction
images_flattened = images.reshape(images.shape[0], -1)

# Dimensionality reduction using PCA
pca = PCA(n_components=3)
features_3d = pca.fit_transform(images_flattened)

# Separate features for the two classes
normal_features = features_3d[labels == 0]
pneumonia_features = features_3d[labels == 1]

# Create the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Initial scatter plot
normal_scatter = ax.scatter(
    normal_features[:, 0], normal_features[:, 1], normal_features[:, 2],
    c="blue", label="Normal", s=20
)
pneumonia_scatter = ax.scatter(
    pneumonia_features[:, 0], pneumonia_features[:, 1], pneumonia_features[:, 2],
    c="red", label="Pneumonia", s=20
)

# Axis labels and title
ax.set_title("Animated 3D Scatter Plot")
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
ax.set_zlabel("Component 3")
ax.legend()

# Animation function
def update(frame):
    ax.view_init(elev=20, azim=frame)  # Rotate the view angle
    return normal_scatter, pneumonia_scatter

# Create animation
anim = FuncAnimation(
    fig, update, frames=np.arange(0, 360, 2), interval=50, blit=False
)

# Show the animation
plt.show()

# Optional: Save animation as a video (requires ffmpeg or similar tool)
anim.save('3d_scatter_animation.mp4', writer='ffmpeg', fps=30)
