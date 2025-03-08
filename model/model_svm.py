import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

# Paths to datasets
train_dir = r"C:\Users\Acer\Desktop\X-Ray\data\train\train"
val_dir = r"C:\Users\Acer\Desktop\X-Ray\data\val"
test_dir = r"C:\Users\Acer\Desktop\X-Ray\data\test"

# Labels for classification
labels = ['NORMAL', 'PNEUMONIA']

# Count images in training dataset
train_normal = os.path.join(train_dir, "NORMAL")
train_pneumonia = os.path.join(train_dir, "PNEUMONIA")
print("Training Dataset:")
print("Number of NORMAL images:", len(os.listdir(train_normal)))
print("Number of PNEUMONIA images:", len(os.listdir(train_pneumonia)))

# Visualizing a sample image
sample_normal = os.listdir(train_normal)[0]
sample_pneumonia = os.listdir(train_pneumonia)[0]

normal_image_path = os.path.join(train_normal, sample_normal)
pneumonia_image_path = os.path.join(train_pneumonia, sample_pneumonia)

print("Normal Example:")
normal_img = cv2.imread(normal_image_path)
plt.imshow(cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB))
plt.show()

print("Pneumonia Example:")
pneumonia_img = cv2.imread(pneumonia_image_path)
plt.imshow(cv2.cvtColor(pneumonia_img, cv2.COLOR_BGR2RGB))
plt.show()

# Function to load images and labels
def load_data(directory, labels, img_size=(256, 180)):
    data = []
    target = []
    for label in labels:
        path = os.path.join(directory, label)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img_resized = cv2.resize(img, img_size)
                data.append(img_resized.flatten())  # Flatten the image
                target.append(labels.index(label))  # Label index
    return np.array(data), np.array(target)

# Load datasets
print("Loading training dataset...")
x_train, y_train = load_data(train_dir, labels)

print("Loading validation dataset...")
x_val, y_val = load_data(val_dir, labels)

print("Loading testing dataset...")
x_test, y_test = load_data(test_dir, labels)

# SVM Model Training
print("Training SVM model...")
model = SVC(kernel='poly', max_iter=10000, class_weight='balanced') 
model.fit(x_train, y_train)

# Evaluate on training dataset
y_train_pred = model.predict(x_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

# Evaluate on validation dataset
y_val_pred = model.predict(x_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Evaluate on testing dataset
y_test_pred = model.predict(x_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Classification report and confusion matrix
print("\nClassification Report (Test Dataset):")
print(classification_report(y_test, y_test_pred, target_names=labels))

print("\nConfusion Matrix (Test Dataset):")
print(confusion_matrix(y_test, y_test_pred))

# Perform 5-Fold Cross-Validation on training data
print("\nPerforming Cross-Validation on Training Dataset...")
cv_scores = cross_val_score(model, x_train, y_train, cv=5)
print(f"Cross-Validation Accuracy: {np.mean(cv_scores) * 100:.2f}%")
