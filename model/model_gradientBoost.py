import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, log_loss, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to datasets
train_dir = r"C:\Users\Acer\Desktop\X-Ray\data\train\train"
val_dir = r"C:\Users\Acer\Desktop\X-Ray\data\val"
test_dir = r"C:\Users\Acer\Desktop\X-Ray\data\test"

# Labels for classification
labels = ['NORMAL', 'PNEUMONIA']

# Data augmentation for the training dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data preprocessing for validation and test datasets (only rescaling)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training dataset with augmentation
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),  
    batch_size=32,
    class_mode='binary',  
    shuffle=True          
)

# Load validation dataset
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Load test dataset
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Convert data generators to NumPy arrays for sklearn compatibility
def generator_to_numpy(generator):
    images, labels = [], []
    for _ in range(len(generator)):
        img_batch, label_batch = next(generator)  
        images.extend(img_batch)
        labels.extend(label_batch)
    return np.array(images), np.array(labels)

# Extract data from generators
print("Converting training data to NumPy arrays...")
x_train, y_train = generator_to_numpy(train_generator)

print("Converting validation data to NumPy arrays...")
x_val, y_val = generator_to_numpy(val_generator)

print("Converting test data to NumPy arrays...")
x_test, y_test = generator_to_numpy(test_generator)

# Flatten the images for Gradient Boosting compatibility
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_val_flat = x_val.reshape(x_val.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# Train Gradient Boosting Classifier
print("Training Gradient Boosting Classifier...")
model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,           
    max_depth=8,                 
    min_samples_split=20,        
    min_samples_leaf=15,         
    subsample=0.8,               
    random_state=42             
)

model.fit(x_train_flat, y_train)
print("Model trained successfully!")

# Evaluate on training dataset
y_train_pred = model.predict(x_train_flat)
y_train_prob = model.predict_proba(x_train_flat)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_loss = log_loss(y_train, y_train_prob)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Training Recall: {train_recall * 100:.2f}%")
print(f"Training Loss (Log Loss): {train_loss:.4f}")

# Evaluate on validation dataset
y_val_pred = model.predict(x_val_flat)
y_val_prob = model.predict_proba(x_val_flat)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_recall = recall_score(y_val, y_val_pred)
val_loss = log_loss(y_val, y_val_prob)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Validation Recall: {val_recall * 100:.2f}%")
print(f"Validation Loss (Log Loss): {val_loss:.4f}")

# Evaluate on testing dataset
y_test_pred = model.predict(x_test_flat)
y_test_prob = model.predict_proba(x_test_flat)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_loss = log_loss(y_test, y_test_prob)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Recall: {test_recall * 100:.2f}%")
print(f"Test Loss (Log Loss): {test_loss:.4f}")

# Classification report and confusion matrix for test dataset
print("\nClassification Report (Test Dataset):")
print(classification_report(y_test, y_test_pred, target_names=labels))

print("\nConfusion Matrix (Test Dataset):")
print(confusion_matrix(y_test, y_test_pred))