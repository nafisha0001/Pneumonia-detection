import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import zipfile
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model.model_densenet import create_densenet121, create_densenet169
from model.model_resnet import create_resnet50
from model.model_mobilenet import create_mobilenet
from model.model_inception import create_inceptionv3
from model.model_vgg import create_vgg16
from model.model_simple_cnn import create_simple_cnn

tf.config.run_functions_eagerly(True)

# zip_filename = r'C:\Users\Acer\Desktop\X-Ray\data\train.zip'  
extract_folder = r'C:\Users\Acer\Desktop\X-Ray\data\train'
training_data_path = os.path.join(extract_folder, 'train')  
validation_data_path = r'C:\Users\Acer\Desktop\X-Ray\data\val'
test_data_path = r'C:\Users\Acer\Desktop\X-Ray\data\test'
saved_model_path = r'C:\Users\Acer\Desktop\Pneumonia\densenet121_model.h5'


datagen = ImageDataGenerator(rescale=1./255)

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

train_data = train_datagen.flow_from_directory(
    training_data_path,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    validation_data_path,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

test_data = datagen.flow_from_directory(
    test_data_path,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

model = tf.keras.models.load_model(saved_model_path)

# Checking if the model is already trained (if having just one model)
# if os.path.exists(saved_model_path):
#     print(f"Loading pre-trained model from {saved_model_path}")
#     model = tf.keras.models.load_model(saved_model_path)
# else:
#     print("No pre-trained model found. Creating and training a new model.")
#     model = create_densenet121()
  

# Train the model
# history = model.fit(
#     train_data, 
#     steps_per_epoch=train_data.samples // train_data.batch_size,
#     epochs=10, 
#     validation_data=val_data, 
#     validation_steps=val_data.samples // val_data.batch_size,
#     verbose=1
# )

# Evaluate on train, validation, and test datasets
# train_results = model.evaluate(train_data, steps=train_data.samples // train_data.batch_size, verbose=1)
# val_results = model.evaluate(val_data, steps=val_data.samples // val_data.batch_size, verbose=1)
test_results = model.evaluate(test_data, steps=test_data.samples // test_data.batch_size, verbose=1)

# print(f"Training Loss: {train_results[0]:.2f}, Accuracy: {train_results[1]:.2f}, Recall: {train_results[2]:.2f}")
# print(f"Validation Loss: {val_results[0]:.2f}, Accuracy: {val_results[1]:.2f}, Recall: {val_results[2]:.2f}")
# print(f"Test Loss: {test_results[0]:.2f}, Accuracy: {test_results[1]:.2f}, Recall: {test_results[2]:.2f}")










# import os
# import numpy as np
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from imblearn.over_sampling import SMOTE  
# from sklearn.utils import shuffle
# from tensorflow import keras
# import tensorflow as tf

# # Paths to your local dataset
# train_dir = r'C:\Users\Acer\Desktop\X-Ray\data\train\train'
# test_dir = r'C:\Users\Acer\Desktop\X-Ray\data\test'
# val_dir = r'C:\Users\Acer\Desktop\X-Ray\data\val'

# # Parameters
# img_dim = (180, 180)
# batch_size = 32
# epochs = 10

# # Load Data and Preprocess
# def load_data(data_dir, img_dim):
#     X = []
#     y = []
#     categories = ['NORMAL', 'PNEUMONIA']  # Ensure consistent ordering
    
#     for label, category in enumerate(categories):
#         folder_path = os.path.join(data_dir, category)
#         for img_name in os.listdir(folder_path):
#             img_path = os.path.join(folder_path, img_name)
#             try:
#                 img = tf.keras.utils.load_img(img_path, target_size=img_dim)
#                 img_array = tf.keras.utils.img_to_array(img)
#                 X.append(img_array)
#                 y.append(label)
#             except Exception as e:
#                 print(f"Error loading image {img_path}: {e}")
    
#     return np.array(X), np.array(y)

# # Load training data
# X_train, y_train = load_data(train_dir, img_dim)

# # Normalize images
# X_train = X_train / 255.0

# # Separate Normal and Pneumonia Instances
# X_normal = X_train[y_train == 0]  
# X_pneumonia = X_train[y_train == 1]  
# y_normal = y_train[y_train == 0]  
# y_pneumonia = y_train[y_train == 1]  

# # Flatten Normal images for SMOTE
# X_normal_flat = X_normal.reshape(X_normal.shape[0], -1)

# # Apply SMOTE to Normal instances
# smote = SMOTE()
# X_resampled_flat, y_resampled = smote.fit_resample(X_normal_flat, y_normal)

# # Reshape back to image dimensions
# X_resampled = X_resampled_flat.reshape(-1, img_dim[0], img_dim[1], 3)

# # Combine resampled Normal instances with original Pneumonia instances
# X_train_balanced = np.concatenate([X_resampled, X_pneumonia], axis=0)
# y_train_balanced = np.concatenate([y_resampled, y_pneumonia], axis=0)

# # Shuffle the balanced dataset
# X_train_balanced, y_train_balanced = shuffle(X_train_balanced, y_train_balanced, random_state=42)

# # Convert labels to categorical
# y_train_balanced = keras.utils.to_categorical(y_train_balanced, num_classes=2)

# # Load validation data
# X_val, y_val = load_data(val_dir, img_dim)
# X_val = X_val / 255.0  
# y_val = keras.utils.to_categorical(y_val, num_classes=2)

# # Build the CNN Model
# cnn = keras.models.Sequential([
#     keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_dim[0], img_dim[1], 3)),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Conv2D(128, (3, 3), activation='relu'),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Flatten(),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(64, activation='relu'),
#     keras.layers.Dense(2, activation='softmax')  # 2 classes: Pneumonia and Normal
# ])

# # Compile the Model
# cnn.compile(optimizer='adam',
#             loss='categorical_crossentropy',
#             metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

# # Train the Model
# history = cnn.fit(
#     X_train_balanced, y_train_balanced,
#     validation_data=(X_val, y_val),
#     epochs=epochs,
#     batch_size=batch_size
# )

# # Load test data
# X_test, y_test = load_data(test_dir, img_dim)
# X_test = X_test / 255.0  # Normalize
# y_test = keras.utils.to_categorical(y_test, num_classes=2)

# # Evaluate the Model
# print("\n--- Evaluating on Test Dataset ---")
# test_results = cnn.evaluate(X_test, y_test, verbose=1, return_dict=True)
# print(f"Test Accuracy: {test_results['accuracy'] * 100:.2f}%")
# print(f"Test Precision: {test_results['precision'] * 100:.2f}%")
# print(f"Test Recall: {test_results['recall'] * 100:.2f}%")
# print(f"Test AUC: {test_results['auc'] * 100:.2f}%")














# for printing k fold validation

# import os
# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import KFold
# from sklearn.metrics import accuracy_score, recall_score
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from model.model_densenet import create_densenet121

# # Ensure eager execution is enabled
# tf.config.run_functions_eagerly(True)

# # Paths
# extract_folder = r'C:\Users\Acer\Desktop\X-Ray\data\train'
# training_data_path = os.path.join(extract_folder, 'train')
# saved_model_path = 'densenet121_model.h5'

# # Image data generator for preprocessing
# datagen = ImageDataGenerator(rescale=1.0 / 255)

# # Load dataset into memory for splitting
# train_data = datagen.flow_from_directory(
#     training_data_path,
#     target_size=(256, 256),
#     batch_size=32,
#     class_mode='binary',
#     shuffle=True
# )

# # Extract data and labels for splitting
# images, labels = [], []
# for i in range(len(train_data)):
#     batch_images, batch_labels = next(train_data)
#     images.extend(batch_images)
#     labels.extend(batch_labels)
# images = np.array(images)
# labels = np.array(labels)

# # K-Fold Cross-Validation
# k = 5  # Number of folds
# kf = KFold(n_splits=k, shuffle=True, random_state=42)

# fold = 1
# accuracy_scores = []
# recall_scores = []

# for train_index, val_index in kf.split(images):
#     print(f"Starting Fold {fold}...")

#     # Splitting data
#     x_train, x_val = images[train_index], images[val_index]
#     y_train, y_val = labels[train_index], labels[val_index]

#     # Create data generators
#     train_gen = datagen.flow(x_train, y_train, batch_size=32, shuffle=True)
#     val_gen = datagen.flow(x_val, y_val, batch_size=32, shuffle=False)

#     # Create or load the model
#     if os.path.exists(saved_model_path) and fold == 1:  # Load only for the first fold
#         print(f"Loading pre-trained model from {saved_model_path}")
#         model = tf.keras.models.load_model(saved_model_path)
#     else:
#         model = create_densenet121()

#     # Compile the model (Ensure metrics are defined before training)
#     model.compile(
#         optimizer='adam',
#         loss='binary_crossentropy',
#         metrics=['accuracy', tf.keras.metrics.Recall()]
#     )

#     # Train the model
#     model.fit(
#         train_gen,
#         steps_per_epoch=len(train_gen),
#         epochs=5,  # Reduce epochs for faster cross-validation
#         validation_data=val_gen,
#         validation_steps=len(val_gen),
#         verbose=1
#     )

#     # Evaluate on validation set
#     val_predictions = (model.predict(val_gen) > 0.5).astype("int32")
#     val_accuracy = accuracy_score(y_val, val_predictions)
#     val_recall = recall_score(y_val, val_predictions)

#     print(f"Fold {fold} - Accuracy: {val_accuracy:.2f}, Recall: {val_recall:.2f}")

#     accuracy_scores.append(val_accuracy)
#     recall_scores.append(val_recall)
#     fold += 1

# # Average cross-validation results
# print(f"\nCross-Validation Results:")
# print(f"Mean Accuracy: {np.mean(accuracy_scores):.2f}")
# print(f"Mean Recall: {np.mean(recall_scores):.2f}")

# # Save the model after cross-validation (optional)
# model.save(saved_model_path)
# print(f"Final model saved to {saved_model_path}")
