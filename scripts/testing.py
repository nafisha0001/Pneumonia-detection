import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_model(model_path):
    """Loads the pre-trained model from the given path."""
    if os.path.exists(model_path):
        print(f"🔄 Loading model from {model_path}")
        return tf.keras.models.load_model(model_path)
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

def preprocess_image(img_path):
    """Preprocesses the image to match model input requirements."""
    img = image.load_img(img_path, target_size=(256, 256))  
    img_array = image.img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

def predict_pneumonia(model, img_path):
    """Predicts whether the image has pneumonia or not."""
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    return "Pneumonia Detected" if prediction[0][0] > 0.5 else "Normal"

if __name__ == "__main__":
    model_path = r"C:\Users\Acer\Desktop\Pneumonia\densenet121_model.h5"
    # test_img_path = r"C:\Users\Acer\Desktop\Pneumonia\data\val\NORMAL\NORMAL2-IM-1427-0001.jpeg"  
    test_img_path = r"C:\Users\Acer\Desktop\Pneumonia\data\val\PNEUMONIA\person1947_bacteria_4876.jpeg"
    
    model = load_model(model_path)
    result = predict_pneumonia(model, test_img_path)
    print(f"Prediction: {result}")
