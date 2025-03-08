import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121, DenseNet169

def create_densenet121(input_shape=(256, 256, 3)):
    # Loading the DenseNet121 model without the top layer
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False  # Freezing the base model layers
    # base_model.trainable = True   # Allowing fine tuning

    # Creating the custom model on top of DenseNet121
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        # layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compiling the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        # optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0001, weight_decay=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall()]
    )
    return model


def create_densenet169(input_shape=(256, 256, 3)):
    base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
