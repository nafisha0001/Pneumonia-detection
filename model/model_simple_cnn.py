import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers, models


def create_simple_cnn(input_shape=(256, 256, 3)):
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(16, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='sigmoid'),
        keras.layers.Dense(128, activation='sigmoid'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')  
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',  
        metrics=['accuracy', tf.keras.metrics.Recall()]
    )
    return model

# def create_simple_cnn(input_shape=(256, 256, 3)):
#     model = models.Sequential([
#         layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
#         layers.MaxPooling2D((2, 2)),
#         layers.Conv2D(64, (3, 3), activation='relu'),
#         layers.MaxPooling2D((2, 2)),
#         layers.Conv2D(128, (3, 3), activation='relu'),
#         layers.MaxPooling2D((2, 2)),
#         layers.Conv2D(256, (3, 3), activation='relu'),  
#         layers.MaxPooling2D((2, 2)),
#         layers.Conv2D(512, (3, 3), activation='relu'),  
#         layers.MaxPooling2D((2, 2)),  
#         layers.Flatten(),
#         layers.Dense(512, activation='relu'),
#         layers.Dropout(0.5),
#         layers.Dense(1, activation='sigmoid')
#     ])
    # model.compile(
    #             #   optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
    #               optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    #             #   optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001),
    #             #   optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0001, weight_decay=1e-5),
    #               loss='binary_crossentropy',
    #               metrics=['accuracy', tf.keras.metrics.Recall()])
    # return model


# import tensorflow as tf
# from tensorflow.keras import layers, models

# def create_simple_cnn(input_shape=(256, 256, 3)):
#     model = models.Sequential([
#         layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D((2, 2)),

#         layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D((2, 2)),

#         layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D((2, 2)),

#         layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D((2, 2)),

#         layers.Flatten(),
#         layers.Dense(512, activation='relu'),
#         layers.Dropout(0.5),  
#         layers.Dense(1, activation='sigmoid')
#     ])

#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#         loss='binary_crossentropy',
#         metrics=['accuracy', tf.keras.metrics.Recall()]
#     )

#     return model