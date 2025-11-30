from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def build_model(input_shape=(224, 224, 3), learning_rate=1e-4):
    
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.2),        
    ])

    base_model = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    base_model.trainable = False  

    inputs = keras.Input(shape=input_shape)
    
    x = data_augmentation(inputs)
    
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    
    x = base_model(x, training=False) 
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x) 
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)


    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
