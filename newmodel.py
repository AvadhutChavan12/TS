import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Disable oneDNN logs to reduce unnecessary warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Define dataset paths
train_dir = "e:/TY Btech/6 sem/TS/Project/tomato/train"
val_dir = "e:/TY Btech/6 sem/TS/Project/tomato/val"

# Image Preprocessing & Data Augmentation
datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)

# Load training and validation data
train_data = datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
val_data = datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

# Define CNN Model
classifier = Sequential([
    Input(shape=(224, 224, 3)),
    Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(units=128, activation='relu'),
    Dropout(0.5),
    Dense(units=len(train_data.class_indices), activation='softmax')
])

# Compile the Model
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
classifier.fit(train_data, validation_data=val_data, epochs=10)

# Save Model & Class Labels
classifier.save("tomato_disease_model.h5")
class_labels = {v: k for k, v in train_data.class_indices.items()}

# Save to .pkl file
with open("tomato_model.pkl", "wb") as f:
    pickle.dump(("tomato_disease_model.h5", class_labels), f)

print("Model and labels saved successfully.")
