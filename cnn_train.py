import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Disable oneDNN logs to reduce unnecessary warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Define dataset paths (Update these paths to match your dataset location)
# Correct absolute paths
train_dir = "e:/TY Btech/6 sem/TS/Project/tomato/train"
val_dir = "e:/TY Btech/6 sem/TS/Project/tomato/val"


# Image Preprocessing & Data Augmentation
datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)

# Load training and validation data
train_data = datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
val_data = datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

# Define CNN Model
classifier = Sequential([
    Input(shape=(224, 224, 3)),  # Explicit input layer
    Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(units=128, activation='relu'),  # Fixed `output_dim` issue
    Dropout(0.5),
    Dense(units=len(train_data.class_indices), activation='softmax')  # Dynamic output layer
])

# Compile the Model
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print Model Summary
classifier.summary()

# Train Model
classifier.fit(train_data, validation_data=val_data, epochs=3)

# Save Model
classifier.save("tomato_disease_model.h5")

# Load Class Labels
class_labels = {v: k for k, v in train_data.class_indices.items()}
print("Class Labels:", class_labels)

# Function to Predict Disease from User Image
def predict_disease(image_path):
    """Predicts the disease from a given image"""
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    model = load_model("tomato_disease_model.h5")  # Load trained model
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    disease_name = class_labels[predicted_class]

    return disease_name, prediction[0][predicted_class]

# Example Prediction
user_image = "image2.jpg"  # Replace with actual image path
predicted_disease, confidence = predict_disease(user_image)
print(f"Predicted Disease: {predicted_disease} (Confidence: {confidence:.2f})")
