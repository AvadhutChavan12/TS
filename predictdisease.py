import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_saved_model():
    """Load the trained model and class labels from the .pkl file."""
    with open("tomato_model.pkl", "rb") as f:
        model_path, class_labels = pickle.load(f)
    model = load_model(model_path)
    return model, class_labels

def predict_disease(image_path, model, class_labels):
    """Predicts the disease from a given image."""
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    disease_name = class_labels[predicted_class]
    
    return disease_name, prediction[0][predicted_class]

if __name__ == "__main__":
    model, class_labels = load_saved_model()
    
    user_image = input("Enter the path of the image: ")
    predicted_disease, confidence = predict_disease(user_image, model, class_labels)
    
    print(f"Predicted Disease: {predicted_disease} (Confidence: {confidence:.2f})")
