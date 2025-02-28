import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

def load_saved_model():
    """Load the trained model and class labels from the .pkl file."""
    with open("tomato_model.pkl", "rb") as f:
        model_path, class_labels = pickle.load(f)
    model = load_model(model_path)
    return model, class_labels

model, class_labels = load_saved_model()

def predict_disease(image_path, model, class_labels):
    """Predicts the disease from a given image."""
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    disease_name = class_labels[predicted_class]

    return disease_name, float(prediction[0][predicted_class])

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            disease_name, confidence = predict_disease(filepath, model, class_labels)
            return render_template("index.html", prediction=disease_name, confidence=confidence, image_url=filepath)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
