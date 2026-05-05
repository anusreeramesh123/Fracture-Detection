from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# -----------------------
# Load model safely
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mura_phase1.keras")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (224, 224)

# -----------------------
# Image preprocessing
# -----------------------
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0

    if image.shape[-1] == 4:  # remove alpha channel if present
        image = image[..., :3]

    image = np.expand_dims(image, axis=0)
    return image

# -----------------------
# Home route
# -----------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------
# Prediction route
# -----------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No selected file"

    try:
        image = Image.open(file).convert("RGB")
        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)[0][0]

        if prediction > 0.5:
            result = "Abnormal (Positive)"
            confidence = prediction
        else:
            result = "Normal (Negative)"
            confidence = 1 - prediction

        return render_template(
            "result.html",
            prediction=result,
            confidence=round(float(confidence * 100), 2)
        )

    except Exception as e:
        return f"Error during prediction: {str(e)}"

# -----------------------
# Run app
# -----------------------
if __name__ == "__main__":
    app.run(debug=True)