"""
Flask API for Cat vs Dog Image Classification

This API serves the trained model and handles image classification requests.
"""

import os
import io
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image as PILImage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
MODEL_PATH = "cat_dog_model.h5"
model = None


def load_trained_model():
    """Load the trained model from file."""
    global model
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading model from {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
    else:
        logger.error(f"Model file {MODEL_PATH} not found")
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found")


def preprocess_image(img):
    """Preprocess the image to match model input requirements."""
    # Resize image to 224x224
    img = img.resize((224, 224))

    # Convert to array and normalize pixel values
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0,1]

    # Expand dimensions to match model input (batch_size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_image(img):
    """Make prediction on a single image."""
    global model

    if model is None:
        raise Exception("Model not loaded")

    # Preprocess the image
    processed_img = preprocess_image(img)

    # Make prediction
    prediction = model.predict(processed_img)[0][0]

    # Calculate probabilities for both classes
    dog_prob = float(prediction)
    cat_prob = float(1 - prediction)

    # Determine the predicted class
    predicted_class = "dog" if prediction > 0.5 else "cat"
    confidence = max(dog_prob, cat_prob)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": {
            "cat": cat_prob,
            "dog": dog_prob
        }
    }


@app.route('/')
def home():
    """Home endpoint with API information."""
    return jsonify({
        "message": "Cat vs Dog Classification API",
        "endpoints": {
            "predict": {
                "method": "POST",
                "url": "/predict",
                "description": "Classify an image as cat or dog",
                "request_format": {
                    "multipart_form_data": "Upload image file with key 'image'",
                    "json_base64": "Send base64 encoded image in JSON with key 'image'"
                },
                "response": {
                    "predicted_class": "string ('cat' or 'dog')",
                    "confidence": "float (0.0-1.0)",
                    "probabilities": {
                        "cat": "float (probability of cat)",
                        "dog": "float (probability of dog)"
                    }
                }
            }
        }
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Predict whether the uploaded image is a cat or dog."""
    try:
        global model

        # Check if model is loaded
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # Check if image was provided
        if 'image' not in request.files and 'image' not in request.json:
            return jsonify({"error": "No image provided. Send image as multipart form data with key 'image' or as base64 in JSON with key 'image'"}), 400

        # Get image from request
        img = None

        if 'image' in request.files:
            # Handle file upload
            file = request.files['image']
            if file.filename == '':
                return jsonify({"error": "Empty filename provided"}), 400

            # Open and process the image
            img = PILImage.open(file.stream)

        elif 'image' in request.json:
            # Handle base64 encoded image
            img_data = request.json['image']

            # Check if it's a base64 string with data URI prefix
            if img_data.startswith('data:image'):
                # Remove data URI prefix (e.g., "data:image/jpeg;base64,")
                img_data = img_data.split(',')[1]

            # Decode base64 string
            img_bytes = base64.b64decode(img_data)
            img = PILImage.open(io.BytesIO(img_bytes))

        # Validate image
        if img is None:
            return jsonify({"error": "Could not process image"}), 400

        # Verify image format
        if img.format not in ['JPEG', 'PNG', 'JPG']:
            return jsonify({"error": f"Unsupported image format: {img.format}. Supported formats: JPEG, PNG, JPG"}), 400

        # Make prediction
        result = predict_image(img)

        # Return prediction result
        return jsonify({
            "success": True,
            "prediction": result
        })

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })


if __name__ == '__main__':
    try:
        # Load the model when the application starts
        load_trained_model()

        # Run the Flask app
        app.run(host='0.0.0.0', port=5000, debug=False)

    except FileNotFoundError as e:
        logger.error(f"Failed to start API: {e}")
        print(f"Error: {e}")
        print("Please ensure the model file 'cat_dog_model.h5' exists in the current directory.")
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        print(f"Unexpected error: {e}")
