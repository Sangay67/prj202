from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import cv2
import os
import ssl
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Disable SSL verification for downloading pre-trained models, if needed
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
print("Loading model...")
model = tf.keras.models.load_model('./model', compile=False)
print("Model loaded successfully.")

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess the image
def preprocess_image(file_path):
    """
    Preprocesses the image for model prediction.
    Args:
        file_path (str): Path to the image file.
    Returns:
        numpy.ndarray: Preprocessed image ready for prediction.
    """
    print(f"Preprocessing image: {file_path}")
    img = load_img(file_path, target_size=(64, 64))  # Resize
    img_array = img_to_array(img) / 255.0            # Normalize
    img_array = np.expand_dims(img_array, axis=0)    # Add batch dimension
    print(f"Image preprocessed. Shape: {img_array.shape}")
    return img_array

# Home page route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("POST /predict received.")

        if 'file' not in request.files:
            print("No file part in request.")
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            print("No file selected.")
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            print(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type. Allowed types are png, jpg, jpeg, bmp'}), 400

        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        print(f"File saved to: {file_path}")

        # Preprocess the image
        img = preprocess_image(file_path)

        # Perform prediction
        print("Running model prediction...")
        # Get serving function
        infer = model.signatures["serving_default"]

# Run inference
        output = infer(tf.convert_to_tensor(img))

# The output is a dictionary → get first tensor
        predictions = list(output.values())[0].numpy()

# Now same as before:
        predicted_class = int(np.argmax(predictions[0]))

        print('Predicted class is:', predicted_class)

        # Define class names
        class_names = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
        predicted_label = class_names[predicted_class]

        # Return result
        response = {
            'predicted_class': predicted_class,
            'predicted_label': predicted_label,
            'confidence': float(np.max(predictions[0]))
        }

        print("Prediction successful:", response)
        return jsonify(response)

    except Exception as e:
        print("ERROR in /predict:", str(e))
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up: remove saved file
        try:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
                print(f"Cleaned up uploaded file: {file_path}")
        except Exception as cleanup_error:
            print(f"Error cleaning up file: {cleanup_error}")
print("Model type:", type(model))

# Main entry
if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
        print("Created 'uploads' directory.")
    app.run(debug=True)
