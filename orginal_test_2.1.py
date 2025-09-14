from flask import Flask, request, render_template, jsonify
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
model_1 = load_model('D_R.h5')

# Define the target shape for input images
target_shape = (64, 64)  # Adjust this to match the target shape used during training

# Define your class labels
classes_1 = ['Moderate', 'No_DR']

# Function to preprocess and classify an image file
def classify_image(file_path, model):
    # Load and preprocess the image
    img = load_img(file_path, target_size=target_shape)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values to [0, 1]

    # Make predictions
    predictions = model.predict(img_array)

    # Get the class probabilities
    class_probabilities = predictions[0]

    # Get the predicted class index
    predicted_class_index = np.argmax(class_probabilities)

    return class_probabilities, predicted_class_index

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload and classification
@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the file to a temporary location
    temp_file_path = os.path.join('uploads', image_file.filename)
    os.makedirs('uploads', exist_ok=True)
    image_file.save(temp_file_path)

    # Classify the image
    try:
        class_probabilities, predicted_class_index = classify_image(temp_file_path, model_1)
        predicted_class = classes_1[predicted_class_index]

        # Prepare the response
        results = {
            'predicted_class': predicted_class,
            'class_probabilities': {
                classes_1[i]: float(prob) for i, prob in enumerate(class_probabilities)
            }
        }

        return jsonify(results)

    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
