from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os

from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import cv2

app = Flask(__name__)

# Define the path to the uploaded images folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the 'uploads' directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the saved model
loaded_model = load_model('defect_detection_model.h5')

# Function to detect defects
def detect_defect(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (224, 224))  # Resize image if needed
    img = img / 255.0  # Normalize pixel values

    # Make predictions using the loaded model
    prediction = loaded_model.predict(np.expand_dims(img, axis=0))

    # Check the prediction result
    if prediction[0][0] >= 0.5:  # Assuming binary classification (defect or non-defect)
        result = "Defect Detected"
    else:
        result = "No Defect Detected"

    return result

# Route to the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']

        # If the user does not select a file, browser also
        # submits an empty part without filename
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        

        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            detection_result = detect_defect(image_path)
            return render_template('result.html', image_name=filename, result=detection_result)


if __name__ == '__main__':
    app.run(debug=True)
