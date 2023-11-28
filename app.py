import streamlit as st
from werkzeug.utils import secure_filename
import os

import cv2
import numpy as np
from keras.models import load_model


# Load the saved model
# loaded_model = None  # Load your model here (ensure it's accessible)
loaded_model = load_model('defect_detection_model.h5')

# Define the path to the uploaded images folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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

# Streamlit app code
st.title('Defect Detection')

uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg', 'bmp'])

if uploaded_file is not None:
    
    file_location = os.path.join(UPLOAD_FOLDER, secure_filename(uploaded_file.name))
    with open(file_location, "wb") as f:
        f.write(uploaded_file.getbuffer())

    detection_result = detect_defect(file_location)
    st.write("Detection Result:", detection_result)
    st.image(file_location, caption='Uploaded Image', use_column_width=True)
