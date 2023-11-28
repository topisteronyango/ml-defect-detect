import streamlit as st
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from keras.models import load_model
from tensorflow.keras.optimizers import Adam 


os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Define a custom optimizer
class CustomAdam(Adam):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        super().__init__(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
        # You can add additional custom parameters or initialization here

    def apply_gradients(self, grads_and_vars, name=None):
        # Custom logic to apply gradients
        # Override this method to define your custom optimization algorithm

        # Example: Negate gradients
        grads_and_vars = [(tf.negative(grad), var) for grad, var in grads_and_vars]

        return super().apply_gradients(grads_and_vars, name=name)
custom_adam = CustomAdam()

custom_objects = {'CustomAdam': CustomAdam}  # Provide the custom optimizer to custom_objects

# Load the saved model with custom_objects
loaded_model = load_model('./defect_detection_model.h5', custom_objects=custom_objects)


# Define the path to the uploaded images folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to detect defects
def detect_defect(img):
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
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    
    if img is not None:
        file_location = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        cv2.imwrite(file_location, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        detection_result = detect_defect(img)
        st.write("Detection Result:", detection_result)
        st.image(file_location, caption='Uploaded Image', use_column_width=True)
    else:
        st.write("Invalid Image File. Please upload a valid image.")
