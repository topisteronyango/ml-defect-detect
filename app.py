import streamlit as st
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from keras.models import load_model
from tensorflow.keras.optimizers import Adam 


os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from tensorflow.keras.optimizers import Optimizer
import tensorflow.keras.backend as K

class CustomAdam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7, **kwargs):
        super(CustomAdam, self).__init__(**kwargs)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.learning_rate
        t = self.iterations + 1

        lr_t = lr * (K.sqrt(1. - K.pow(self.beta2, t)) / (1. - K.pow(self.beta1, t)))

        for p, g in zip(params, grads):
            m = K.zeros(K.int_shape(p))
            v = K.zeros(K.int_shape(p))

            m_t = (self.beta1 * m) + (1. - self.beta1) * g
            v_t = (self.beta2 * v) + (1. - self.beta2) * K.square(g)

            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            self.updates.append(K.update(p, p_t))
        return self.updates

    def get_config(self):
        config = {
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
        }
        base_config = super(CustomAdam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




# Define the custom_objects dictionary for loading the model
# custom_objects = {'CustomAdam': custom_adam}

# Load the model using load_model and pass the custom_objects
# loaded_model = load_model('defect_detection_model.h5', custom_objects={'CustomAdam': CustomAdam})

loaded_model = load_model('defect_detection_model.h5', compile=False)


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
