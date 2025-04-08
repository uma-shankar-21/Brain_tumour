import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st

model = Sequential()

model.add(Conv2D(32, kernel_size=(2, 2), input_shape=(128, 128, 3), padding = 'Same'))
model.add(Conv2D(32, kernel_size=(2, 2),  activation ='relu', padding = 'Same'))


model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))
model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss = "categorical_crossentropy", optimizer='Adamax')


# Load your model (assuming model.pkl is the model file)
#model = tf.keras.models.load_model('model.pkl')

def names(number):
    if number == 0:
        return 'Its a Tumor'
    else:
        return 'No, Its not a tumor'

def recognize_image(image):
    # Resize the image to the expected dimensions
    img = image.resize((128, 128))
    # Convert the image to a NumPy array
    x = np.array(img)
    # Reshape the image to match the model input
    x = x.reshape(1, 128, 128, 3)

    # Make a prediction
    res = model.predict(x)
    classification = np.argmax(res, axis=-1)[0]

    # Map the class index to the actual class name
    class_names = ['No Tumor', 'Tumor']  # Example class names, update according to your model
    result = names(classification)

    return result

# Streamlit App Layout
st.title("Brain Tumor Prediction App")
st.markdown("### Upload an image to check if a brain tumor is present or not")

# Display the description and long description
desc = "Brain tumor app. Let's learn!"
long_desc = "Select an image or upload one to predict if a brain tumor is present or not."

st.markdown(f"<p style='font-size:16px'>{desc}</p>", unsafe_allow_html=True)
st.markdown(f"<p style='font-size:14px'>{long_desc}</p>", unsafe_allow_html=True)

# Upload image
image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if image_file is not None:
    # Load image
    image = Image.open(image_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Perform prediction
    if st.button('Predict'):
        result = recognize_image(image)
        st.write(f"Prediction: {result}")
