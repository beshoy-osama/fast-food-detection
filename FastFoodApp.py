import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load trained model
model = load_model('C:/Users/besho/Downloads/FastFoodDetection/FastFoodDetection.keras')

st.title('Fast-Food 10-Class Image Classifier')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Target size matches your model input
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Define your real class names in order here
CLASS_NAMES = [
    "Baked Potato", "Burger", "Crispy Chicken", "Donut", "Fries", 
    "Hot Dog", "Pizza", "Sandwich", "Taco", "Taquito"
]

def preprocess_image(image):
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image = image.convert('RGB')
    img_array = np.array(image) / 255.0
    return img_array.reshape(1, IMG_HEIGHT, IMG_WIDTH, 3)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    st.write("")
    st.write("Classifying...")

    img_array = preprocess_image(image)
    prediction = model.predict(img_array)[0]

    predicted_class_idx = np.argmax(prediction)
    predicted_label = CLASS_NAMES[predicted_class_idx]
    confidence = prediction[predicted_class_idx]

    st.write(f"**Prediction:** {predicted_label}")
    st.write(f"**Confidence:** {confidence:.2%}")
