import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

def load_model():
    try:
        # Adjust the path to where your model is saved
        model_path = 'C:/Users/Rohith Sai/Documents/AI IMAGE CLASSIFIER/AIGeneratedModel.h5'
        model = tf.keras.models.load_model(model_path)
        st.write("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_image(model, image):
    try:
        # Preprocess the image
        img = image.resize((48, 48))  # Adjust size as per your model's requirements
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Make prediction
        predictions = model.predict(img_array)
        st.write("Prediction successful.")
        return predictions
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

st.title("Image Classifier")
st.write("Upload an image to classify")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        
        model = load_model()
        if model:
            predictions = predict_image(model, image)
            if predictions is not None:
                st.write(f"Predictions: {predictions}")

                if predictions > 0.5:
                    st.write("AI GENERATED IMAGE")
                else:
                    st.write("REAL WORLD IMAGE")
            else:
                st.error("Failed to get predictions.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
