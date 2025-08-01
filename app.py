import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
def predict(data):
    clf = joblib.load("rf_model.sav")  # Your saved EEG model
    return clf.predict(data)

# Map class labels to display text/images
def class_to_image(class_label):
    if class_label == 1:  # Positive
        return "images/positive.jpg"  # Replace with your image path
    elif class_label == 0:  # Negative
        return "images/negative.jpg"  # Replace with your image path

st.title('EEG Emotion Classification')
st.markdown('Predicting **positive** or **negative** emotions from EEG data using 14 features.')

st.header("EEG Feature Inputs")

# Create input fields for 14 EEG features
features = ['AF3', 'AF4', 'F3', 'F4', 'F7', 'F8', 'FC5', 'FC6',
            'T7', 'T8', 'P7', 'P8', 'O1', 'O2']

input_values = []
cols = st.columns(2)  # Split into two columns for readability

for i, feature in enumerate(features):
    with cols[i % 2]:
        val = st.number_input(f"{feature} value", value=0.0, step=0.01)
        input_values.append(val)

# Prediction button
if st.button("Predict Emotion"):
    result = predict(np.array([input_values]))
    label = "Positive Emotion" if result[0] == 1 else "Negative Emotion"
    st.success(f"Prediction: {label}")

    # Show image based on prediction
    image_path = class_to_image(result[0])
    if image_path:
        st.image(image_path, use_column_width=True)
