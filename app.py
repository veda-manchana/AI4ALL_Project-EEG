import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
def predict(data):
    clf = joblib.load("rf_model.sav")  # Your saved EEG model
    scaler = joblib.load("scaler.sav")  # Your saved scaler
    data_scaled = scaler.transform(data)  # Scale inputs
    return clf.predict(data_scaled)

st.title('EEG Emotion Classification')
st.markdown('Predicting **positive** or **negative** emotions from EEG data using 14 features.')

st.header("EEG Feature Inputs")

# EEG features
features = ['AF3', 'AF4', 'F3', 'F4', 'F7', 'F8', 'FC5', 'FC6',
            'T7', 'T8', 'P7', 'P8', 'O1', 'O2']

input_values = []
cols = st.columns(2)

i = 0
for feature in features:
    with cols[i % 2]:
        val = st.number_input(f"{feature} value", value=0.0, step=0.01)
        input_values.append(val)
    i += 1

# Prediction
if st.button("Predict Emotion"):
    result = predict(np.array([input_values]))
    label = "Positive" if result[0] == 1 else "Negative"
    st.success(f"Prediction: {label}")
