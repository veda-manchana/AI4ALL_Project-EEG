import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
def predict(data):
    clf = joblib.load("EEG_RF_Model.pkl")
    scaler = joblib.load("EEG_Scaler.pkl")
    # data is 2D array with 14 EEG channels
    df = pd.DataFrame(data, columns=['AF3', 'AF4', 'F3', 'F4', 'F7', 'F8',
                                     'FC5', 'FC6', 'T7', 'T8', 'P7', 'P8',
                                     'O1', 'O2'])
    
    # compute variance features like in training
    df["frontal_var"] = df[['AF3', 'AF4', 'F3', 'F4', 'F7', 'F8']].var(axis=1)
    df["fronto_central_var"] = df[['FC5', 'FC6']].var(axis=1)
    df["temporal_var"] = df[['T7', 'T8']].var(axis=1)
    df["parietal_var"] = df[['P7', 'P8']].var(axis=1)
    df["occipital_var"] = df[['O1', 'O2']].var(axis=1)
    
    data_scaled = scaler.transform(df)  # now feature count matches training
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


