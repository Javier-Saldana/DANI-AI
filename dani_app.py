import streamlit as st
import joblib
import numpy as np
from PIL import Image


# Load trained model
model = joblib.load("dani_model.pkl")

# UI layout
st.set_page_config(page_title="Dani - AI Gas Classifier", layout="centered")
image = Image.open("dani_avatar.png")
st.image(image, width=120)
st.title("🤖 Dani - AI Gas Classifier")
st.write("Simulate sensor readings below and let Dani identify the gas.")

# Sliders for sensor input
s1 = st.slider("MQ8_1", 0, 4095, 1000)
s2 = st.slider("MQ135_1", 0, 4095, 1000)
s3 = st.slider("MQ4_1", 0, 4095, 1000)
s4 = st.slider("MQ8_2", 0, 4095, 1000)
s5 = st.slider("MQ135_2", 0, 4095, 1000)
s6 = st.slider("MQ4_2", 0, 4095, 1000)

# Predict
features = np.array([[s1, s2, s3, s4, s5, s6]])
prediction = model.predict(features)[0]
proba = model.predict_proba(features)[0]

# Display result
st.markdown(f"### 🧠 Predicted Gas: **`{prediction.upper()}`**")
st.write("Prediction Confidence:")
for label, p in zip(model.classes_, proba):
    st.write(f"- **{label}**: {round(p*100, 2)}%")
