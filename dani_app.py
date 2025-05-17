import streamlit as st
import joblib
import numpy as np
from PIL import Image

st.markdown("---")
st.markdown("<h3 style='color:#FF9800'>🛰️ Reactor Monitoring Dashboard</h3>", unsafe_allow_html=True)

# Simulated reactor data
reactors = [
    {"name": "Reactor 1", "status": "🟢 Stable", "co2": 2.31, "note": "Normal operation"},
    {"name": "Reactor 2", "status": "🟡 Warning", "co2": 1.45, "note": "Sensor drift detected"},
    {"name": "Reactor 3", "status": "🔴 Error", "co2": 0.00, "note": "No response detected"},
]

# Show reactor cards
total_co2 = 0
for reactor in reactors:
    st.markdown(f"""
    <div style='border:1px solid #ddd;padding:10px;border-radius:8px;margin-bottom:10px'>
        <strong>{reactor['status']} {reactor['name']}</strong><br>
        CO₂ Converted: <strong>{reactor['co2']} kg</strong><br>
        <i>{reactor['note']}</i>
    </div>
    """, unsafe_allow_html=True)
    total_co2 += reactor['co2']

# Summary
st.markdown(f"### 🧠 Status Summary")
st.markdown(f"**Reactors Online:** 2 / 3")
st.markdown(f"**Estimated Total CO₂ Converted Today:** `{round(total_co2, 2)} kg`")

# Load trained model
model = joblib.load("dani_model.pkl")

# Set UI config
st.set_page_config(page_title="Dani - AI Gas Classifier", layout="centered")

# Load and show Promex logo at top-left
logo = Image.open("promex_logo.png")
col1, col2 = st.columns([1, 6])
with col1:
    st.image(logo, width=80)
with col2:
    st.markdown("## ")  # spacing adjustment

# Load and show Dani's avatar
image = Image.open("avatar_dani.png")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(image, width=120)
    st.markdown("<h2 style='text-align: center;'>🤖 Dani – AI Gas Classifier</h2>", unsafe_allow_html=True)

st.markdown("""
Meet **Dani** — your AI-based digital engineer trained to monitor and manage the PROMEX M41 reactor.

Dani doesn’t sleep, doesn’t blink, and never misses an anomaly.

🧠 Powered by AI  
📟 Keeps M41 working at optimal conditions 
🧪 Prevents failure
""")

# Sliders for simulated sensor input
s1 = st.slider("MQ8_1", 0, 4095, 1000)
s2 = st.slider("MQ135_1", 0, 4095, 1000)
s3 = st.slider("MQ4_1", 0, 4095, 1000)
s4 = st.slider("MQ8_2", 0, 4095, 1000)
s5 = st.slider("MQ135_2", 0, 4095, 1000)
s6 = st.slider("MQ4_2", 0, 4095, 1000)

# Predict gas type
features = np.array([[s1, s2, s3, s4, s5, s6]])
prediction = model.predict(features)[0]
proba = model.predict_proba(features)[0]

# Display prediction results
st.markdown("---")
st.markdown("---")
st.markdown("<h3 style='color:#4CAF50'>🧠 Prediction Engine</h3>", unsafe_allow_html=True)

st.markdown(f"""
<div style='padding:10px;background-color:#f0f0f0;border-radius:10px;margin-bottom:10px'>
    <h4>Predicted Gas: <span style='color:#2196F3'>{prediction.upper()}</span></h4>
</div>
""", unsafe_allow_html=True)

st.markdown("Confidence Levels:")
for label, p in zip(model.classes_, proba):
    st.write(f"- **{label}**: {round(p*100, 2)}%")

# --- Simulated Billing Section ---
st.markdown("---")
st.subheader("💸 Simulated Billing")

st.markdown("#### Site: Promex Pilot Facility – Birmingham, Alabama")
st.markdown("**Reactors Online:** 3")

# Simulated data (you can later make this dynamic)
reactor_data = {
    "Reactor 1": 2.35,  # kg CO2 converted
    "Reactor 2": 3.12,
    "Reactor 3": 1.87
}
price_per_kg = 0.40  # USD

# Display each reactor
total_co2 = 0
for name, kg in reactor_data.items():
    st.write(f"- {name}: {kg} kg CO₂ converted today")
    total_co2 += kg

# Billing summary
total_cost = total_co2 * price_per_kg
st.markdown(f"**Total CO₂ converted:** `{round(total_co2, 2)} kg`")
st.markdown(f"**Billing Rate:** `${price_per_kg:.2f} USD/kg`")
st.markdown(f"**Estimated Daily Cost:** `${total_cost:.2f}`")


# Sidebar with Dani info
st.sidebar.title("About Dani")
st.sidebar.markdown("""
DANI  
Version: `1.0`

Developed by **Promex Technologies, Inc.**, Dani is a reactor-class AI trained to monitor and maintain the M41 reactor at optimal performance.

Dani uses machine learning to replace expensive infrared sensors with simple, low-cost alternatives - making the total cost of the M41 reactor approximately **5%** of what it would be without AI.

Built for a decarbonized future.
""")
