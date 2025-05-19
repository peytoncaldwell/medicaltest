import streamlit as st
import pandas as pd
import joblib
from discharge_notes_ai import generate_ai_discharge_notes

# Load ML model
model = joblib.load("ml_model.pkl")

# App title
st.title("🏥 Patient Discharge Assistant AI")

# Patient data input
st.header("📊 Enter Patient Vitals")

temperature = st.number_input("Temperature (°C)", 35.0, 42.0, 37.0)
bp_systolic = st.number_input("Systolic Blood Pressure (mmHg)", 80, 200, 120)
heart_rate = st.number_input("Heart Rate (bpm)", 30, 200, 80)
oxygen_saturation = st.number_input("Oxygen Saturation (%)", 70, 100, 98)
recent_surgery = st.radio("Recent Surgery?", ["Yes", "No"]) == "Yes"
stable_post_op = st.radio("If surgery, is stable post-op?", ["Yes", "No"]) == "Yes"
pending_lab = st.radio("Pending Critical Lab Results?", ["Yes", "No"]) == "Yes"
pain_level = st.slider("Pain Level (0-10)", 0, 10, 3)
mobility = st.radio("Mobility Status", ["Ambulatory", "Bedbound"])

# Upload chart
chart_file = st.file_uploader("📄 Upload Medical Chart (text file)", type="txt")
if chart_file:
    chart_content = chart_file.read().decode("utf-8")
else:
    chart_content = "No chart provided."

# Prediction & Decision
if st.button("🔍 Evaluate Patient"):
    # ML model expects [temp, bp, hr, o2, pain]
    input_data = [[temperature, bp_systolic, heart_rate, oxygen_saturation, pain_level]]
    discharge_risk = model.predict(input_data)[0]

    if discharge_risk == 1:
        decision = "✅ Discharge patient."
        ai_notes = generate_ai_discharge_notes(chart_content)
    else:
        decision = "❌ Continue inpatient care."
        ai_notes = "Discharge not recommended."

    st.subheader("🩺 Decision")
    st.write(decision)

    st.subheader("📑 AI Discharge Notes")
    st.text(ai_notes)

    st.subheader("📜 Uploaded Chart")
    st.text(chart_content)