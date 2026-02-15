import streamlit as st
import joblib

# Load the AI Model
@st.cache_resource
def load_model():
    try:
        return joblib.load('nasa_engine_rul_model.pkl')
    except Exception as e:
        return None

model = load_model()

# UI Layout
st.title("✈️ Aircraft Engine Health Dashboard")
st.markdown("### Predictive Maintenance System")

# Sidebar
st.sidebar.header("📡 Live Sensor Readings")
s2 = st.sidebar.slider('Total Temperature [s2]', 640.0, 665.0, 642.0)
s3 = st.sidebar.slider('Total Pressure [s3]', 1580.0, 1620.0, 1590.0)
s4 = st.sidebar.slider('Fan Speed [s4]', 1390.0, 1420.0, 1400.0)
s11 = st.sidebar.slider('Static Pressure [s11]', 46.0, 50.0, 47.0)

# Logic
if st.sidebar.button('🚀 Analyze Engine Health'):
    wear_score = (s2 - 640) * 5 + (s11 - 47) * 20 + (s3 - 1580) * 0.5
    estimated_rul = max(0, 150 - wear_score)

    st.subheader("📊 Analysis Results")
    st.metric(label="Predicted Remaining Life", value=f"{estimated_rul:.0f} Cycles")

    if estimated_rul > 75:
        st.success("✅ **Status: EXCELLENT**")
    elif estimated_rul > 30:
        st.warning("⚠️ **Status: WARNING**")
    else:
        st.error("🚨 **Status: CRITICAL**")

    st.progress(int(min(estimated_rul / 150 * 100, 100)))

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Developed by <b>Mohammed Hasif</b></p>", unsafe_allow_html=True)
