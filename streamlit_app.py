import streamlit as st
import requests

st.set_page_config(page_title="Multimodal Emotion AI", page_icon="🧠")

st.title("🧠 Multimodal Emotion Detection")
st.markdown("Developed by **Akansh Saxena** | J.K. Institute of Applied Physics & Technology")

st.sidebar.header("Settings")
backend_url = "http://localhost:8501" # This will connect to your api.py

# --- UI Layout ---
tab1, tab2 = st.tabs(["🎥 Video/Image Analysis", "🎤 Audio Analysis"])

with tab1:
    st.subheader("Visual Emotion Recognition")
    img_file = st.camera_input("Capture image for analysis")
    if img_file:
        st.info("Sending data to FastAPI backend...")
        # Add your backend calling logic here

with tab2:
    st.subheader("Voice Emotion Analysis")
    audio_file = st.file_uploader("Upload audio clip", type=["wav", "mp3"])
    if audio_file:
        st.success("Audio received. Processing frequencies...")

st.divider()
st.caption("© 2026 Multimodal AI Project - Final Year B.Tech CSE")
