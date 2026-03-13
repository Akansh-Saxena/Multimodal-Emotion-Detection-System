import streamlit as st
import requests
from PIL import Image
import io

# ... (keep your title and header code) ...

# REPLACE the backend_url with your LIVE Streamlit URL but change '.streamlit.app' to '.streamlit.app/predict'
# For now, we will use a relative path trick:
backend_url = "https://akansh-saxena-multimodal-emotion-detection-system-srcapi-tkz2zh.streamlit.app/predict"

if img_file:
    st.info("🚀 Analyzing your emotion in real-time...")
    
    # Convert the camera photo to bytes
    img = Image.open(img_file)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    byte_im = buf.getvalue()

    # Send it to your FastAPI backend
    try:
        files = {"file": ("image.jpg", byte_im, "image/jpeg")}
        response = requests.post(backend_url, files=files, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ Emotion Detected: {result['emotion']}")
            st.write(f"Confidence: {result['confidence']}%")
        else:
            st.error("Backend is busy. Retrying in 2s...")
    except Exception as e:
        st.warning("Connecting to Engine... Ensure src/api.py is running in the background.")
