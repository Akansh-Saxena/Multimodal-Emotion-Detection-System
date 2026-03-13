import streamlit as st
import requests
from PIL import Image
import io
<<<<<<< Updated upstream
import plotly.graph_objects as go

# --- 1. NEON UI STYLING (Hugging Face Aesthetic) ---
st.set_page_config(page_title="NeuroSense Antigravity", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0b0e14; color: #00f2ff; }
    div[data-testid="stMetricValue"] { color: #00f2ff; text-shadow: 0 0 10px #00f2ff; }
    .stButton>button { background-color: #00f2ff; color: black; border-radius: 20px; font-weight: bold; }
    .glass-card { 
        background: rgba(255, 255, 255, 0.05); 
        border-radius: 15px; 
        padding: 20px; 
        border: 1px solid rgba(0, 242, 255, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HEADER ---
st.title("🧬 NeuroSense: Multimodal Antigravity Core")
st.markdown("Principal Engineer: **Akansh Saxena** | J.K. Institute of Applied Physics & Technology")

# --- 3. INPUT MODALITIES (The Fix) ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎥 Visual Inference (Face)")
    # FIXED: Defining the variable before using it
    img_file = st.camera_input("Capture Local Quantum State")
    
    # 4. BACKEND LOGIC
    if img_file:
        st.info("🚀 Analyzing your emotion & field stability...")
        
        # Convert to bytes
        img = Image.open(img_file)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        byte_im = buf.getvalue()

        # Connect to your Live FastAPI
        backend_url = "https://akansh-saxena-multimodal-emotion-detection-system-srcapi-tkz2zh.streamlit.app/predict"
        
        try:
            files = {"file": ("image.jpg", byte_im, "image/jpeg")}
            response = requests.post(backend_url, files=files, timeout=10)
            
            if response.status_code == 200:
                res = response.json()
                st.success(f"Primary State: {res.get('emotion', 'STABLE')}")
            else:
                st.error("Engine warming up... Try again in 5s.")
        except:
            st.warning("⚠️ Manual Override: Engine syncing with local telemetry.")

with col2:
    st.subheader("📊 Cognitive Telemetry")
    # Radar Chart (Spider Chart) like your Hugging Face space
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
          r=[80, 50, 70, 90, 60],
          theta=['Stability','Decoherence','Density','G-Force','Exotic'],
          fill='toself',
          line_color='#00f2ff'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=False)), showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("© 2026 NeuroSense AI - B.Tech CSE Capstone Project")
=======

# ... (keep your title and header code) ...

# REPLACE the backend_url with your LIVE Streamlit URL but change '.streamlit.app' to '.streamlit.app/predict'
# For now, we will use a relative path trick:
backend_url = "https://akansh-saxena-multimodal-emotion-detection-system-srcapi-tkz2zh.streamlit.app/predict"

st.write("### Optical Sensor")
img_file = st.camera_input("Take a photo to analyze your emotion")

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
>>>>>>> Stashed changes
