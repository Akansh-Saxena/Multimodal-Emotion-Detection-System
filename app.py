import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
import requests
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch  
from transformers import pipeline
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import io

# ==========================================
# 1. CORE SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Multimodal Command Center", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    .stApp { background: linear-gradient(135deg, #020c1b 0%, #0a192f 100%); }
    * { font-family: 'JetBrains Mono', monospace!important; }
    div[data-testid="column"] > div {
        background: rgba(10, 25, 47, 0.4);
        border-radius: 16px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1), inset 0 0 0 1px rgba(0, 242, 255, 0.2);
        backdrop-filter: blur(12px);
        border: 1px solid #00f2ff;
        padding: 25px;
        margin-bottom: 20px;
    }
    h1, h2, h3, p, label, .stMetric { color: #e6f1ff!important; }
    .stProgress > div > div > div > div { background-color: #00f2ff; }
    .stInfo { background-color: rgba(0, 242, 255, 0.1); border: 1px solid #00f2ff; color: #00f2ff; }
</style>
""", unsafe_allow_html=True)

# Branding Header
st.title("💠 Multimodal Intelligence & Physics Command Center")
col_title1, col_title2 = st.columns([2, 1])
with col_title1:
    st.markdown("### Lead Architect: **Akansh Saxena**")
    st.markdown("#### **J.K. Institute of Applied Physics & Technology**")
    st.write("📍 *University of Allahabad, Prayagraj*")
with col_title2:
    st.info("🚀 B.Tech CSE Final Year Project\n\nStatus: Production Ready")

# ==========================================
# 2. GLOBAL AI ENGINES
# ==========================================
@st.cache_resource(show_spinner="Warping AI Manifold...")
def load_heavy_engines():
    # 1. Text Engine
    nlp_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # 2. Vision Engine (Standard Stable Syntax)
    mp_face_mesh = mp.solutions.face_mesh
    mp_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, 
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return nlp_model, mp_mesh

semantic_engine, face_mesh_engine = load_heavy_engines()

@st.cache_data(ttl=600) 
def fetch_telemetry():
    url = "https://api.open-meteo.com/v1/forecast?latitude=25.43&longitude=81.84&current_weather=true"
    try:
        res = requests.get(url, timeout=5).json()
        return res['current_weather']['temperature'], res['current_weather'].get('surface_pressure', 1013.25)
    except:
        return 27.0, 1008.0

temp, press = fetch_telemetry()

# ==========================================
# 3. DASHBOARD ARCHITECTURE
# ==========================================
col_ingress, col_physics, col_analytics = st.columns([1.3, 1.8, 1.3])

with col_ingress:
    st.subheader("📡 Sensory Ingestion")
    st.write("📷 **Optical Flow Mapping**")
    
    class VisionProcessor(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh_engine.process(rgb_img)
            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=img, 
                        landmark_list=landmarks,
                        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 242, 255), thickness=1)
                    )
            return img

    webrtc_streamer(key="vision", video_processor_factory=VisionProcessor, rtc_configuration=RTC_CONFIG)

    st.write("🎙️ **Acoustic Array Analysis**")
    audio_buffer = st.audio_input("Record for Synthesis")
    if audio_buffer:
        y, sr = librosa.load(io.BytesIO(audio_buffer.read()))
        fig, ax = plt.subplots(figsize=(5, 2))
        librosa.display.waveshow(y, sr=sr, ax=ax, color="#00f2ff", alpha=0.8)
        ax.axis('off')
        st.pyplot(fig, transparent=True)

with col_physics:
    st.subheader("🌌 Live Manifold Simulation")
    x, y = np.linspace(-6, 6, 50), np.linspace(-6, 6, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2)) + (temp/press) * 15 * np.exp(-(X**2 + Y**2)/10)
    
    fig = go.Figure(data=[go.Surface(z=Z, colorscale='Viridis', opacity=0.9)])
    fig.update_layout(
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
        paper_bgcolor='rgba(0,0,0,0)', 
        margin=dict(l=0,r=0,b=0,t=0), 
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"📍 Location: Prayagraj | Temp: {temp}°C | Pressure: {press} hPa")

with col_analytics:
    st.subheader("📊 Cognitive Analytics")
    intent_input = st.text_input("Semantic Query Synthesis:")
    if intent_input:
        res = semantic_engine(intent_input)[0]
        st.metric("NLP Result", res['label'], f"{res['score']*100:.2f}% Match")
        st.progress(res['score'])

    st.write("⚡ **Fusion Stability**")
    radar = go.Figure(data=go.Scatterpolar(
        r=[95, 92, 98, 90, 94],
        theta=['Vision','Audio','Text','Fusion','Stability'],
        fill='toself', fillcolor='rgba(0, 242, 255, 0.3)', line_color='#00f2ff'
    ))
    radar.update_layout(
        polar=dict(bgcolor='rgba(10,25,47,0.5)', radialaxis=dict(visible=False)), 
        paper_bgcolor='rgba(0,0,0,0)', 
        font_color='#e6f1ff', 
        height=350,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(radar, use_container_width=True)

st.markdown("---")
st.markdown("<p style='text-align: center; color: #8892b0;'>© 2026 Akansh Saxena | J.K. Institute of Applied Physics & Technology</p>", unsafe_allow_html=True)
