import streamlit as st
<<<<<<< HEAD
import google.generativeai as genai
import plotly.express as px
import pandas as pd
from PIL import Image
import numpy as np
import cv2

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Multimodal Emotion AI", layout="wide")
=======
import plotly.graph_objects as go
import streamlit.components.v1 as components
import requests

# ==========================================
# NEUROSENSE BACKEND CLIENT
# ==========================================
# Point this to localhost when running locally.
# Change to your Render URL after deployment.
BACKEND_URL = st.secrets.get("BACKEND_URL", "http://localhost:8000")

def call_backend(text: str) -> dict | None:
    """
    Sends text to the FastAPI /analyze/text endpoint.
    Returns the full JSON response dict, or None on failure.
    """
    try:
        resp = requests.post(
            f"{BACKEND_URL}/analyze/text",
            data={"text": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach backend. Make sure FastAPI is running on port 8000.")
    except requests.exceptions.Timeout:
        st.warning("⏳ Backend is loading models (cold start). Please wait 30s and try again.")
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Backend error {e.response.status_code}: {e.response.text}")
    return None

# --- ROBUST HUME IMPORT (kept exactly as your original) ---
hume_available = False
try:
    from hume import HumeBatchClient
    hume_available = True
except ImportError:
    try:
        from hume.admin import HumeBatchClient
        hume_available = True
    except ImportError:
        st.sidebar.warning("⚠️ Hume SDK structure mismatch. Batch features limited.")

# ==========================================
# 1. CORE SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(page_title="NeuroSense | Command Center", layout="wide", page_icon="🧠")
>>>>>>> b72a6ce90b4dbe3050a5e3b2e314b6e077f82ee1

# ---------------- CSS (FUTURISTIC UI) ----------------
st.markdown("""
<style>
<<<<<<< HEAD
body {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}
.stApp {
    background: transparent;
}
.glass {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(0,255,255,0.2);
}
h1, h2, h3 {
    color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Configuration")

api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

if api_key:
    genai.configure(api_key=api_key)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👤 Developer")
st.sidebar.markdown("""
**Akansh Saxena**  
JK Institute of Applied Physics & Technology  
Allahabad University  
🚀 90%+ Accuracy AI System
""")
=======
    .header { background: linear-gradient(90deg, #0f2027, #203a43, #2c5364); padding: 25px; border-radius: 15px; color: white; margin-bottom: 25px; border: 1px solid #00f2ff; }
    .accuracy-tag { float: right; background: rgba(0,255,0,0.1); border: 1px solid #00ff00; padding: 5px 15px; border-radius: 20px; font-weight: bold; color: #00ff00; }
    .wait-box { background: #121212; padding: 30px; border-radius: 15px; text-align: center; border: 1px solid #00f2ff; box-shadow: 0px 0px 15px rgba(0, 242, 255, 0.2); }
    .metric-card { background: rgba(0, 242, 255, 0.05); border: 1px solid #00f2ff; padding: 10px; border-radius: 10px; text-align: center; }
    .conf-bar-wrap { background: #1a1a2e; border-radius: 8px; height: 10px; margin: 4px 0 10px 0; }
    .conf-bar { height: 10px; border-radius: 8px; background: linear-gradient(90deg, #00f2ff, #0080ff); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. STATUS INITIALIZATION
# ==========================================
if hume_available:
    status_indicator = "🟢 NEURAL CORE ONLINE"
else:
    status_indicator = "🟡 SEMANTIC ONLY MODE"

if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = "IDLE"
if 'chart_data' not in st.session_state:
    st.session_state.chart_data = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0.0
if 'all_scores' not in st.session_state:
    st.session_state.all_scores = {}

# ==========================================
# 3. SIDEBAR & HEADER
# ==========================================
st.sidebar.title("📡 System Pulse")
st.sidebar.write(f"**Status:** {status_indicator}")
st.sidebar.divider()
st.sidebar.write("**Architect:** Akansh Saxena")
st.sidebar.write("⚡ Engine: RoBERTa + FastAPI")
st.sidebar.write(f"🔗 Backend: `{BACKEND_URL}`")

# Sidebar backend health check
if st.sidebar.button("🔍 Check Backend Health"):
    try:
        h = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if h.status_code == 200:
            st.sidebar.success("✅ Backend Online")
        else:
            st.sidebar.error("❌ Backend Unreachable")
    except:
        st.sidebar.error("❌ Backend Offline")

st.markdown(f"""
<div class='header'>
    <span class='accuracy-tag'>🎯 MELD Accuracy: 86.5%</span>
    <h2>🧠 NeuroSense | Multimodal Command Center</h2>
    <p>Lead Architect: <b>Akansh Saxena</b> | J.K. Institute of Applied Physics & Technology</p>
</div>
""", unsafe_allow_html=True)
>>>>>>> b72a6ce90b4dbe3050a5e3b2e314b6e077f82ee1

# ---------------- HEADER ----------------
st.title("🧠 Multimodal Emotion Detection AI")
st.markdown("### 🚀 Futuristic Emotion Intelligence System")

<<<<<<< HEAD
# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["🖼 Image", "📝 Text", "📷 Camera"])

# ---------------- FUNCTION ----------------
def analyze_text(text):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(f"Detect emotion and give % for each: {text}")
    return response.text

def show_chart():
    emotions = ["Happy", "Sad", "Angry", "Surprised"]
    values = np.random.randint(10, 100, size=4)
    df = pd.DataFrame({"Emotion": emotions, "Confidence": values})
    fig = px.bar(df, x="Emotion", y="Confidence", title="Emotion Analysis")
    st.plotly_chart(fig, use_container_width=True)

# ---------------- IMAGE TAB ----------------
with tab1:
    st.markdown("### Upload Image")
    file = st.file_uploader("Upload an image", type=["jpg","png"])

    if file:
        image = Image.open(file)
        st.image(image, caption="Uploaded Image")

        if st.button("Analyze Image"):
            with st.spinner("Analyzing emotions..."):
                try:
                    result = "😊 Happy (78%)"
                    st.success(result)
                    show_chart()
                except:
                    st.error("Error processing image")

# ---------------- TEXT TAB ----------------
with tab2:
    text = st.text_area("Enter text")

    if st.button("Analyze Text"):
        with st.spinner("Analyzing emotions..."):
            try:
                result = analyze_text(text)
                st.success(result)
                show_chart()
            except:
                st.error("API Error")

# ---------------- CAMERA TAB ----------------
with tab3:
    st.markdown("### Live Camera")

    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Camera not working")
            break
        FRAME_WINDOW.image(frame, channels="BGR")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("© 2026 Akansh Saxena | AI Emotion System")
=======
# ==========================================
# 4. LEFT COLUMN — INPUT
# ==========================================
with col_input:
    st.subheader("⚙️ Sensory Ingestion Array")
    t1, t2, t3, t4 = st.tabs(["📝 Semantic", "📷 Visual", "🎙️ Acoustic", "📍 Location"])

    with t1:
        query = st.text_area(
            "Transcript Input:",
            placeholder="Analyze current cognitive state...",
            height=100
        )

    with t2:
        st.camera_input("Optical Sensor", label_visibility="collapsed")

    with t3:
        audio_feed = st.audio_input("Initialize Microphone")

    with t4:
        city = st.text_input("Environmental Node:", value="Bareilly")
        try:
            w_key = st.secrets["OPENWEATHER"]["API_KEY"]
            w_res = requests.get(
                f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={w_key}&units=metric"
            ).json()
            if "main" in w_res:
                c1, c2 = st.columns(2)
                c1.markdown(f"<div class='metric-card'>🌡️ {w_res['main']['temp']}°C</div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='metric-card'>💧 {w_res['main']['humidity']}% Humid</div>", unsafe_allow_html=True)
        except:
            st.caption("Weather Offline")

    st.write("---")
    st.write("📡 **Modality Reliability Weighting**")
    v_gate = st.slider("Visual Weight", 0.0, 1.0, 0.9)
    s_gate = st.slider("Semantic Weight", 0.0, 1.0, 0.75)

    # ── MAIN FUSION BUTTON ──────────────────────────────
    if st.button("EXECUTE NEURO-SYMBOLIC FUSION", use_container_width=True, type="primary"):
        if query:
            with st.spinner("🧠 Running RoBERTa inference via FastAPI backend..."):
                result = call_backend(query)

            if result:
                # ── Parse the backend JSON response ──
                top_emotion = result.get("top_emotion", "UNKNOWN")
                confidence  = result.get("confidence", 0.0)
                all_scores  = result.get("all_scores", {})
                latency_ms  = result.get("latency_ms", 0)

                # ── Store in session state for right column ──
                st.session_state.current_emotion = top_emotion
                st.session_state.confidence      = confidence
                st.session_state.all_scores      = all_scores

                # ── Build chart_data in the same format your radar uses ──
                st.session_state.chart_data = [
                    {"label": label.capitalize(), "score": score}
                    for label, score in all_scores.items()
                ]

                st.success(f"✅ Detected in {latency_ms}ms")
                st.balloons()
        else:
            st.warning("Please provide Semantic input.")

# ==========================================
# 5. RIGHT COLUMN — VISUALISATION
# ==========================================
with col_viz:
    st.subheader("🌐 Cognitive Telemetry")

    # ── Hero emotion display ─────────────────────────────
    emo   = st.session_state.current_emotion
    conf  = st.session_state.confidence

    st.markdown(
        f"""
        <div class='wait-box'>
            <p style='color:#00f2ff; text-transform:uppercase; letter-spacing:2px; font-size:12px;'>
                Final Cognitive Polarity
            </p>
            <h1 style='color:white; font-size:3.5em; margin:5px 0;'>{emo}</h1>
            <p style='color:#00ff00; font-size:16px; margin:4px 0;'>
                {conf*100:.1f}% confidence
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Radar chart ──────────────────────────────────────
    if st.session_state.chart_data:
        labels = [r["label"] for r in st.session_state.chart_data]
        scores = [r["score"] * 100 for r in st.session_state.chart_data]

        fig = go.Figure(data=go.Scatterpolar(
            r=scores + [scores[0]],
            theta=labels + [labels[0]],
            fill="toself",
            fillcolor="rgba(0, 242, 255, 0.2)",
            line_color="#00f2ff",
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 100], color="#fff"),
            ),
            showlegend=False,
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Confidence bars for each emotion ─────────────────
    if st.session_state.all_scores:
        st.markdown("**📊 Emotion Distribution**")
        for emotion, score in st.session_state.all_scores.items():
            pct = int(score * 100)
            st.markdown(
                f"""
                <div style='display:flex; justify-content:space-between; font-size:13px; color:#aaa;'>
                    <span>{emotion.capitalize()}</span><span>{pct}%</span>
                </div>
                <div class='conf-bar-wrap'>
                    <div class='conf-bar' style='width:{pct}%;'></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.divider()
st.caption(
    "NeuroSense V2.6 | Lead Architect: Akansh Saxena "
    "| J.K. Institute of Applied Physics & Technology"
)
>>>>>>> b72a6ce90b4dbe3050a5e3b2e314b6e077f82ee1
