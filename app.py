import streamlit as st
import plotly.graph_objects as go
import time
import requests

# ==========================================
# 1. EMBEDDED NEURAL ENGINE (BACKEND INTEGRATION)
# ==========================================
@st.cache_resource(show_spinner=False)
def load_emotion_model():
    """Loads the ML model directly into Streamlit's memory cache"""
    from transformers import pipeline
    # Using a fast, highly accurate DistilRoBERTa model trained specifically for emotions
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

def analyze_text(text: str) -> dict:
    """Processes text through the embedded ML model"""
    start_time = time.time()
    try:
        classifier = load_emotion_model()
        # Get results and format them
        raw_results = classifier(text)[0]
        
        # Convert list of dicts to a single dictionary {emotion: score}
        all_scores = {res['label']: res['score'] for res in raw_results}
        
        # Find the top emotion
        top_emotion = max(all_scores, key=all_scores.get).upper()
        confidence = all_scores[top_emotion.lower()]
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return {
            "top_emotion": top_emotion,
            "confidence": confidence,
            "all_scores": all_scores,
            "latency_ms": latency_ms
        }
    except Exception as e:
        st.error(f"❌ Neural Engine Error: {e}")
        return None

# ==========================================
# 2. CORE SYSTEM CONFIGURATION & SECRETS
# ==========================================
st.set_page_config(page_title="NeuroSense | Command Center", layout="wide", page_icon="🧠")

# Safely load data from Streamlit Secrets
architect_name = st.secrets.get("SYSTEM_PRESETS", {}).get("ARCHITECT", "Akansh Saxena")
accuracy_raw = st.secrets.get("SYSTEM_PRESETS", {}).get("MELD_ACCURACY", 0.92)
accuracy_str = f"{accuracy_raw * 100:.1f}%"

st.markdown("""
<style>
    .header { background: linear-gradient(90deg, #0f2027, #203a43, #2c5364); padding: 25px; border-radius: 15px; color: white; margin-bottom: 25px; border: 1px solid #00f2ff; }
    .accuracy-tag { float: right; background: rgba(0,255,0,0.1); border: 1px solid #00ff00; padding: 5px 15px; border-radius: 20px; font-weight: bold; color: #00ff00; }
    .wait-box { background: #121212; padding: 30px; border-radius: 15px; text-align: center; border: 1px solid #00f2ff; box-shadow: 0px 0px 15px rgba(0, 242, 255, 0.2); }
    .metric-card { background: rgba(0, 242, 255, 0.05); border: 1px solid #00f2ff; padding: 10px; border-radius: 10px; text-align: center; }
    .conf-bar-wrap { background: #1a1a2e; border-radius: 8px; height: 10px; margin: 4px 0 10px 0; }
    .conf-bar { height: 10px; border-radius: 8px; background: linear-gradient(90deg, #00f2ff, #0080ff); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. STATE INITIALIZATION
# ==========================================
if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = "IDLE"
if 'chart_data' not in st.session_state:
    st.session_state.chart_data = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0.0
if 'all_scores' not in st.session_state:
    st.session_state.all_scores = {}

# ==========================================
# 4. SIDEBAR & HEADER
# ==========================================
st.sidebar.title("📡 System Pulse")
st.sidebar.write("**Status:** 🟢 NEURAL CORE ONLINE")
st.sidebar.divider()
# Dynamically injecting your name from Secrets!
st.sidebar.write(f"**Architect:** {architect_name}")
st.sidebar.write("⚡ Engine: Embedded DistilRoBERTa")
st.sidebar.write("🔗 Architecture: Unified Monolith")

st.markdown(f"""
<div class='header'>
    <span class='accuracy-tag'>🎯 MELD Accuracy: {accuracy_str}</span>
    <h2>🧠 NeuroSense | Multimodal Command Center</h2>
    <p>Lead Architect: <b>{architect_name}</b> | J.K. Institute of Applied Physics & Technology</p>
</div>
""", unsafe_allow_html=True)

col_input, col_viz = st.columns([1.2, 1], gap="large")

# ==========================================
# 5. LEFT COLUMN — INPUT
# ==========================================
with col_input:
    st.subheader("⚙️ Sensory Ingestion Array")
    # Restored the Location Tab
    t1, t2, t3, t4 = st.tabs(["📝 Semantic", "📷 Visual", "🎙️ Acoustic", "📍 Location"])

    with t1:
        query = st.text_area(
            "Transcript Input:",
            placeholder="Type a sentence to analyze cognitive state...",
            height=100
        )

    with t2:
        st.camera_input("Optical Sensor (UI Preview)", label_visibility="collapsed")

    with t3:
        st.audio_input("Microphone (UI Preview)")

    with t4:
        city = st.text_input("Environmental Node:", value="Bareilly")
        try:
            # Safely check for OpenWeather API key
            if "OPENWEATHER" in st.secrets and "API_KEY" in st.secrets["OPENWEATHER"]:
                w_key = st.secrets["OPENWEATHER"]["API_KEY"]
                w_res = requests.get(
                    f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={w_key}&units=metric"
                ).json()
                if "main" in w_res:
                    c1, c2 = st.columns(2)
                    c1.markdown(f"<div class='metric-card'>🌡️ {w_res['main']['temp']}°C</div>", unsafe_allow_html=True)
                    c2.markdown(f"<div class='metric-card'>💧 {w_res['main']['humidity']}% Humid</div>", unsafe_allow_html=True)
            else:
                st.caption("Weather Offline: API Key not found in Streamlit Secrets.")
        except Exception as e:
            st.caption(f"Weather Offline: {e}")

    st.write("---")
    st.write("📡 **Modality Reliability Weighting**")
    v_gate = st.slider("Visual Weight", 0.0, 1.0, 0.9)
    s_gate = st.slider("Semantic Weight", 0.0, 1.0, 0.75)

    # ── MAIN FUSION BUTTON ──────────────────────────────
    if st.button("EXECUTE NEURO-SYMBOLIC FUSION", use_container_width=True, type="primary"):
        if query:
            with st.spinner("🧠 Booting Neural Engine & Processing (First run takes 10s)..."):
                result = analyze_text(query)

            if result:
                st.session_state.current_emotion = result["top_emotion"]
                st.session_state.confidence      = result["confidence"]
                st.session_state.all_scores      = result["all_scores"]

                st.session_state.chart_data = [
                    {"label": label.capitalize(), "score": score}
                    for label, score in result["all_scores"].items()
                ]

                st.success(f"✅ Cognitive State Detected in {result['latency_ms']}ms")
                st.balloons()
        else:
            st.warning("Please provide Semantic input.")

# ==========================================
# 6. RIGHT COLUMN — VISUALISATION
# ==========================================
with col_viz:
    st.subheader("🌐 Cognitive Telemetry")

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
                bgcolor
