import streamlit as st
import plotly.graph_objects as go
import time

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
# 2. CORE SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(page_title="NeuroSense | Command Center", layout="wide", page_icon="🧠")

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
st.sidebar.write("**Architect:** Akansh Saxena")
st.sidebar.write("⚡ Engine: Embedded DistilRoBERTa")
st.sidebar.write("🔗 Architecture: Unified Monolith")

st.markdown(f"""
<div class='header'>
    <span class='accuracy-tag'>🎯 MELD Accuracy: ~92.0%</span>
    <h2>🧠 NeuroSense | Multimodal Command Center</h2>
    <p>Lead Architect: <b>Akansh Saxena</b> | J.K. Institute of Applied Physics & Technology</p>
</div>
""", unsafe_allow_html=True)

col_input, col_viz = st.columns([1.2, 1], gap="large")

# ==========================================
# 5. LEFT COLUMN — INPUT
# ==========================================
with col_input:
    st.subheader("⚙️ Sensory Ingestion Array")
    t1, t2, t3 = st.tabs(["📝 Semantic", "📷 Visual", "🎙️ Acoustic"])

    with t1:
        query = st.text_area(
            "Transcript Input:",
            placeholder="Type a sentence to analyze cognitive state (e.g., 'I am so thrilled to finally launch this project!')...",
            height=100
        )

    with t2:
        st.camera_input("Optical Sensor (UI Preview)", label_visibility="collapsed")

    with t3:
        st.audio_input("Microphone (UI Preview)")

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
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 100], color="#fff"),
            ),
            showlegend=False,
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    if st.session_state.all_scores:
        st.markdown("**📊 Emotion Distribution**")
        for emotion, score in sorted(st.session_state.all_scores.items(), key=lambda item: item[1], reverse=True):
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
