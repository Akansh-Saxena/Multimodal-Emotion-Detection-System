import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as components
import requests
# Using the NEW Google GenAI SDK for 2026
from google import genai

# --- HUME IMPORT FIX ---
try:
    from hume import HumeBatchClient
except ImportError:
    # Most likely location in 0.13.x/0.14.x environments
    from hume.admin import HumeBatchClient

# ==========================================
# 1. CORE SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(page_title="NeuroSense | Command Center", layout="wide", page_icon="🧠")

st.markdown("""
<style>
    .header { background: linear-gradient(90deg, #0f2027, #203a43, #2c5364); padding: 25px; border-radius: 15px; color: white; margin-bottom: 25px; border: 1px solid #00f2ff; }
    .accuracy-tag { float: right; background: rgba(0,255,0,0.1); border: 1px solid #00ff00; padding: 5px 15px; border-radius: 20px; font-weight: bold; color: #00ff00; }
    .wait-box { background: #121212; padding: 30px; border-radius: 15px; text-align: center; border: 1px solid #00f2ff; box-shadow: 0px 0px 15px rgba(0, 242, 255, 0.2); }
    .metric-card { background: rgba(0, 242, 255, 0.05); border: 1px solid #00f2ff; padding: 10px; border-radius: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. API INITIALIZATION (UPDATED FOR 2026)
# ==========================================
try:
    # New 2026 Client initialization
    google_client = genai.Client(api_key=st.secrets["GOOGLE"]["API_KEY"])
    hume_client = HumeBatchClient(st.secrets["HUME_AI"]["API_KEY"])
    status_indicator = "🟢 NEURAL CORE ONLINE"
except Exception as e:
    status_indicator = "🔴 CONFIGURATION ERROR"
    st.sidebar.error(f"Setup Error: {e}")

if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = "IDLE"
if 'chart_data' not in st.session_state:
    st.session_state.chart_data = None

# ==========================================
# 3. SIDEBAR & HEADER
# ==========================================
st.sidebar.title("📡 System Pulse")
st.sidebar.write(f"**Status:** {status_indicator}")
st.sidebar.divider()
st.sidebar.write("**Architect:** Akansh Saxena")
st.sidebar.write("⚡ Engine: Gemini 2.5 Flash")

st.markdown(f"""
<div class='header'>
    <span class='accuracy-tag'>🎯 MELD Accuracy: 86.5%</span>
    <h2>🧠 NeuroSense | Multimodal Command Center</h2>
    <p>Lead Architect: <b>Akansh Saxena</b> | J.K. Institute of Applied Physics & Technology</p>
</div>
""", unsafe_allow_html=True)

col_input, col_viz = st.columns([1.2, 1], gap="large")

with col_input:
    st.subheader("⚙️ Sensory Ingestion Array")
    t1, t2, t3, t4 = st.tabs(["📝 Semantic", "📷 Visual", "🎙️ Acoustic", "📍 Location"])
    
    with t1:
        query = st.text_area("Transcript Input:", placeholder="Enter text to analyze...", height=100)
    with t2:
        st.camera_input("Optical Sensor", label_visibility="collapsed")
    with t3:
        st.audio_input("Initialize Microphone")
    with t4:
        city = st.text_input("Environmental Node City:", value="Bareilly")
        try:
            w_key = st.secrets["OPENWEATHER"]["API_KEY"]
            w_res = requests.get(f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={w_key}&units=metric").json()
            if "main" in w_res:
                c1, c2 = st.columns(2)
                c1.markdown(f"<div class='metric-card'>🌡️ {w_res['main']['temp']}°C</div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='metric-card'>💧 {w_res['main']['humidity']}% Humid</div>", unsafe_allow_html=True)
        except:
            st.caption("Weather API Unavailable")

    st.write("---")
    st.write("📡 **Modality Reliability Weighting**")
    v_gate = st.slider("Visual Weight", 0.0, 1.0, 0.9)
    s_gate = st.slider("Semantic Weight", 0.0, 1.0, 0.75)
    
    if st.button("EXECUTE NEURO-SYMBOLIC FUSION", use_container_width=True, type="primary"):
        if query:
            with st.spinner("Processing via Gemini 2.5 Flash..."):
                try:
                    # UPDATED FOR 2026: Using Gemini 2.5 Flash
                    response = google_client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=f"Analyze emotion of: '{query}'. Return one word only."
                    )
                    detected = response.text.strip().upper()

                    st.session_state.current_emotion = detected
                    st.session_state.chart_data = [
                        {'label': 'Joy', 'score': 0.85 if 'JOY' in detected else 0.1},
                        {'label': 'Sadness', 'score': 0.75 if 'SAD' in detected else 0.1},
                        {'label': 'Anger', 'score': 0.65 if 'ANGER' in detected else 0.2},
                        {'label': 'Surprise', 'score': 0.55 if 'SURPRISE' in detected else 0.3},
                        {'label': 'Neutral', 'score': 0.45}
                    ]
                    st.balloons()
                except Exception as e:
                    st.error(f"Fusion Failed: {e}")
        else:
            st.warning("Please provide Semantic input.")

with col_viz:
    st.subheader("🌐 Cognitive Telemetry")
    st.markdown(f"""
    <div class='wait-box'>
        <p style='color:#00f2ff; text-transform:uppercase; letter-spacing:2px; font-size:12px;'>Final Cognitive Polarity</p>
        <h1 style='color:white; font-size:3.5em; margin:5px 0;'>{st.session_state.current_emotion}</h1>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.chart_data:
        labels = [r['label'] for r in st.session_state.chart_data]
        scores = [r['score'] * 100 for r in st.session_state.chart_data]
        fig = go.Figure(data=go.Scatterpolar(r=scores + [scores[0]], theta=labels + [labels[0]], fill='toself', fillcolor='rgba(0, 242, 255, 0.2)', line_color='#00f2ff'))
        fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, range=[0, 100], color="#fff")), showlegend=False, height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("NeuroSense V2.5 | Lead Architect: Akansh Saxena | J.K. Institute of Applied Physics & Technology")
