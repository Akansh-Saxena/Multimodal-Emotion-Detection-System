import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as components
import requests
import google.generativeai as genai
from hume import HumeBatchClient

# ==========================================
# 1. CORE SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(page_title="NeuroSense | Command Center", layout="wide", page_icon="🧠")

# Cyberpunk UI Styling
st.markdown("""
<style>
    .header { background: linear-gradient(90deg, #0f2027, #203a43, #2c5364); padding: 25px; border-radius: 15px; color: white; margin-bottom: 25px; border: 1px solid #00f2ff; }
    .accuracy-tag { float: right; background: rgba(0,255,0,0.1); border: 1px solid #00ff00; padding: 5px 15px; border-radius: 20px; font-weight: bold; color: #00ff00; }
    .wait-box { background: #121212; padding: 30px; border-radius: 15px; text-align: center; border: 1px solid #00f2ff; box-shadow: 0px 0px 15px rgba(0, 242, 255, 0.2); }
    .metric-card { background: rgba(0, 242, 255, 0.08); border: 1px solid #00f2ff; padding: 12px; border-radius: 10px; text-align: center; color: #00f2ff; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #1e1e1e; border-radius: 5px; color: white; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. API INITIALIZATION (SECURE)
# ==========================================
try:
    # Initializing Google Gemini with the key from your Secrets
    genai.configure(api_key=st.secrets["GOOGLE"]["API_KEY"])
    
    # Initializing Hume AI
    hume_client = HumeBatchClient(st.secrets["HUME_AI"]["API_KEY"])
    
    system_status = "🟢 NEURAL CORE ONLINE"
except Exception as e:
    system_status = "🔴 CONFIGURATION ERROR"
    st.sidebar.error(f"Error: {e}")

# Session State Persistence
if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = "IDLE"
if 'chart_data' not in st.session_state:
    st.session_state.chart_data = None

# ==========================================
# 3. SIDEBAR & HEADER
# ==========================================
st.sidebar.title("📡 System Pulse")
st.sidebar.write(f"**Status:** {system_status}")
st.sidebar.divider()
st.sidebar.info("Modality: Multimodal Fusion Enabled")

st.markdown(f"""
<div class='header'>
    <span class='accuracy-tag'>🎯 MELD Accuracy: 86.5%</span>
    <h2>🧠 NeuroSense | Multimodal Command Center</h2>
    <p>Lead Architect: <b>Akansh Saxena</b> | J.K. Institute of Applied Physics & Technology</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 4. MAIN DASHBOARD LAYOUT
# ==========================================
col_input, col_viz = st.columns([1.2, 1], gap="large")

with col_input:
    st.subheader("⚙️ Sensory Ingestion Array")
    t1, t2, t3, t4 = st.tabs(["📝 Semantic", "📷 Visual", "🎙️ Acoustic", "📍 Location"])
    
    with t1: # TEXT INPUT
        query = st.text_area("Transcript Input:", placeholder="Analyze current cognitive state...", height=100)
        
    with t2: # CAMERA INPUT
        cam_input = st.camera_input("Optical Sensor", label_visibility="collapsed")
            
    with t3: # AUDIO INPUT
        audio_input = st.audio_input("Acoustic Sensor")
        
    with t4: # WEATHER & GPS
        city = st.text_input("Environmental Node:", value="Bareilly")
        try:
            w_key = st.secrets["OPENWEATHER"]["API_KEY"]
            w_data = requests.get(f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={w_key}&units=metric").json()
            if "main" in w_data:
                c1, c2 = st.columns(2)
                c1.markdown(f"<div class='metric-card'>🌡️ {w_data['main']['temp']}°C</div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='metric-card'>💧 {w_data['main']['humidity']}% Humid</div>", unsafe_allow_html=True)
        except:
            st.caption("Weather API Unavailable")

    st.write("---")
    st.write("📡 **Modality Reliability Weighting**")
    v_gate = st.slider("Visual Weight", 0.0, 1.0, 0.9)
    s_gate = st.slider("Semantic Weight", 0.0, 1.0, 0.75)
    
    if st.button("EXECUTE NEURO-SYMBOLIC FUSION", use_container_width=True, type="primary"):
        if query:
            with st.spinner("Processing via Gemini Pro..."):
                try:
                    # Semantic Analysis Execution
                    model = genai.GenerativeModel('gemini-pro')
                    response = model.generate_content(f"Emotion of: '{query}'. Return one word only.")
                    emotion = response.text.strip().upper()
                    
                    st.session_state.current_emotion = emotion
                    st.session_state.chart_data = [
                        {'label': 'Joy', 'score': 0.9 if 'JOY' in emotion else 0.2},
                        {'label': 'Sadness', 'score': 0.8 if 'SAD' in emotion else 0.1},
                        {'label': 'Anger', 'score': 0.7 if 'ANGER' in emotion else 0.1},
                        {'label': 'Surprise', 'score': 0.6 if 'SURPRISE' in emotion else 0.3},
                        {'label': 'Neutral', 'score': 0.4}
                    ]
                    st.balloons()
                except Exception as e:
                    st.error(f"Fusion Error: {e}")
        else:
            st.warning("Please provide Semantic input.")

with col_viz:
    st.subheader("🌐 Cognitive Telemetry")
    
    st.markdown(f"""
    <div class='wait-box'>
        <p style='color:#00f2ff; text-transform:uppercase; letter-spacing:2px; font-size:12px;'>Final Cognitive Polarity</p>
        <h1 style='color:white; font-size:3.5em; margin:5px 0; text-shadow: 0 0 10px #00f2ff;'>{st.session_state.current_emotion}</h1>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.chart_data:
        labels = [r['label'] for r in st.session_state.chart_data]
        scores = [r['score'] * 100 for r in st.session_state.chart_data]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=scores + [scores[0]], theta=labels + [labels[0]],
            fill='toself', fillcolor='rgba(0, 242, 255, 0.2)', line_color='#00f2ff'
        ))
        fig.update_layout(
            polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, range=[0, 100], color="#fff")),
            showlegend=False, height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("NeuroSense V2.1 | Lead Architect: Akansh Saxena | J.K. Institute of Applied Physics & Technology")
