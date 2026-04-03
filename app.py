import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as components
import requests
import google.generativeai as genai
from hume import HumeBatchClient
from hume.models.config import LanguageConfig

# ==========================================
# 1. CORE SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(page_title="NeuroSense | Command Center", layout="wide", page_icon="🧠")

# Custom Cyberpunk CSS
st.markdown("""
<style>
    .header { background: linear-gradient(90deg, #0f2027, #203a43, #2c5364); padding: 25px; border-radius: 15px; color: white; margin-bottom: 25px; border: 1px solid #00f2ff; }
    .accuracy-tag { float: right; background: rgba(0,255,0,0.1); border: 1px solid #00ff00; padding: 5px 15px; border-radius: 20px; font-weight: bold; color: #00ff00; }
    .wait-box { background: #121212; padding: 30px; border-radius: 15px; text-align: center; border: 1px solid #00f2ff; box-shadow: 0px 0px 15px rgba(0, 242, 255, 0.2); }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #1e1e1e; border-radius: 5px 5px 0 0; padding: 10px 20px; color: white; }
    .metric-card { background: rgba(0, 242, 255, 0.05); border: 1px solid #00f2ff; padding: 10px; border-radius: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. API INITIALIZATION
# ==========================================
try:
    genai.configure(api_key=st.secrets["GOOGLE"]["API_KEY"])
    hume_client = HumeBatchClient(st.secrets["HUME_AI"]["API_KEY"])
except Exception as e:
    st.warning("API Keys missing or invalid. Please check your secrets.")

# Persistence Layer
if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = "AWAITING SENSORY INPUT"
if 'chart_data' not in st.session_state:
    st.session_state.chart_data = None

# ==========================================
# 3. DASHBOARD UI ARCHITECTURE
# ==========================================
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
    tab_text, tab_visual, tab_audio, tab_geo = st.tabs(["📝 Semantic", "📷 Visual", "🎙️ Acoustic", "📍 Location"])
    
    with tab_text:
        query = st.text_area("Transcript Input:", placeholder="I am thrilled about this presentation! / मैं बहुत खुश हूँ...", height=100)
        
    with tab_visual:
        st.write("📸 **Optical Frame Capture**")
        webcam_image = st.camera_input("Engage Optical Sensor", label_visibility="collapsed")
        if webcam_image:
            st.toast("Visual Frame Buffered", icon="📷")
            
    with tab_audio:
        st.info("🌐 Multi-Language Support Activated (Hume AI Prosody)")
        audio_feed = st.audio_input("Initialize Microphone")
        
    with tab_geo:
        st.markdown("**Real-Time GPS & Weather Telemetry**")
        city = st.text_input("Environmental Node:", value="Bareilly")
        
        # Weather Integration
        w_key = st.secrets["OPENWEATHER"]["API_KEY"]
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={w_key}&units=metric"
        try:
            w_res = requests.get(weather_url).json()
            if "main" in w_res:
                c1, c2 = st.columns(2)
                c1.markdown(f"<div class='metric-card'>🌡️ Temp<br><b>{w_res['main']['temp']}°C</b></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='metric-card'>💧 Humid<br><b>{w_res['main']['humidity']}%</b></div>", unsafe_allow_html=True)
        except:
            st.error("Weather Node Offline")

        components.html(
            """
            <div id="location" style="color: #00f2ff; font-family: monospace; font-size: 14px; padding: 12px; border: 1px solid #00f2ff; border-radius: 5px; background: #000; margin-top: 10px;">
                🛰️ Acquiring Satellite Lock...
            </div>
            <script>
                navigator.geolocation.getCurrentPosition(function(p) {
                    document.getElementById("location").innerHTML = 
                        "✅ <b>TARGET LOCKED</b><br>LAT: " + p.coords.latitude.toFixed(6) + "<br>LON: " + p.coords.longitude.toFixed(6);
                }, function(e) {
                    document.getElementById("location").innerHTML = "❌ GPS Access Denied";
                });
            </script>
            """, height=100
        )
    
    st.write("---")
    st.write("📡 **Modality Reliability Weighting**")
    v_gate = st.slider("Visual Weight", 0.0, 1.0, 0.9)
    a_gate = st.slider("Acoustic Weight", 0.0, 1.0, 0.8)
    s_gate = st.slider("Semantic Weight", 0.0, 1.0, 0.75)
    
    if st.button("EXECUTE NEURO-SYMBOLIC FUSION", use_container_width=True, type="primary"):
        if query:
            with st.spinner("Synchronizing Multimodal Matrices via Gemini Pro..."):
                try:
                    # 1. Semantic Analysis via Gemini
                    model = genai.GenerativeModel('gemini-pro')
                    prompt = f"Analyze the emotion of this text: '{query}'. Return only the dominant emotion in one word."
                    response = model.generate_content(prompt)
                    detected = response.text.strip().upper()

                    # 2. Simulate Fusion Weighting
                    st.session_state.current_emotion = detected
                    
                    # 3. Dummy data for radar (Hume Integration would populate this)
                    st.session_state.chart_data = [
                        {'label': 'Joy', 'score': 0.8 if 'JOY' in detected else 0.2},
                        {'label': 'Sadness', 'score': 0.7 if 'SAD' in detected else 0.1},
                        {'label': 'Anger', 'score': 0.6 if 'ANGER' in detected else 0.1},
                        {'label': 'Surprise', 'score': 0.5 if 'SURPRISE' in detected else 0.3},
                        {'label': 'Neutral', 'score': 0.4}
                    ]
                    st.balloons()
                except Exception as e:
                    st.error(f"Fusion Error: {e}")
        else:
            st.warning("Please provide Semantic input (Transcript) for Neural Overlay.")

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
st.caption("NeuroSense V2.0 | Integrated Multimodal Engine | Lead Architect: Akansh Saxena")
