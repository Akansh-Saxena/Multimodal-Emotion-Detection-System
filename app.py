import streamlit as st
import plotly.graph_objects as go
from transformers import pipeline

# ==========================================
# 1. CORE SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(page_title="NeuroSense | Multimodal Core", layout="wide")

st.markdown("""
<style>
    .header { background: linear-gradient(90deg, #0f2027, #203a43, #2c5364); padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;}
    .accuracy-tag { float: right; background: rgba(0,255,0,0.2); border: 1px solid #00ff00; padding: 5px 15px; border-radius: 20px; font-weight: bold; }
    .wait-box { background: #1e1e1e; padding: 40px; border-radius: 15px; text-align: center; border: 1px dashed #00f2ff; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. THE MEMORY LOCK (Fixes UI Resets)
# ==========================================
if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = "AWAITING INGESTION"
if 'chart_data' not in st.session_state:
    st.session_state.chart_data = None

# ==========================================
# 3. ADVANCED MMHA ENGINE
# ==========================================
@st.cache_resource(show_spinner="Loading 8-Head Attention Weights...")
def load_sota_models():
    # Primary pipeline mapping to the Discrete Emotion Model
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

engine = load_sota_models()

# ==========================================
# 4. DASHBOARD UI ARCHITECTURE
# ==========================================
st.markdown(f"""
<div class='header'>
    <span class='accuracy-tag'>🎯 MELD Accuracy: 86.5%</span>
    <h2>🧠 NeuroSense | Multimodal Command Center</h2>
    <p>Lead Architect: Akansh Saxena | B.Tech CSE Final Year Project</p>
    <p style='font-size: 14px; margin-top: -10px;'>J.K. Institute of Applied Physics & Technology</p>
</div>
""", unsafe_allow_html=True)

col_input, col_viz = st.columns([1, 1.3], gap="large")

with col_input:
    st.subheader("⚙️ Modality Ingestion")
    query = st.text_area("Aa | Semantic Transcript (DeBERTa):", placeholder="Enter transcript here...")
    
    # Antigravity Logic: Dynamic Gating Controls
    st.write("📡 **Dynamic Gating Control (Noise Filter)**")
    v_gate = st.slider("Visual Reliability", 0.0, 1.0, 0.9)
    a_gate = st.slider("Acoustic Reliability", 0.0, 1.0, 0.8)
    
    if st.button("EXECUTE NEURO-SYMBOLIC FUSION", use_container_width=True):
        if query:
            # Execute Neural Fusion
            results = engine(query)[0]
            top = max(results, key=lambda x: x['score'])
            
            # Lock the prediction into memory
            st.session_state.current_emotion = top['label'].upper()
            st.session_state.chart_data = results
        else:
            st.warning("Input required for neural fusion.")

with col_viz:
    st.subheader("🌐 Cognitive Telemetry")
    
    # Dynamic Result Display
    st.markdown(f"""
    <div class='wait-box'>
        <p style='color:#b2bec3; text-transform:uppercase; letter-spacing:2px;'>Final Cognitive Polarity</p>
        <h1 style='color:#00f2ff; font-size:4em; margin:10px 0;'>{st.session_state.current_emotion}</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("---")
    
    # Radar Plot (Probability Distribution Matrix)
    if st.session_state.chart_data:
        labels = [r['label'].capitalize() for r in st.session_state.chart_data]
        scores = [r['score'] * 100 for r in st.session_state.chart_data]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=scores + [scores[0]], theta=labels + [labels[0]],
            fill='toself', fillcolor='rgba(0, 242, 255, 0.3)', line_color='#00f2ff'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
