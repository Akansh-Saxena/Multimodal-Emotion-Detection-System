import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import requests
import json
import time

# --- INITIALIZATION & OPTIMIZATION ---
# Add this at the top of your app.py to stop the "slow" feeling
if 'initiated' not in st.session_state:
    st.session_state.initiated = True
    # Load your heavy AI models only ONCE here
    # Example: st.session_state.model = load_model()
    st.session_state.telemetry_data = {
        'Gravitational Stability': 0.85,
        'Quantum Decoherence': 0.12,
        'Exotic Matter Density': 0.45
    }

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Antigravity Engine Hub",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- MODERN DARK THEME CSS ---
# Inject custom CSS to create a glass-morphism effect, neon cyan borders (#00f2ff), and dark navy backgrounds.
st.markdown("""
<style>
    /* Global Base */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0a1128 0%, #010409 100%);
        color: #e0e6ed;
        font-family: 'Space Mono', 'Inter', monospace;
    }
    
    /* System Online Header */
    .system-online {
        position: absolute;
        top: 1rem;
        right: 2rem;
        color: #00f2ff;
        font-weight: 700;
        letter-spacing: 2px;
        text-shadow: 0 0 10px #00f2ff, 0 0 20px #00f2ff;
        display: flex;
        align-items: center;
        gap: 8px;
        z-index: 1000;
    }
    .system-online-dot {
        width: 10px;
        height: 10px;
        background-color: #00f2ff;
        border-radius: 50%;
        box-shadow: 0 0 10px #00f2ff, 0 0 20px #00f2ff;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 242, 255, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(0, 242, 255, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 242, 255, 0); }
    }

    /* Glowing Cards / Glassmorphism */
    .glowing-card {
        background: rgba(10, 17, 40, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 242, 255, 0.3);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5), inset 0 0 20px rgba(0, 242, 255, 0.05);
        transition: all 0.3s ease;
        margin-bottom: 20px;
    }
    .glowing-card:hover {
        border-color: rgba(0, 242, 255, 0.8);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5), 0 0 20px rgba(0, 242, 255, 0.2), inset 0 0 20px rgba(0, 242, 255, 0.1);
    }
    
    .card-title {
        color: #00f2ff;
        font-size: 1.2rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 15px;
        border-bottom: 1px solid rgba(0, 242, 255, 0.2);
        padding-bottom: 10px;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 0 0 15px rgba(255, 255, 255, 0.5);
    }

    /* Modality Zones */
    .zone-container {
        border: 1px solid rgba(0, 242, 255, 0.1);
        border-radius: 12px;
        background: rgba(1, 4, 9, 0.7);
        padding: 20px;
        margin-top: 10px;
    }
    .zone-header {
        font-size: 1rem;
        color: #a8b2d1;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Streamlit overrides */
    div[data-testid="stMetricValue"] { color: #00f2ff !important; text-shadow: 0 0 10px rgba(0,242,255,0.5); font-weight: bold; }
    h1, h2, h3 { color: #ffffff !important; font-weight: 700; }
    
</style>
""", unsafe_allow_html=True)

# "System Online" indicator
st.markdown("""
<div class="system-online">
    <div class="system-online-dot"></div>
    SYSTEM ONLINE
</div>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<h1 style='text-align: center; margin-top: 2rem; margin-bottom: 0;'>ANTIGRAVITY ENGINE HUB</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a8b2d1; letter-spacing: 3px; font-size: 0.9rem;'>NEUROSENSE COCKPIT INTERFACE /// SECTOR 7G</p>", unsafe_allow_html=True)
st.write("---")

# --- BACKEND FUSION: OPEN-METEO API ---
@st.cache_data(ttl=60) # Cache for 60 seconds
def fetch_weather_data():
    try:
        # Example coordinates (New York)
        url = "https://api.open-meteo.com/v1/forecast?latitude=40.7128&longitude=-74.0060&current=temperature_2m,surface_pressure,wind_speed_10m&timezone=auto"
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()
        return {
            "temp": data["current"]["temperature_2m"],
            "pressure": data["current"]["surface_pressure"],
            "wind": data["current"]["wind_speed_10m"]
        }
    except Exception as e:
        return {"temp": 22.5, "pressure": 1013.25, "wind": 5.0} # Fallback data

weather = fetch_weather_data()

# --- MAIN LAYOUT ---
col1, col2 = st.columns([1, 1], gap="large")

# --- LEFT COLUMN: TELEMETRY & METRICS ---
with col1:
    st.markdown('<div class="glowing-card"><div class="card-title">ENVIRONMENTAL FUSION</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Atmospheric Temp", f"{weather['temp']} °C")
    m2.metric("Manifold Pressure", f"{weather['pressure']} hPa")
    m3.metric("Flux Velocity", f"{weather['wind']} km/h")
    st.markdown('</div>', unsafe_allow_html=True)

    # Use st.fragment for instant updates on UI components
    @st.fragment
    def telemetry_radar():
        st.markdown('<div class="glowing-card"><div class="card-title">COGNITIVE TELEMETRY</div>', unsafe_allow_html=True)
        
        # Interactive slider to instantly warp radar
        intensity = st.slider("Neural Saturation Levels", 0.0, 1.0, 0.75, 0.05, key="radar_slider")
        
        # Radar Chart (Spider Chart)
        categories = ['Gravitational Stability', 'Quantum Decoherence', 'Exotic Matter Density']
        
        # Dynamic calculation based on "Instant Response" state
        base_vals = [st.session_state.telemetry_data[c] for c in categories]
        dynamic_vals = [min(1.0, v + (intensity * 0.2)) for v in base_vals]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=dynamic_vals + [dynamic_vals[0]], # Close the loop
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(0, 242, 255, 0.3)',
            line=dict(color='#00f2ff', width=3),
            marker=dict(color='#ffffff', size=8, symbol='diamond'),
            hoverinfo='r+theta'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1.2], gridcolor='rgba(0, 242, 255, 0.2)', linecolor='rgba(0, 242, 255, 0.2)', tickfont=dict(color='#a8b2d1')),
                angularaxis=dict(gridcolor='rgba(0, 242, 255, 0.2)', linecolor='rgba(0, 242, 255, 0.2)', tickfont=dict(color='#e0e6ed', size=12)),
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            margin=dict(t=20, b=20, l=40, r=40),
            height=350
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
    
    telemetry_radar()

# --- RIGHT COLUMN: 3D MANIFOLD & MODALITY ZONES ---
with col2:
    @st.fragment
    def manifold_graph():
        st.markdown('<div class="glowing-card"><div class="card-title">3D MANIFOLD WARP</div>', unsafe_allow_html=True)
        # Warp the manifold based on real Open-Meteo pressure/temp
        warp_factor = (weather['pressure'] / 1000.0) * (weather['temp'] / 20.0)
        
        # Generate surface data
        x = np.linspace(-5, 5, 50)
        y = np.linspace(-5, 5, 50)
        x_grid, y_grid = np.meshgrid(x, y)
        
        # Interactive offset
        phase = st.slider("Phase Shift", 0.0, 2*np.pi, 0.0, 0.1, key="manifold_phase")
        
        # Z = Sin(R)/R warped by real-world weather data
        r = np.sqrt(x_grid**2 + y_grid**2) + 0.1
        z = np.sin(r * warp_factor + phase) / r
        
        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Cyan', showscale=False)])
        fig.update_layout(
            scene=dict(
                xaxis=dict(showbackground=False, visible=False),
                yaxis=dict(showbackground=False, visible=False),
                zaxis=dict(showbackground=False, visible=False),
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=0, b=0, l=0, r=0),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    manifold_graph()

    # INPUT MODALITIES
    st.markdown('<div class="glowing-card"><div class="card-title">INPUT MODALITIES</div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["👁️ Visual Inference", "🧠 Semantic Control", "🎙️ Acoustic Processing"])
    
    with tabs[0]:
        st.markdown('<div class="zone-container">', unsafe_allow_html=True)
        st.markdown('<div class="zone-header">OPTICAL ARRAY INPUT</div>', unsafe_allow_html=True)
        st.camera_input("Engage Optical Sensor", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with tabs[1]:
        st.markdown('<div class="zone-container">', unsafe_allow_html=True)
        st.markdown('<div class="zone-header">SEMANTIC DIRECTIVE</div>', unsafe_allow_html=True)
        st.text_area("Input system directives...", height=100, label_visibility="collapsed", placeholder="Awaiting command...")
        st.button("EXECUTE NEURAL COMMAND", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with tabs[2]:
        st.markdown('<div class="zone-container">', unsafe_allow_html=True)
        st.markdown('<div class="zone-header">VOICE COMMAND COPILOT</div>', unsafe_allow_html=True)
        st.markdown("<p style='color: #a8b2d1; font-size: 0.9rem;'>Engage primary acoustic sensor for vocal intelligence interface.</p>", unsafe_allow_html=True)
        # Using Streamlit's new audio input (st.audio_input) which renders a mic icon!
        audio_val = st.audio_input("Record Voice Command", label_visibility="collapsed")
        if audio_val:
            st.success("Acoustic package received. Decrypting intent...")
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)
