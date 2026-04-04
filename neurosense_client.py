"""
============================================================
NeuroSense AI — Streamlit Frontend Integration Module
============================================================
Author  : Akansh Saxena
Institute: JK Institute of Applied Physics & Technology,
           University of Allahabad, Prayagraj
============================================================
HOW TO USE:
  Drop this file into your project root alongside app.py.
  Then inside app.py add:
      from neurosense_client import analyze_text, analyze_audio, analyze_multimodal, render_results_panel
============================================================
"""

import io
import time
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────────────────────
# CONFIGURATION — Change BACKEND_URL to your Render service URL
# ─────────────────────────────────────────────────────────────
BACKEND_URL = st.secrets.get(
    "BACKEND_URL",
    "http://localhost:8000",  # Default: local dev
)
# In production, set this in Streamlit Cloud: Settings → Secrets
# [secrets]
# BACKEND_URL = "https://neurosense-api.onrender.com"

REQUEST_TIMEOUT = 60  # seconds (model inference can be slow on cold start)


# ╔══════════════════════════════════════════════════════════╗
# ║  1. API CLIENT FUNCTIONS                                ║
# ╚══════════════════════════════════════════════════════════╝

def analyze_text(text: str) -> dict | None:
    """
    Send text to the /analyze/text endpoint.

    Data flow:
      Streamlit form → POST multipart/form-data → FastAPI → RoBERTa → JSON response
    
    Returns:
        dict with keys: top_emotion, confidence, all_scores, latency_ms
        None on network/server error
    """
    try:
        with st.spinner("🧠 Running RoBERTa inference…"):
            response = requests.post(
                f"{BACKEND_URL}/analyze/text",
                data={"text": text},           # FastAPI Form() expects form-data
                timeout=REQUEST_TIMEOUT,
            )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach the backend. Is the FastAPI server running?")
    except requests.exceptions.Timeout:
        st.warning("⏳ Request timed out. The model may still be loading (cold start). Try again in 30s.")
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Backend returned error {e.response.status_code}: {e.response.text}")
    return None


def analyze_audio(audio_bytes: bytes, filename: str = "audio.wav") -> dict | None:
    """
    Send audio bytes to the /analyze/audio endpoint.

    Data flow:
      Streamlit audio_input → bytes → POST multipart file upload
      → FastAPI → librosa decode → Wav2Vec2 → JSON response
    
    Args:
        audio_bytes : raw bytes from st.audio_input() or file uploader
        filename    : hint for MIME detection (e.g. 'clip.mp3')
    
    Returns:
        dict with keys: top_emotion, confidence, all_scores, latency_ms
        None on error
    """
    try:
        with st.spinner("🎤 Running Wav2Vec2 speech-emotion inference…"):
            response = requests.post(
                f"{BACKEND_URL}/analyze/audio",
                files={"audio": (filename, io.BytesIO(audio_bytes), "audio/wav")},
                timeout=REQUEST_TIMEOUT,
            )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach the backend.")
    except requests.exceptions.Timeout:
        st.warning("⏳ Audio inference timed out. Try a shorter clip (<15s).")
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Backend error {e.response.status_code}: {e.response.text}")
    return None


def analyze_multimodal(
    text: str | None = None,
    audio_bytes: bytes | None = None,
    text_weight: float = 0.55,
) -> dict | None:
    """
    Call the /analyze/multimodal fusion endpoint.
    Accepts text, audio, or both.

    Data flow:
      → POST multipart form with both text + audio file
      → FastAPI runs both models independently
      → Weighted late-fusion merges predictions
      → Returns fused result + per-modality sub-results

    Args:
        text         : transcript string (optional)
        audio_bytes  : raw audio bytes (optional)
        text_weight  : fusion weight for text branch (0.0–1.0)
    """
    if not text and not audio_bytes:
        st.warning("Provide at least one input (text or audio).")
        return None

    files = {}
    data = {"text_weight": str(text_weight)}

    if text:
        data["text"] = text
    if audio_bytes:
        files["audio"] = ("recording.wav", io.BytesIO(audio_bytes), "audio/wav")

    try:
        with st.spinner("⚡ Running multimodal fusion inference…"):
            response = requests.post(
                f"{BACKEND_URL}/analyze/multimodal",
                data=data,
                files=files if files else None,
                timeout=REQUEST_TIMEOUT,
            )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach the backend.")
    except requests.exceptions.Timeout:
        st.warning("⏳ Multimodal inference timed out.")
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Backend error: {e.response.text}")
    return None


# ╔══════════════════════════════════════════════════════════╗
# ║  2. DISPLAY COMPONENTS                                  ║
# ╚══════════════════════════════════════════════════════════╝

# Colour palette per emotion — cyberpunk neon theme
EMOTION_COLOURS = {
    "JOY":      "#FFD700",
    "SADNESS":  "#00BFFF",
    "ANGER":    "#FF4500",
    "FEAR":     "#DA70D6",
    "DISGUST":  "#32CD32",
    "SURPRISE": "#FF8C00",
    "NEUTRAL":  "#00F2FF",
}

EMOTION_EMOJIS = {
    "JOY":      "😄",
    "SADNESS":  "😢",
    "ANGER":    "😡",
    "FEAR":     "😨",
    "DISGUST":  "🤢",
    "SURPRISE": "😲",
    "NEUTRAL":  "😐",
}


def render_hero_emotion(result: dict):
    """
    Render the large, glowing 'top emotion' hero card.
    """
    emo = result.get("top_emotion", "UNKNOWN")
    conf = result.get("confidence", 0.0)
    latency = result.get("latency_ms", 0)
    modality = result.get("modality", "")
    emoji = EMOTION_EMOJIS.get(emo, "🧠")
    color = EMOTION_COLOURS.get(emo, "#00F2FF")

    st.markdown(
        f"""
        <div style="
            background: rgba(0,0,0,0.5);
            border: 2px solid {color};
            border-radius: 18px;
            padding: 30px;
            text-align: center;
            box-shadow: 0 0 30px {color}55;
            margin: 10px 0;
        ">
            <p style="color:{color}; font-size:12px; letter-spacing:4px; text-transform:uppercase; margin:0;">
                DETECTED EMOTION · {modality.replace('_',' ').upper()}
            </p>
            <div style="font-size:72px; margin: 10px 0;">{emoji}</div>
            <h1 style="color:white; font-size:48px; margin:0; font-weight:900;">{emo}</h1>
            <p style="color:{color}; font-size:22px; margin:8px 0;">
                {conf*100:.1f}% confidence
            </p>
            <p style="color:#555; font-size:12px; margin:0;">
                Inference: {latency} ms
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_radar_chart(all_scores: dict) -> go.Figure:
    """
    Render a polar/radar chart of the full emotion probability distribution.
    Returns a Plotly figure object.
    """
    labels = list(all_scores.keys())
    values = [v * 100 for v in all_scores.values()]

    # Close the polygon
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill="toself",
            fillcolor="rgba(0, 242, 255, 0.15)",
            line=dict(color="#00F2FF", width=2),
            marker=dict(color="#00F2FF", size=6),
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color="#aaa"),
                gridcolor="#222",
            ),
            angularaxis=dict(
                tickfont=dict(color="#e6f1ff", size=13),
                gridcolor="#222",
            ),
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=30, b=30),
        height=380,
    )
    return fig


def render_bar_chart(all_scores: dict) -> go.Figure:
    """
    Render a horizontal bar chart of emotion probabilities.
    Bars are color-coded per emotion.
    """
    labels = list(all_scores.keys())
    values = [v * 100 for v in all_scores.values()]
    colors = [EMOTION_COLOURS.get(l, "#00F2FF") for l in labels]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, opacity=0.85),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(color="white"),
        )
    )
    fig.update_layout(
        xaxis=dict(
            range=[0, 105],
            tickfont=dict(color="#aaa"),
            gridcolor="#222",
            title=dict(text="Confidence (%)", font=dict(color="#aaa")),
        ),
        yaxis=dict(tickfont=dict(color="#e6f1ff", size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=30, t=20, b=20),
        height=300,
    )
    return fig


def render_fusion_breakdown(result: dict):
    """
    For multimodal results, show the text vs audio sub-result breakdown
    and the fusion weights applied.
    """
    sub = result.get("sub_results", {})
    weights = result.get("fusion_weights", {})

    if not sub.get("text") and not sub.get("audio"):
        return

    st.markdown("#### 🔀 Fusion Breakdown")
    col_t, col_a = st.columns(2)

    with col_t:
        if sub.get("text"):
            t = sub["text"]
            st.metric(
                label=f"📝 Text Branch ({weights.get('text', 0)*100:.0f}% weight)",
                value=t["top_emotion"],
                delta=f"{t['confidence']*100:.1f}% confidence",
            )
    with col_a:
        if sub.get("audio"):
            a = sub["audio"]
            st.metric(
                label=f"🎤 Audio Branch ({weights.get('audio', 0)*100:.0f}% weight)",
                value=a["top_emotion"],
                delta=f"{a['confidence']*100:.1f}% confidence",
            )


def render_results_panel(result: dict):
    """
    Master display function — call this with any API response dict.
    Renders the full dashboard panel with hero card + charts.

    Usage inside app.py:
        result = analyze_text(my_text)
        if result:
            render_results_panel(result)
    """
    if not result:
        return

    st.markdown("---")
    render_hero_emotion(result)

    all_scores = result.get("all_scores", {})
    if not all_scores:
        return

    st.markdown("#### 📊 Full Probability Distribution")
    tab_radar, tab_bar = st.tabs(["🕸️ Radar Map", "📊 Bar Chart"])

    with tab_radar:
        st.plotly_chart(render_radar_chart(all_scores), use_container_width=True)

    with tab_bar:
        st.plotly_chart(render_bar_chart(all_scores), use_container_width=True)

    # Show fusion breakdown only for multimodal responses
    if result.get("modality", "").startswith("multimodal"):
        render_fusion_breakdown(result)

    # Show raw JSON for developers / portfolio demo
    with st.expander("🔧 Raw API Response (JSON)"):
        st.json(result)


# ╔══════════════════════════════════════════════════════════╗
# ║  3. STANDALONE DEMO — run: streamlit run neurosense_client.py ║
# ╚══════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    st.set_page_config(page_title="NeuroSense Client Demo", layout="wide", page_icon="🧠")

    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #020c1b 0%, #0a192f 100%); }
        h1, h2, h3, p, label { color: #e6f1ff !important; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🧠 NeuroSense — API Client Demo")
    st.caption(f"Backend: `{BACKEND_URL}`")

    mode = st.radio("Select modality", ["Text Only", "Audio Only", "Multimodal Fusion"], horizontal=True)

    text_input = None
    audio_bytes = None

    if mode in ["Text Only", "Multimodal Fusion"]:
        text_input = st.text_area("Enter text:", height=100, placeholder="I'm absolutely thrilled about this project!")

    if mode in ["Audio Only", "Multimodal Fusion"]:
        audio_data = st.audio_input("Record audio:")
        if audio_data:
            audio_bytes = audio_data.read()

    if mode in ["Multimodal Fusion"]:
        tw = st.slider("Text fusion weight", 0.0, 1.0, 0.55)
    else:
        tw = 0.55

    if st.button("⚡ Analyze", type="primary", use_container_width=True):
        if mode == "Text Only":
            result = analyze_text(text_input)
        elif mode == "Audio Only":
            result = analyze_audio(audio_bytes) if audio_bytes else None
        else:
            result = analyze_multimodal(text_input, audio_bytes, tw)

        render_results_panel(result)
