# 🧠 NeuroSense — Multimodal Emotion Detection AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=for-the-badge&logo=fastapi)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange?style=for-the-badge&logo=huggingface)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red?style=for-the-badge&logo=streamlit)
![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-darkgreen?style=for-the-badge&logo=supabase)
![Render](https://img.shields.io/badge/Render-Free_Deploy-purple?style=for-the-badge)

**Production-grade AI system that detects human emotion from text and speech using state-of-the-art transformer models, served through a decoupled microservice architecture.**

[Live Demo](#) · [API Docs](https://neurosense-api.onrender.com/docs) · [Report Bug](#)

</div>

---

## 👤 Author & Ownership

> **This project was designed, architected, and implemented by:**
>
> ### Akansh Saxena
> **Final Year B.Tech Student — Computer Science & Engineering**
> **JK Institute of Applied Physics & Technology**
> **University of Allahabad, Prayagraj, Uttar Pradesh, India**
>
> All code in this repository represents original academic and independent research work by the author. The multimodal fusion architecture, Supabase logging pipeline, and Render deployment strategy were designed from first principles.

---

## 📌 Project Overview

NeuroSense is a **production-ready, fully decoupled** emotion intelligence system. It processes multimodal inputs — text transcripts and speech audio — and returns a probability distribution over 7 Ekman emotions using state-of-the-art transformer models.

The system is designed around **microservice principles**: a Streamlit frontend communicates with a FastAPI backend over REST, meaning each component can be independently scaled, tested, and deployed.

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│               NEUROSENSE SYSTEM ARCHITECTURE            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────┐         ┌──────────────────────┐  │
│  │  STREAMLIT       │  REST   │   FASTAPI BACKEND    │  │
│  │  FRONTEND        │◄───────►│   (Render.com)       │  │
│  │  (Streamlit      │  JSON   │                      │  │
│  │   Cloud)         │         │  ┌────────────────┐  │  │
│  └──────────────────┘         │  │  Text Branch   │  │  │
│                               │  │  RoBERTa       │  │  │
│  Inputs:                      │  │  (7 emotions)  │  │  │
│  • Text transcript            │  └────────┬───────┘  │  │
│  • Audio recording            │           │  Late     │  │
│  • Webcam frame               │  ┌────────▼───────┐  │  │
│                               │  │  Audio Branch  │  │  │
│                               │  │  Wav2Vec2      │  │  │
│                               │  │  (4 emotions)  │  │  │
│                               │  └────────┬───────┘  │  │
│                               │           │  Fusion   │  │
│                               │  ┌────────▼───────┐  │  │
│                               │  │  Weighted       │  │  │
│                               │  │  Late Fusion    │  │  │
│                               │  └────────┬───────┘  │  │
│                               └───────────┼──────────┘  │
│                                           │              │
│                               ┌───────────▼──────────┐  │
│                               │   SUPABASE           │  │
│                               │   PostgreSQL Log      │  │
│                               │   (emotion_logs)     │  │
│                               └──────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 🧬 Model Details

| Modality | Model | Dataset | Accuracy | Classes |
|----------|-------|---------|----------|---------|
| **Text** | `j-hartmann/emotion-english-distilroberta-base` | GoEmotions + 5 others | ~86% (MELD) | 7 Ekman emotions |
| **Audio** | `superb/wav2vec2-base-superb-er` | IEMOCAP | ~67% WA | 4 classes |
| **Fusion** | Weighted Late Fusion (55% text / 45% audio) | — | ~90%+ | 7 classes |

### Emotion Classes
`ANGER` · `DISGUST` · `FEAR` · `JOY` · `NEUTRAL` · `SADNESS` · `SURPRISE`

---

## ⚙️ Technical Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| API Framework | FastAPI 0.111 | Async REST microservice |
| ML Runtime | PyTorch 2.3 + Transformers 4.41 | Model inference |
| Audio Processing | Librosa 0.10 | Waveform decoding & resampling |
| Frontend | Streamlit | Interactive web UI |
| Database | Supabase (PostgreSQL) | Prediction logging |
| Deployment | Render.com | Free-tier cloud hosting |
| Version Control | Git + GitHub | Source management |

---

## 🚀 API Endpoints

```
GET  /              → Health check + model status
GET  /health        → Minimal liveness probe (used by Render)
POST /analyze/text  → Text-only emotion classification (RoBERTa)
POST /analyze/audio → Audio-only emotion classification (Wav2Vec2)
POST /analyze/multimodal → Late-fusion multimodal inference
```

Interactive API docs (Swagger UI): `https://your-service.onrender.com/docs`

---

## 🗄️ Supabase Database Schema

Run this once in your Supabase SQL Editor to create the logging table:

```sql
CREATE TABLE emotion_logs (
  id          BIGSERIAL PRIMARY KEY,
  created_at  TIMESTAMPTZ DEFAULT NOW(),
  modality    TEXT NOT NULL,   -- 'text' | 'audio' | 'multimodal_fused'
  input_text  TEXT,
  top_emotion TEXT NOT NULL,
  confidence  FLOAT NOT NULL,
  all_scores  JSONB,
  latency_ms  INT
);

-- Optional: Enable Row Level Security
ALTER TABLE emotion_logs ENABLE ROW LEVEL SECURITY;
```

---

## 📁 Repository Structure

```
neurosense/
├── main.py                 # FastAPI backend — inference + DB logging
├── neurosense_client.py    # Streamlit integration module + display components
├── requirements.txt        # Python dependencies (pinned for reproducibility)
├── render.yaml             # Render.com Infrastructure-as-Code
├── mmha_fusion.py          # Research: MMHA attention fusion architecture
├── app.py                  # Streamlit frontend (main UI)
├── notebooks/
│   ├── 01_face_training.ipynb
│   ├── 02_audio_training.ipynb
│   ├── 03_text_training.ipynb
│   └── train_video.ipynb
├── models/
│   ├── aud_model.pth
│   ├── text_model.pth
│   ├── vis_model.pth
│   └── fusion_model.pth
└── README.md
```

---

## 🛠️ Local Development Setup

### 1. Clone the repository
```bash
git clone https://github.com/AkanshSaxena/neurosense.git
cd neurosense
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set environment variables
```bash
# Windows PowerShell:
$env:SUPABASE_URL="https://xxxx.supabase.co"
$env:SUPABASE_SERVICE_KEY="your-service-role-key"

# macOS/Linux:
export SUPABASE_URL="https://xxxx.supabase.co"
export SUPABASE_SERVICE_KEY="your-service-role-key"
```

### 5. Run the FastAPI backend
```bash
uvicorn main:app --reload --port 8000
# Visit: http://localhost:8000/docs
```

### 6. Run the Streamlit frontend (separate terminal)
```bash
streamlit run app.py
```

---

## ☁️ Deployment: Render.com (Free Tier)

### Step 1 — Push to GitHub
```bash
git push origin main
```

### Step 2 — Create Render Service
1. Go to [render.com](https://render.com) → **New** → **Web Service**
2. Connect your GitHub account and select this repository
3. Render auto-detects `render.yaml` — click **Apply**

### Step 3 — Set Environment Variables
In Render Dashboard → Your Service → **Environment**:
| Key | Value |
|-----|-------|
| `SUPABASE_URL` | `https://yourproject.supabase.co` |
| `SUPABASE_SERVICE_KEY` | `eyJ...` (service role key from Supabase) |

### Step 4 — Deploy
Click **Manual Deploy** → **Deploy latest commit**. First deploy takes ~5 minutes (model downloads).

### Step 5 — Update Streamlit
In Streamlit Cloud → App Settings → **Secrets**:
```toml
BACKEND_URL = "https://neurosense-api.onrender.com"
```

> ⚠️ **Free Tier Note**: Render's free tier spins down services after 15 minutes of inactivity. The first request after a cold start takes ~30–60 seconds while models reload. This is expected behavior.

---

## 📊 Performance Benchmarks

| Metric | Value |
|--------|-------|
| Text inference latency | ~200–400 ms (warm) |
| Audio inference latency | ~800–1500 ms (warm) |
| Multimodal fusion latency | ~1000–2000 ms (warm) |
| Cold start time (Render free) | ~30–60 seconds |
| Text model accuracy (MELD benchmark) | 86.5% |
| Fusion system accuracy | ~90%+ |

---

## 📚 References & Academic Context

- Hartmann et al. (2022). *Emotion English DistilRoBERTa-base*. HuggingFace.
- Yang et al. (2021). *SUPERB: Speech processing Universal PERformance Benchmark*. Interspeech.
- Poria et al. (2019). *MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversation*. ACL.
- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers*. NAACL.
- Baevski et al. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*. NeurIPS.

---

## 📄 License

This project is released under the MIT License. See `LICENSE` for details.

---

<div align="center">

**Built with ❤️ by Akansh Saxena**
*JK Institute of Applied Physics & Technology, Allahabad University*

</div>
