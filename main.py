"""
============================================================
NeuroSense AI — Multimodal Emotion Detection Backend
============================================================
Author  : Akansh Saxena
Institute: JK Institute of Applied Physics & Technology,
           University of Allahabad, Prayagraj
============================================================
Architecture : FastAPI Microservice
Models       : RoBERTa (Text) · Wav2Vec2 (Audio)
Database     : Supabase (PostgreSQL) via REST API
Deployment   : Render.com (Free Tier)
============================================================
"""

import os
import io
import time
import logging
import traceback
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
import librosa
import requests as http_requests

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Hugging Face Transformers ──────────────────────────────
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
)

# ── Logging Configuration ─────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("neurosense")


# ╔══════════════════════════════════════════════════════════╗
# ║  1. GLOBAL MODEL REGISTRY — Loaded once at startup      ║
# ╚══════════════════════════════════════════════════════════╝

# Store all loaded models in a simple dict to avoid reloading
model_registry: dict = {}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Inference device: {DEVICE.upper()}")

# ── Emotion label maps ─────────────────────────────────────
# RoBERTa (j-hartmann) maps to 7 Ekman emotions
ROBERTA_LABELS = {
    "anger": "ANGER",
    "disgust": "DISGUST",
    "fear": "FEAR",
    "joy": "JOY",
    "neutral": "NEUTRAL",
    "sadness": "SADNESS",
    "surprise": "SURPRISE",
}

# Wav2Vec2 (superb/wav2vec2-base-superb-er) maps to 4 classes
WAV2VEC_LABELS = {
    "ang": "ANGER",
    "hap": "JOY",
    "neu": "NEUTRAL",
    "sad": "SADNESS",
}


def _load_text_model():
    """
    Load j-hartmann/emotion-english-distilroberta-base.
    — 7 emotion classes, trained on 6 datasets (GoEmotions, SST, etc.)
    — ~82 MB on disk, fits comfortably in Render's 512 MB free RAM
    — Returns a HuggingFace pipeline for one-call inference
    """
    logger.info("Loading RoBERTa text-emotion model …")
    t0 = time.time()
    pipe = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        device=0 if DEVICE == "cuda" else -1,  # -1 forces CPU
        top_k=None,  # Return ALL class probabilities, not just top-1
    )
    logger.info(f"✅ Text model loaded in {time.time()-t0:.1f}s")
    return pipe


def _load_audio_model():
    """
    Load superb/wav2vec2-base-superb-er.
    — Fine-tuned on IEMOCAP for speech emotion recognition
    — Returns the feature extractor + model separately for raw audio input
    """
    model_id = "superb/wav2vec2-base-superb-er"
    logger.info(f"Loading Wav2Vec2 audio-emotion model ({model_id}) …")
    t0 = time.time()
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
    model.to(DEVICE)
    model.eval()
    logger.info(f"✅ Audio model loaded in {time.time()-t0:.1f}s")
    return extractor, model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Models are loaded ONCE when the server boots — not per-request.
    This is the correct production pattern; avoids cold-start latency.
    """
    # ── STARTUP ───────────────────────────────────────────
    logger.info("=== NeuroSense Backend Initializing ===")
    model_registry["text_pipe"] = _load_text_model()
    extractor, wav2vec = _load_audio_model()
    model_registry["audio_extractor"] = extractor
    model_registry["audio_model"] = wav2vec
    logger.info("=== All models ready. Accepting requests. ===")
    yield
    # ── SHUTDOWN ──────────────────────────────────────────
    logger.info("=== NeuroSense Backend Shutting Down ===")
    model_registry.clear()


# ╔══════════════════════════════════════════════════════════╗
# ║  2. FASTAPI APP INITIALIZATION                          ║
# ╚══════════════════════════════════════════════════════════╝

app = FastAPI(
    title="NeuroSense Emotion Intelligence API",
    description=(
        "Production-grade Multimodal Emotion Detection microservice. "
        "Supports text (RoBERTa) and audio (Wav2Vec2) inference with "
        "Supabase logging. Built by Akansh Saxena, JKIAP&T, Allahabad University."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — allow your Streamlit app's domain ──────────────
# In production, replace "*" with your exact Streamlit Cloud URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ╔══════════════════════════════════════════════════════════╗
# ║  3. SUPABASE LOGGING LAYER                              ║
# ╚══════════════════════════════════════════════════════════╝

SUPABASE_URL = os.getenv("SUPABASE_URL", "")          # Set on Render dashboard
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")  # Use the service role key


def _log_to_supabase(payload: dict) -> bool:
    """
    Insert a single prediction record into the Supabase `emotion_logs` table.

    Table schema (run once in Supabase SQL editor):
    ┌────────────────────────────────────────────────────────┐
    │ CREATE TABLE emotion_logs (                            │
    │   id          BIGSERIAL PRIMARY KEY,                   │
    │   created_at  TIMESTAMPTZ DEFAULT NOW(),               │
    │   modality    TEXT NOT NULL,   -- 'text' | 'audio'     │
    │   input_text  TEXT,                                    │
    │   top_emotion TEXT NOT NULL,                           │
    │   confidence  FLOAT NOT NULL,                          │
    │   all_scores  JSONB,                                   │
    │   latency_ms  INT                                      │
    │ );                                                     │
    └────────────────────────────────────────────────────────┘

    Uses the raw Supabase REST API — no extra SDK required.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("Supabase credentials not set — skipping DB log.")
        return False

    endpoint = f"{SUPABASE_URL}/rest/v1/emotion_logs"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",  # Don't return the inserted row (faster)
    }
    try:
        resp = http_requests.post(endpoint, json=payload, headers=headers, timeout=5)
        resp.raise_for_status()
        logger.info(f"Supabase log → {resp.status_code}")
        return True
    except Exception as exc:
        # Never let a logging failure crash the inference response
        logger.error(f"Supabase log failed: {exc}")
        return False


# ╔══════════════════════════════════════════════════════════╗
# ║  4. INFERENCE HELPERS                                   ║
# ╚══════════════════════════════════════════════════════════╝

def _run_text_inference(text: str) -> dict:
    """
    Run RoBERTa text-emotion classification.

    Data flow:
      raw text → HuggingFace pipeline → list[{label, score}]
      → sort by score → build response dict
    """
    pipe = model_registry["text_pipe"]
    raw: list[dict] = pipe(text)[0]  # [{'label': 'joy', 'score': 0.92}, ...]

    # Normalise label names to uppercase for consistency
    scores = {
        ROBERTA_LABELS.get(r["label"], r["label"].upper()): round(r["score"], 4)
        for r in raw
    }

    # Sort descending for display clarity
    sorted_scores = dict(sorted(scores.items(), key=lambda x: -x[1]))
    top_emotion = max(scores, key=scores.get)
    confidence = scores[top_emotion]

    return {
        "top_emotion": top_emotion,
        "confidence": confidence,
        "all_scores": sorted_scores,
    }


def _run_audio_inference(audio_bytes: bytes) -> dict:
    """
    Run Wav2Vec2 speech emotion recognition on raw audio bytes.

    Data flow:
      bytes → librosa decode (16 kHz mono) → feature extractor
      → Wav2Vec2ForSequenceClassification → softmax → top emotion
    """
    extractor = model_registry["audio_extractor"]
    model = model_registry["audio_model"]

    # ── Decode audio ──────────────────────────────────────
    # librosa.load always resamples to target_sr and converts to mono
    try:
        waveform, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=16000,   # Wav2Vec2 expects 16 kHz
            mono=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Audio decode error: {exc}")

    # ── Feature extraction ────────────────────────────────
    inputs = extractor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # ── Forward pass ──────────────────────────────────────
    with torch.no_grad():
        logits = model(**inputs).logits  # shape: (1, num_classes)

    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    label_ids = model.config.id2label  # e.g. {0: 'ang', 1: 'hap', ...}

    scores = {
        WAV2VEC_LABELS.get(label_ids[i], label_ids[i].upper()): round(float(p), 4)
        for i, p in enumerate(probs)
    }
    sorted_scores = dict(sorted(scores.items(), key=lambda x: -x[1]))
    top_emotion = max(scores, key=scores.get)
    confidence = scores[top_emotion]

    return {
        "top_emotion": top_emotion,
        "confidence": confidence,
        "all_scores": sorted_scores,
    }


def _fuse_predictions(text_result: dict, audio_result: dict, text_weight: float = 0.55) -> dict:
    """
    Late-fusion: weighted average of text and audio confidence vectors.

    Strategy:
    — Align emotion labels from both modalities into a common 7-class space.
    — Weight text higher (0.55) vs audio (0.45) — text models are more accurate
      on clean data; audio matters for sarcasm/tonality detection.
    — Return a single fused emotion prediction.

    This mirrors the MMHA paper's late-fusion baseline.
    """
    audio_weight = 1.0 - text_weight
    all_emotions = set(text_result["all_scores"]) | set(audio_result["all_scores"])

    fused_scores = {}
    for emo in all_emotions:
        t_score = text_result["all_scores"].get(emo, 0.0)
        a_score = audio_result["all_scores"].get(emo, 0.0)
        fused_scores[emo] = round(t_score * text_weight + a_score * audio_weight, 4)

    sorted_fused = dict(sorted(fused_scores.items(), key=lambda x: -x[1]))
    top_emotion = max(fused_scores, key=fused_scores.get)

    return {
        "top_emotion": top_emotion,
        "confidence": fused_scores[top_emotion],
        "all_scores": sorted_fused,
        "fusion_weights": {"text": text_weight, "audio": audio_weight},
    }


# ╔══════════════════════════════════════════════════════════╗
# ║  5. API ENDPOINTS                                       ║
# ╚══════════════════════════════════════════════════════════╝

@app.get("/", tags=["Health"])
async def root():
    """Health check — Render pings this to keep the service alive."""
    return {
        "service": "NeuroSense Emotion Intelligence API",
        "status": "operational",
        "author": "Akansh Saxena",
        "institute": "JKIAP&T, Allahabad University",
        "models_loaded": list(model_registry.keys()),
        "device": DEVICE,
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# ── TEXT ENDPOINT ─────────────────────────────────────────
@app.post("/analyze/text", tags=["Inference"])
async def analyze_text(text: str = Form(..., description="Raw text to classify")):
    """
    Classify the emotional content of a text string.

    Returns:
    - top_emotion  : highest-confidence emotion label
    - confidence   : float 0–1
    - all_scores   : full probability distribution over 7 emotions
    - latency_ms   : server-side inference time
    """
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")

    t_start = time.time()
    result = _run_text_inference(text.strip())
    latency_ms = int((time.time() - t_start) * 1000)

    # Fire-and-forget Supabase log (non-blocking; won't crash endpoint on failure)
    _log_to_supabase({
        "modality": "text",
        "input_text": text[:512],  # Truncate for DB storage
        "top_emotion": result["top_emotion"],
        "confidence": result["confidence"],
        "all_scores": result["all_scores"],
        "latency_ms": latency_ms,
    })

    return {
        **result,
        "modality": "text",
        "latency_ms": latency_ms,
    }


# ── AUDIO ENDPOINT ────────────────────────────────────────
@app.post("/analyze/audio", tags=["Inference"])
async def analyze_audio(audio: UploadFile = File(..., description="Audio file (wav/mp3/ogg)")):
    """
    Classify the emotional content of a speech audio file.
    Accepts any format librosa can decode (wav, mp3, ogg, flac, m4a).

    Returns:
    - top_emotion  : highest-confidence emotion label (4-class)
    - confidence   : float 0–1
    - all_scores   : probability over ANGER / JOY / NEUTRAL / SADNESS
    - latency_ms   : inference time
    """
    audio_bytes = await audio.read()
    if len(audio_bytes) < 1000:
        raise HTTPException(status_code=400, detail="Audio file too small or empty.")

    t_start = time.time()
    result = _run_audio_inference(audio_bytes)
    latency_ms = int((time.time() - t_start) * 1000)

    _log_to_supabase({
        "modality": "audio",
        "input_text": None,
        "top_emotion": result["top_emotion"],
        "confidence": result["confidence"],
        "all_scores": result["all_scores"],
        "latency_ms": latency_ms,
    })

    return {
        **result,
        "modality": "audio",
        "latency_ms": latency_ms,
    }


# ── MULTIMODAL FUSION ENDPOINT ────────────────────────────
@app.post("/analyze/multimodal", tags=["Inference"])
async def analyze_multimodal(
    text: Optional[str] = Form(None, description="Optional text transcript"),
    audio: Optional[UploadFile] = File(None, description="Optional audio file"),
    text_weight: float = Form(0.55, ge=0.0, le=1.0, description="Fusion weight for text (0–1)"),
):
    """
    Multimodal Late-Fusion Endpoint.

    Accepts text AND/OR audio. Performs:
    1. Independent inference on each provided modality
    2. Weighted late-fusion if both modalities are provided
    3. Falls back gracefully to single-modality if only one is supplied

    This endpoint is the centerpiece of the NeuroSense system.
    """
    if not text and not audio:
        raise HTTPException(status_code=400, detail="Provide at least one modality (text or audio).")

    t_start = time.time()
    text_result = None
    audio_result = None

    if text and text.strip():
        logger.info("Running text inference …")
        text_result = _run_text_inference(text.strip())

    if audio:
        logger.info("Running audio inference …")
        audio_bytes = await audio.read()
        audio_result = _run_audio_inference(audio_bytes)

    # ── Fusion logic ──────────────────────────────────────
    if text_result and audio_result:
        fused = _fuse_predictions(text_result, audio_result, text_weight)
        modality = "multimodal_fused"
    elif text_result:
        fused = {**text_result, "fusion_weights": {"text": 1.0, "audio": 0.0}}
        modality = "text_only"
    else:
        fused = {**audio_result, "fusion_weights": {"text": 0.0, "audio": 1.0}}
        modality = "audio_only"

    latency_ms = int((time.time() - t_start) * 1000)

    _log_to_supabase({
        "modality": modality,
        "input_text": (text or "")[:512],
        "top_emotion": fused["top_emotion"],
        "confidence": fused["confidence"],
        "all_scores": fused["all_scores"],
        "latency_ms": latency_ms,
    })

    return {
        **fused,
        "modality": modality,
        "sub_results": {
            "text": text_result,
            "audio": audio_result,
        },
        "latency_ms": latency_ms,
    }


# ── GLOBAL EXCEPTION HANDLER ──────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check server logs."},
    )


# ── LOCAL DEV ENTRY POINT ─────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
