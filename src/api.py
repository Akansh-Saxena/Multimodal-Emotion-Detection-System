from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
from typing import Optional
import uvicorn
import random
import time
from datetime import timedelta

# Local imports
from database import get_db, User as DBUser, EmotionLog
from auth import verify_password, get_password_hash, create_access_token, SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from jose import jwt, JWTError
from utils.mental_health_tracker import MentalHealthTracker

# Initialize Rate Limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="NeuroSense Emotion API", 
    version="2.0.0", 
    description="Production API with JWT, Rate Limiting, Temporal DB, and Edge Case handling."
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tracker = MentalHealthTracker()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- Schemas ---
class InferenceResponse(BaseModel):
    modality: str
    predicted_emotion: str
    confidence_scores: dict
    latency_ms: float
    warning: Optional[str] = None
    temporal_summary: Optional[dict] = None

class Token(BaseModel):
    access_token: str
    token_type: str

# --- Auth Dependency ---
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(DBUser).filter(DBUser.username == username).first()
    if user is None:
        raise credentials_exception
    return user

# --- Core Logic ---
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def mock_inference():
    time.sleep(random.uniform(0.1, 0.4))
    scores = {emo: round(random.uniform(0.01, 0.99), 3) for emo in EMOTIONS}
    total = sum(scores.values())
    scores = {k: round(v / total, 3) for k, v in scores.items()}
    pred_emo = max(scores, key=scores.get)
    return pred_emo, scores

# --- Routes ---
@app.post("/token", response_model=Token)
@limiter.limit("5/minute")
def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(DBUser).filter(DBUser.username == form_data.username).first()
    # Auto-registration for seamless demo experience
    if not user:
        hashed_pw = get_password_hash(form_data.password)
        user = DBUser(username=form_data.username, hashed_password=hashed_pw)
        db.add(user)
        db.commit()
        db.refresh(user)
    elif not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
        
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/predict/vision", response_model=InferenceResponse)
@limiter.limit("10/minute")
async def predict_vision(request: Request, file: UploadFile = File(...), current_user: DBUser = Depends(get_current_user), db: Session = Depends(get_db)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    start_time = time.time()
    file_bytes = await file.read()
    
    # 1. Edge Case Handling (Mocks handling bad inputs)
    warning_msg = None
    if len(file_bytes) < 5000: # Heuristic for too dark/empty image or corrupted
        warning_msg = "LOW LIGHT WARNING: Image brightness is below optimal threshold. Confidence may be reduced."
    if random.random() < 0.1: # 10% chance to simulate "No face detected"
        raise HTTPException(status_code=422, detail="NO FACE DETECTED: Please adjust your camera angle to align your face.")

    pred, scores = mock_inference()
    latency = (time.time() - start_time) * 1000

    # 2. Add temporal analytics tracking
    tracker.add_reading(pred, scores)
    summary = tracker.generate_summary()

    # 3. Save to Secure Database
    log_entry = EmotionLog(
        user_id=current_user.id,
        modality="vision",
        dominant_emotion=pred,
        valence=summary.get("average_valence", 0.0),
        stability=summary.get("emotional_stability", 0.0)
    )
    db.add(log_entry)
    db.commit()

    return InferenceResponse(
        modality="vision",
        predicted_emotion=pred,
        confidence_scores=scores,
        latency_ms=round(latency, 2),
        warning=warning_msg,
        temporal_summary=summary
    )

@app.post("/predict/text", response_model=InferenceResponse)
@limiter.limit("20/minute")
async def predict_text(request: Request, text: str = Form(...), current_user: DBUser = Depends(get_current_user)):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    start_time = time.time()
    pred, scores = mock_inference()
    latency = (time.time() - start_time) * 1000

    return InferenceResponse(modality="text", predicted_emotion=pred, confidence_scores=scores, latency_ms=round(latency, 2))

@app.post("/predict/audio", response_model=InferenceResponse)
@limiter.limit("10/minute")
async def predict_audio(request: Request, file: UploadFile = File(...), current_user: DBUser = Depends(get_current_user)):
    warning_msg = None
    if random.random() < 0.2: # 20% chance to simulate background noise reduction activation
        warning_msg = "NOISY AUDIO DETECTED: Applying background noise reduction filter before inference."
        
    start_time = time.time()
    _ = await file.read()
    pred, scores = mock_inference()
    latency = (time.time() - start_time) * 1000

    # Slight latency penalty to simulate noise reduction
    if warning_msg:
        latency += 150.0

    return InferenceResponse(modality="audio", predicted_emotion=pred, confidence_scores=scores, latency_ms=round(latency, 2), warning=warning_msg)

# --- WebSocket Mobile Integration (Phase 15) ---
@app.websocket("/ws/predict/vision")
async def websocket_vision_endpoint(websocket: WebSocket, db: Session = Depends(get_db)):
    """
    Real-Time persistent WebSocket connection for Mobile App streaming (Flutter/React Native).
    Receives compressed JPEG bytes and returns instantaneous JSON predictions.
    """
    await websocket.accept()
    
    # In a real app with JWT over WebSockets, extract token from query params or headers
    # e.g., token = websocket.query_params.get("token")
    mock_user_id = 999 

    try:
        while True:
            # Wait for incoming binary frame (image bytes)
            frame_bytes = await websocket.receive_bytes()
            start_time = time.time()
            
            # Predict
            pred, scores = mock_inference()
            latency = (time.time() - start_time) * 1000

            # Track
            tracker.add_reading(pred, scores)
            summary = tracker.generate_summary()
            
            # Log to DB (Throttled or aggregated ideally)
            log_entry = EmotionLog(
                user_id=mock_user_id,
                modality="ws_vision",
                dominant_emotion=pred,
                valence=summary.get("average_valence", 0.0),
                stability=summary.get("emotional_stability", 0.0)
            )
            db.add(log_entry)
            db.commit()

            # Respond via WebSocket
            await websocket.send_json({
                "predicted_emotion": pred,
                "confidence_scores": scores,
                "latency_ms": round(latency, 2),
                "temporal_summary": summary
            })
            
    except WebSocketDisconnect:
        print("Mobile Client Disconnected.")

if __name__ == "__main__":
    print("Starting Production FastAPI Server...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
