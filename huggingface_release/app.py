# app.py
# This is a wrapper script for Hugging Face Spaces or Render deployments.
# It simply imports the FastAPI app and runs it via uvicorn.

import uvicorn
from src.api import app

if __name__ == "__main__":
    # Hugging Face Spaces typically expose port 7860
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
