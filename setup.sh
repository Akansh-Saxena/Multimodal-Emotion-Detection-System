#!/usr/bin/env bash
# ============================================================
# NeuroSense AI — Project Setup & Git Initialization Script
# Author: Akansh Saxena | JKIAP&T, Allahabad University
# ============================================================
# Run this script from inside your project folder:
#   chmod +x setup.sh && ./setup.sh
#
# Or copy-paste each block into your terminal manually.
# ============================================================

set -e  # Exit on any error

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   NeuroSense AI — Project Initialization             ║"
echo "║   Author: Akansh Saxena                              ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ──────────────────────────────────────────────────────────────
# STEP 1 — Virtual Environment
# ──────────────────────────────────────────────────────────────
echo "📦 [1/5] Creating Python virtual environment..."
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# On Windows PowerShell, use instead:
# venv\Scripts\Activate.ps1

echo "✅ Virtual environment activated."

# ──────────────────────────────────────────────────────────────
# STEP 2 — Install Dependencies
# ──────────────────────────────────────────────────────────────
echo ""
echo "🔧 [2/5] Installing dependencies from requirements.txt..."
echo "    (This downloads ~2 GB of ML models on first run)"
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependencies installed."

# ──────────────────────────────────────────────────────────────
# STEP 3 — Git Repository Initialization
# ──────────────────────────────────────────────────────────────
echo ""
echo "🔑 [3/5] Initializing Git repository with author credentials..."

git init

# Stamp your authorship permanently on this repository
# These settings override any global git config for this repo only
git config user.name  "Akansh Saxena"
git config user.email "akansh.saxena@example.com"   # ← Replace with your real email

echo "✅ Git initialized. Author set to: $(git config user.name)"

# Create .gitignore to prevent committing secrets and large model weights
cat > .gitignore << 'EOF'
# Python
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.egg-info/
dist/
build/

# Environment secrets — NEVER commit these
.env
.env.local
secrets.toml
.streamlit/secrets.toml

# Model weights — too large for GitHub (use Git LFS or HuggingFace Hub)
*.pth
*.pt
*.bin
*.onnx

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Jupyter checkpoints
.ipynb_checkpoints/

# Render / deployment artifacts
*.log
EOF

echo "✅ .gitignore created (model weights and secrets excluded)."

# ──────────────────────────────────────────────────────────────
# STEP 4 — First Authorized Commit
# ──────────────────────────────────────────────────────────────
echo ""
echo "💾 [4/5] Staging files and creating initial commit..."

git add main.py
git add requirements.txt
git add neurosense_client.py
git add render.yaml
git add README.md
git add .gitignore
git add mmha_fusion.py
git add app.py

# The --author flag permanently embeds your identity in this commit's metadata
git commit \
  --author="Akansh Saxena <akansh.saxena@example.com>" \
  -m "feat: initial commit — decoupled multimodal emotion detection system

Implements production-grade NeuroSense AI backend:
- FastAPI microservice with async request handling
- RoBERTa text-emotion classification (7 Ekman emotions)
- Wav2Vec2 speech-emotion recognition (4 classes)
- Weighted late-fusion multimodal inference endpoint
- Supabase PostgreSQL logging pipeline
- Render.com free-tier deployment configuration
- Streamlit frontend integration module with Plotly dashboards

Author: Akansh Saxena
Institute: JK Institute of Applied Physics & Technology
University: Allahabad University, Prayagraj"

echo "✅ Initial commit created."
git log --oneline -1

# ──────────────────────────────────────────────────────────────
# STEP 5 — Push to GitHub (manual steps shown)
# ──────────────────────────────────────────────────────────────
echo ""
echo "🚀 [5/5] Next steps to push to GitHub:"
echo ""
echo "   1. Create a new repository at: https://github.com/new"
echo "      Name: neurosense-emotion-ai"
echo "      Visibility: Public (for portfolio)"
echo "      Do NOT initialize with README (you have one already)"
echo ""
echo "   2. Run these commands (replace YOUR_USERNAME):"
echo "      git remote add origin https://github.com/YOUR_USERNAME/neurosense-emotion-ai.git"
echo "      git branch -M main"
echo "      git push -u origin main"
echo ""
echo "   3. Deploy to Render.com:"
echo "      a) Go to https://render.com → New → Web Service"
echo "      b) Connect GitHub → select neurosense-emotion-ai repo"
echo "      c) Render reads render.yaml automatically"
echo "      d) Add env vars in Dashboard → Environment:"
echo "         SUPABASE_URL        = https://xxxx.supabase.co"
echo "         SUPABASE_SERVICE_KEY = your-service-role-key"
echo "      e) Click 'Deploy' — first build takes ~5 minutes"
echo ""
echo "   4. Update Streamlit secrets.toml:"
echo "      BACKEND_URL = \"https://your-service.onrender.com\""
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   Setup complete! NeuroSense is ready to deploy.     ║"
echo "║   Author: Akansh Saxena | JKIAP&T, AU Prayagraj     ║"
echo "╚══════════════════════════════════════════════════════╝"
