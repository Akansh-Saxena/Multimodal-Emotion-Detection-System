# start_public.ps1
# This script starts the backend API and generated a public HTTPS URL using Localtunnel

Write-Host "Starting Multimodal Emotion API locally..." -ForegroundColor Green

# 1. Start the FastAPI server in the background
$env:PYTHONPATH = "src"
Start-Process -FilePath "uvicorn" -ArgumentList "src.api:app", "--host", "0.0.0.0", "--port", "8000" -WindowStyle Hidden

Write-Host "Waiting 5 seconds for API to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# 2. Start ngrok to expose port 8000 to the world
Write-Host "Creating Public HTTPS Tunnel via Ngrok..." -ForegroundColor Green
Write-Host ">>> CHECK NGROK WINDOW FOR YOUR URL <<<" -ForegroundColor Cyan

# Start ngrok, it will open in its own window or output its URL
Start-Process -FilePath ".\ngrok.exe" -ArgumentList "http", "8000"
Write-Host ""
Write-Host "To find your link programmatically later, go to http://127.0.0.1:4040/api/tunnels" -ForegroundColor Yellow
