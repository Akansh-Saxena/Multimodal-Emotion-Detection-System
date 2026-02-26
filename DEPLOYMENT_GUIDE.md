# Multimodal Emotion Detection - Deployment Guide

This guide provides instructions on how to take the completed application from your local machine to a production environment (like Render, AWS EC2, or Heroku).

## 1. Local Testing Before Deployment
Before pushing to production, verify the production server works locally:
```bash
docker build -t multimodal-ai .
docker run -p 8000:8000 multimodal-ai
```
Visit `http://localhost:8000` to verify the API is alive.

## 2. Choosing a Deployment Platform

### Option A: Render (Easiest / Free Tier Available)
Render makes deploying Dockerized Web Services extremely simple.
1. Create a GitHub repository and push your entire `MULTIMODAL_EMOTION_DETECTION_01` folder.
2. Log into [Render.com](https://render.com/).
3. Create a **New > Web Service**.
4. Connect your GitHub repository.
5. In the configuration:
   - **Environment**: Docker
   - **Instance Type**: Select an instance with at least 2GB RAM (Models like DistilBERT and EfficientNet require memory). *Note: The Free Tier might struggle with out-of-memory errors on heavy ML loads.*
6. Click **Deploy Web Service**. Render will build the Docker container and start Gunicorn automatically.

### Option B: AWS EC2 (Best for GPU Scaling)
For full Major Project capabilities (using GPU for faster inference):
1. Launch an EC2 instance with an NVIDIA GPU (e.g., `g4dn.xlarge`).
2. Select an Amazon Machine Image (AMI) with Deep Learning bases (which includes CUDA/Docker pre-installed).
3. SSH into the instance:
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   ```
4. Clone your repository.
5. Build and run using the `nvidia-docker` runtime to pass GPU access to the container:
   ```bash
   docker build -t multimodal-ai .
   docker run --gpus all -d -p 80:8000 multimodal-ai
   ```
   *(Note: You will need to change the bind port on Gunicorn or map 80->8000 in docker as shown).*

## 3. Serving the Frontend App
Currently, your `index.html` connects to `127.0.0.1:8000`. 
Before deploying the frontend, open `public/index.html` and change:
```javascript
const API_BASE = "http://127.0.0.1:8000";
```
To your deployed backend URL:
```javascript
const API_BASE = "https://your-deployed-secure-url.onrender.com";
```
You can host the `public` folder completely for free on **Vercel**, **Netlify**, or **GitHub Pages**.

## 4. Hardware Sizing Warnings
- **RAM**: Loading Text + Vision + Audio models simultaneously requires significant RAM (~2-4GB).
- **Disk Space**: PyTorch alone is ~2GB. Ensure your deployment disk has at least 8GB of free space.
- **Timeout**: Inference takes time without a GPU. The `base` Dockerfile sets a Gunicorn timeout of `120s`. Adjust `--timeout` in the Dockerfile if needed.
