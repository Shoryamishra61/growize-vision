# Deploying Growize Vision to Google Cloud Run (Free Tier)

This guide walks through deploying the containerized Growize Vision system to **Google Cloud Run** using Google Cloud's always-free tier allowances.

---

## Google Cloud Free Tier Benefits for Cloud Run
Google Cloud provides an **Always Free Tier** every month:
- **2 Million Requests / Month** (100% free)
- **360,000 GB-seconds of memory** / month
- **180,000 vCPU-seconds** / month
- **1 GB egress / month**
- Automatic HTTPS with custom domain support and SSL certificates.

---

## Method 1: Using Google Cloud Console (No CLI required — Easiest & 100% Free)

If you don't have the `gcloud` CLI installed locally, you can deploy in 3 clicks directly from your web browser:

1. **Push your code to GitHub** (either public or private repository):
   ```bash
   git add -A
   git commit -m "feat: add Dockerfile and Cloud Run deployment configuration"
   git push origin main
   ```

2. Open the **[Google Cloud Run Console](https://console.cloud.google.com/run)**.
3. Click **Create Service**.
4. Select **Continuously deploy from a repository (source code)** $\rightarrow$ Click **Set up with Cloud Build**.
5. Select **GitHub** as your provider $\rightarrow$ Select your `agriculture vegetables` repository $\rightarrow$ Branch `main`.
6. Set Build Type to **Dockerfile** (`/Dockerfile`).
7. Configuration:
   - **Service Name:** `growize-vision`
   - **Region:** `asia-south1` (Mumbai / Bangalore) or `us-central1`
   - **Authentication:** Check `Allow unauthenticated invocations` (public access for evaluators).
   - **CPU Allocation:** `CPU is only allocated during request processing` (this ensures it consumes $0 when idle).
   - **Memory:** `2 GiB` (free tier eligible).
   - **Container Port:** `8080`.
8. Click **Create**.

Google Cloud Build will automatically build your Docker container on their cloud infrastructure, push the image to Google Artifact Registry, and provision a live HTTPS link:
```text
https://growize-vision-xxxxxx.a.run.app
```

---

## Method 2: Using Google Cloud Shell (In-Browser CLI)

You can run `gcloud` directly inside Google Cloud's free browser terminal:

1. Open **[shell.cloud.google.com](https://shell.cloud.google.com)**.
2. Clone your repo:
   ```bash
   git clone <YOUR_GITHUB_REPO_URL>
   cd "agriculture vegetables"
   ```
3. Submit build and deploy in one command:
   ```bash
   gcloud builds submit --config cloudbuild.yaml
   ```
4. Or directly deploy with source build:
   ```bash
   gcloud run deploy growize-vision \
       --source . \
       --region asia-south1 \
       --platform managed \
       --allow-unauthenticated \
       --memory 2Gi \
       --cpu 2 \
       --port 8080
   ```

---

## Verification & Evaluator Hand-off
Once deployed, Cloud Run provides:
- Live, low-latency container hosting in Bangalore/Mumbai region.
- End-to-end HTTPS camera permission support on iOS/Android and desktop browsers.
- Automatic zero-instance spin-down when idle, guaranteeing zero surprise charges.
