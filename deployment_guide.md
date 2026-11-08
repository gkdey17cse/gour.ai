# Deploying Dey.ai Backend to Render

This guide outlines the step-by-step process to deploy your FastAPI RAG application to [Render](https://render.com/), a popular cloud platform that offers a generous free tier for web services.

## Prerequisites

1.  **A GitHub Account:** Your code must be pushed to a GitHub repository.
2.  **A Render Account:** Sign up at [dashboard.render.com](https://dashboard.render.com/) using your GitHub account.
3.  **Data Ingested:** Ensure you have already run `python -m src.ingest` locally so your Qdrant Cloud database is populated with your resume and LinkedIn data. (Render will only handle the API serving, not the initial database population).

---

## Step 1: Push Code to GitHub

First, ensure your local code is pushed to a Git repository. 

Make sure your `.gitignore` is working properly. **Do NOT commit your `.env` file or your actual `resume.pdf`.**

```bash
git init
git add .
git commit -m "Initial commit for Dey.ai API"
git branch -M main
git remote add origin https://github.com/yourusername/profile-rag-api.git
git push -u origin main
```

---

## Step 2: Create a New Web Service on Render

1. Log in to your Render Dashboard.
2. Click the **"New +"** button in the top right corner and select **"Web Service"**.
3. Under "Connect a repository", click **"Build and deploy from a Git repository"**.
4. Connect your GitHub account (if you haven't already) and select your `profile-rag-api` repository.

---

## Step 3: Configure the Web Service

Fill out the form with the following details:

*   **Name:** `deyai-backend` (or any name you prefer)
*   **Region:** Choose the region closest to you (e.g., Frankfurt, Ohio, Singapore).
*   **Branch:** `main`
*   **Runtime:** Render will likely auto-detect your `Dockerfile` and select **Docker**. 
    * *Note: If you prefer to use the Native Python environment instead of Docker, change Runtime to **Python 3**, set the Build Command to `pip install -r requirements.txt`, and the Start Command to `uvicorn src.main:app --host 0.0.0.0 --port $PORT`.*
*   **Instance Type:** Select the **Free** tier ($0/month).

---

## Step 4: Set Environment Variables

Scroll down to the **Environment Variables** section. Click **"Add Environment Variable"** and paste the keys and values exactly as they are in your local `.env` file.

You need to add the following keys:

1.  `GROQ_API_KEY` = `gsk_...`
2.  `GEMINI_API_KEY` = `AIzaSy...`
3.  `QDRANT_CLUSTER_ENDPOINT` = `https://your-cluster-id.us-east-4-0.aws.cloud.qdrant.io:6333`
4.  `QDRANT_API_KEY` = `your_qdrant_key`
5.  `EMBEDDING_MODEL_NAME` = `all-MiniLM-L6-v2`
6.  `COLLECTION_NAME` = `profile_docs`

*(Optional but recommended if using Python runtime)*
*   `PYTHON_VERSION` = `3.10.0`

---

## Step 5: Deploy

1. Click the **"Create Web Service"** button at the bottom.
2. Render will now pull your repository, build the Docker container (or install Python dependencies), and launch the server.
3. You can watch the build progress in the terminal window on the Render dashboard. This usually takes 2-4 minutes.
4. Once you see `Uvicorn running on http://0.0.0.0:8000` or a green "Live" badge, your API is successfully deployed!

---

## Step 6: Test Your Live API

In the top left of your Render dashboard, you will see your public URL (e.g., `https://deyai-backend.onrender.com`).

**Test the Health Check:**
Open your browser and navigate to:
`https://deyai-backend.onrender.com/health`

**Test the Chat Endpoint (via cURL/Terminal):**
```bash
curl -X POST "https://deyai-backend.onrender.com/chat" \
     -H "Content-Type: application/json" \
     -d '{"query": "Hi Gour, what is your current project?"}'
```

---

## Important Render Free Tier Notes

*   **Spin Down (Cold Starts):** On the Free tier, Render will spin down (sleep) your web service after **15 minutes of inactivity**. 
*   When a new request comes in while the service is asleep, it will automatically wake up, but the first request might take **40-60 seconds** to respond. Subsequent requests will be instant.
*   To keep it awake during an interview or demo, you can ping the `/health` endpoint periodically, or use a free cron-job service like [cron-job.org](https://cron-job.org/) to ping `https://your-app.onrender.com/health` every 14 minutes.