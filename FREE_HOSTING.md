# 🚀 100% Free Hosting Guide (Individual Dev Strategy)

This guide explains how to host your 1.6 GB AI model and your persistent database for **$0 per month**.

---

## 🏗️ Step 1: Set up the Database (Supabase)
To keep user chats forever even when the server restarts, we use Supabase (PostgreSQL).

1.  **Sign up**: Go to [supabase.com](https://supabase.com/) and create a free account.
2.  **New Project**: Create a new project named `Haven`.
3.  **Get Connection String**:
    -   Go to **Project Settings** > **Database**.
    -   Find your **Connection string** (URI) in the "Node.js" or "URI" section. It looks like this:
        `postgresql://postgres:[PASSWORD]@db.[REF].supabase.co:5432/postgres`
    -   **Copy this** (replace `[PASSWORD]` with your actual database password).
4.  **Save this URL**: You will need it in Step 2.

---

## 🧠 Step 2: Set up the Backend (Hugging Face Spaces)
Hugging Face is the only platform that gives you **16 GB of RAM** for free.

1.  **Sign up**: Go to [huggingface.co](https://huggingface.co/) and create a free account.
2.  **Create New Space**:
    -   **Space name**: `haven-backend`
    -   **SDK**: Select **Docker** (Blank template).
    -   **Visibility**: Public (Free tier).
3.  **Environment Variables**:
    -   Go to **Settings** > **Variables and secrets**.
    -   Add **New Secret**:
        -   Name: `DATABASE_URL`
        -   Value: Your Supabase URI from Step 1.
    -   Add **New Secret**:
        -   Name: `GROQ_API_KEY`
        -   Value: Your Groq API Key.
4.  **Upload the Code**:
    -   Connect your GitHub repository (`haven-code`) to the Space.
    -   Or just `git push` your local files to the Hugging Face remote.
5.  **Wait for Build**: Hugging Face will automatically build your Docker container. Once it says "Running," your backend is live!

---

## 🌐 Step 3: Set up the Frontend (Vercel)
1.  **Sign up**: Go to [vercel.com](https://vercel.com/) and link your GitHub.
2.  **Import Repo**: Select `haven-code`.
3.  **Settings**:
    -   Vercel will detect it as a static site.
4.  **Update API URL**:
    -   In your `chatbot.html` code, make sure the `API_BASE_URL` points to your Hugging Face Space URL.
    -   Example: `https://[YOUR_USERNAME]-haven-backend.hf.space`
5.  **Deploy**: Click Deploy.

---

## ⚡ Summary of URLs
- **Backend (AI)**: `https://huggingface.co/spaces/[username]/haven-backend`
- **Database**: `https://supabase.com/dashboard/project/[id]`
- **Frontend (Web)**: `https://haven-code.vercel.app`

**You are now officially a cloud developer! 🚀**
