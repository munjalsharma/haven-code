# 🚀 100% Free Hosting & Deployment Guide (Render + Supabase + Vercel + Local)

This guide explains how to host your **Haven AI Chatbot** with **Render** (Backend API), **Supabase** (PostgreSQL Database), **Vercel** (Frontend Website), and **Local Development**.

---

## 🏗️ Step 1: Set up the Database (Supabase)
Supabase provides free PostgreSQL database hosting so user conversations and context persist forever across restarts.

1. **Sign up**: Go to [supabase.com](https://supabase.com/) and create a free account.
2. **New Project**: Create a new project named `Haven`.
3. **Get Connection String**:
   - Go to **Project Settings** > **Database**.
   - Scroll to **Connection Pooler** section.
   - Set **Mode** to **Transaction** (port `6543`).
   - Copy the **Connection string (URI)**:
     `postgresql://postgres:[YOUR-PASSWORD]@aws-0-[region].pooler.supabase.com:6543/postgres`
   - Replace `[YOUR-PASSWORD]` with your database password.

---

## 🧠 Step 2: Set up the Backend (Render)
Render provides free web service hosting for Python FastAPI servers.

1. **Sign up**: Go to [render.com](https://render.com/) and link your GitHub account.
2. **New Web Service**: Click **New +** > **Web Service**.
3. **Connect Repository**: Select your GitHub repo `haven-code`.
4. **Configuration**:
   - **Name**: `myhaven-backend`
   - **Region**: Choose closest to your target users (e.g., Singapore, Frankfurt, Oregon).
   - **Root Directory**: Leave blank (or `backend` if deploying only backend subfolder).
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` (or `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`)
5. **Environment Variables**:
   Add the following under **Environment Variables**:
   - `GROQ_API_KEY`: Your Groq API key (from [console.groq.com](https://console.groq.com/))
   - `DATABASE_URL`: Your Supabase connection string from Step 1.
   - `HAVEN_ADMIN_KEY`: (Optional) Custom password for your admin panel.
6. **Deploy**: Click **Create Web Service**. Once deployed, copy your service URL (e.g. `https://myhaven-backend.onrender.com`).

---

## 🌐 Step 3: Set up the Frontend (Vercel)
Vercel hosts the web frontend with lightning speed.

1. **Sign up**: Go to [vercel.com](https://vercel.com/) and import your `haven-code` GitHub repo.
2. **Configure Backend URL**:
   - Open `config.js` in your repo before pushing, and set:
     ```javascript
     window.HAVEN_CONFIG = {
         BACKEND_URL: "https://myhaven-backend.onrender.com"
     };
     ```
   - Alternatively, users can set the Render URL inside the app by clicking **⚙️ Server Settings** in the header.
3. **Deploy**: Click **Deploy**. Your frontend is now live at `https://your-app.vercel.app`!

---

## 💻 Step 4: Run Locally (Local Development)

You can run Haven completely on your computer with local SQLite database fallback:

1. **Clone & Setup Environment**:
   ```bash
   git clone https://github.com/your-username/haven-code.git
   cd haven-code
   ```
2. **Set up `.env`**:
   Create a `.env` file inside `backend/`:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   HAVEN_ADMIN_KEY=haven_master_2026
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Start Backend**:
   ```bash
   cd backend
   python -m uvicorn main:app --reload --port 8000
   ```
5. **Open Frontend**:
   Open `chatbot.html` or `index.html` in your browser. The frontend auto-detects `http://127.0.0.1:8000` automatically!

---

## ⚡ Summary of Live URLs
- **Backend API**: `https://myhaven-backend.onrender.com`
- **Backend Health Check**: `https://myhaven-backend.onrender.com/health`
- **Database Dashboard**: `https://supabase.com/dashboard`
- **Frontend App**: `https://your-app.vercel.app`
