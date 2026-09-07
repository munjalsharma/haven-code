# 🌐 Haven Production Deployment Guide (Vercel + Render + Supabase)

This guide provides the complete setup for running Haven in production with **Vercel** (Frontend), **Render** (Backend), and **Supabase** (Database).

---

## 🗄️ Step 1: Database Setup (Supabase)

1. Go to [Supabase](https://supabase.com) and create a free project named `Haven`.
2. Go to **Project Settings** > **Database**.
3. Under **Connection Pooler**, copy your **Transaction Connection String (URI)**:
   ```env
   postgresql://postgres.[PROJECT_REF]:[YOUR_PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres?sslmode=require
   ```
4. Save this connection string for Step 2.

---

## ⚙️ Step 2: Backend Setup (Render)

1. Go to [Render](https://render.com) and create a new **Web Service**.
2. Connect your GitHub repository (`haven-code`).
3. Set the build parameters:
   - **Root Directory**: `backend` (or leave empty if using root)
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main.py:app --host 0.0.0.0 --port $PORT`
4. Add **Environment Variables** in Render Dashboard:
   - `GROQ_API_KEY`: Your Groq API key (`gsk_...`)
   - `GROQ_MODEL`: `llama-3.3-70b-versatile`
   - `DATABASE_URL`: Your Supabase connection string from Step 1
   - `HAVEN_ADMIN_KEY`: Your secret admin key
5. Deploy the service and copy your live Render Web Service URL:
   `https://<your-service-name>.onrender.com`

---

## 💻 Step 3: Frontend Setup (Vercel)

1. Open `config.js` in your project root and configure your Render backend URL:
   ```javascript
   window.HAVEN_CONFIG = {
       BACKEND_URL: "https://<your-service-name>.onrender.com"
   };
   ```
2. Go to [Vercel](https://vercel.com) and import your GitHub repository as a **Static Site**.
3. Deploy the application.
4. Access your live Vercel URL (e.g., `https://my-haven.vercel.app/chatbot.html`).

---

## ⚡ URL Resolution & Priority Order

Haven resolves the backend URL in `chatbot.html` using the following priority order:
1. `localStorage` saved URL (`myhaven_backend_url`)
2. `?backend=https://your-backend.onrender.com` URL parameter
3. `window.HAVEN_CONFIG.BACKEND_URL` in `config.js`
4. Localhost (`http://127.0.0.1:8000`) when running locally
5. Same-origin relative paths (`""`) fallback

---

## 🚀 Step 4: Verification & Troubleshooting

1. **"Couldn't reach the server" Error**:
   - Click the **Server URL** (`⚙️`) button in the top bar of `chatbot.html`.
   - Enter your Render backend URL (e.g., `https://my-haven-backend.onrender.com`).
2. **Render Cold Starts**:
   - Free Render services enter sleep mode after 15 minutes of inactivity (~30–50s wake time).
   - Haven automatically retries up to 5 times while displaying:
     `⚡ Server is waking up (Render cold start)... Please wait (Attempt X/5)`.
3. **Admin Panel**:
   - Access the admin monitoring dashboard at `https://<your-service-name>.onrender.com/admin` using your `HAVEN_ADMIN_KEY`.

