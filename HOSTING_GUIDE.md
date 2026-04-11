# 🚀 Hosting Guide - Haven AI

This guide explains how to host your Haven chatbot on a public server (VPS) for free or at a low cost, while keeping your 1.6 GB AI model.

## 📦 1. Preparing the Server
We recommend **Oracle Cloud Always Free** (ARM Instance) or any VPS with at least **2 GB of RAM**.

### Step 1: Install Dependencies
Connect to your server via SSH and run:
```bash
sudo apt update && sudo apt install git python3-pip -y
```

### Step 2: Clone your Repository
```bash
git clone https://github.com/munjalsharma/haven-chatbot.git
cd haven-chatbot
```

### Step 3: Download the 1.6 GB Model
Since we excluded the model from GitHub (to keep it clean), you need to upload it to the server or download it.
**To upload from your PC to the server:**
```bash
scp backend/muril_emotion_model.pth username@your_server_ip:/path/to/haven-chatbot/backend/
```

## 🛡️ 2. Running Haven
1. **Set up Environment**: Create a `.env` file in the `backend/` folder.
   ```bash
   GROQ_API_KEY=your_key_here
   HAVEN_ADMIN_KEY=your_secure_password
   ```
2. **Install Python libs**:
   ```bash
   pip install -r backend/requirements.txt
   ```
3. **Start the Backend**:
   ```bash
   cd backend
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## 🛡️ 3. Accessing the Admin Dashboard
Once hosted, you can access your monitoring panel at:
`http://your_server_ip:8000/admin`

- **Login**: Use the `HAVEN_ADMIN_KEY` you set in your `.env`.
- **View Chats**: You will see all users and can click "View Transcript" to read their full history.

---

## 🏗️ 4. Frontend Deployment
You can host the `chatbot.html` on **Netlify** or **GitHub Pages**. Just ensure you update the `URL` in the JavaScript to point to your server's IP address instead of `localhost:8000`.
