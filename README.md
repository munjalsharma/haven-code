---
title: Haven AI
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# MyHaven — Complete Setup Guide (Windows 10 + VS Code)

---

## 📁 Folder Structure

```
MYHAVEN_UPDATED/
│
├── chatbot.html          ← Frontend chatbot page  ✅
├── index.html            ← Landing page
├── team.html             ← Team page
│
└── backend/
    ├── main.py           ← FastAPI backend  ✅
    ├── requirements.txt  ← Python packages  ✅
    ├── .env              ← Your API keys    ✅
    └── models/
        └── muril_emotion_model.pth   ← (optional) trained weights
```

---

## Step 1 — Get Your Gemini API Key

1. Go to → https://aistudio.google.com/apikey
2. Click **Create API Key**
3. Copy the key (looks like `AIzaSy...`)

---

## Step 2 — Set Up the .env File

1. Open `backend/.env` in VS Code
2. Replace `YOUR_GEMINI_API_KEY_HERE` with your actual key:
   ```
   GEMINI_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXX
   ```
3. Save the file

---

## Step 3 — Create Python Virtual Environment

Open a **terminal in VS Code** (`Ctrl + `` ` ``), then run these commands one by one:

```bash
# Navigate into the backend folder
cd backend

# Create a virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate
```

You should now see `(.venv)` at the start of your terminal line.

---

## Step 4 — Install Dependencies

With the virtual environment activated:

```bash
pip install -r requirements.txt
```

> ⏳ This will take a few minutes — it downloads PyTorch and the Transformers library.
> If you see a red error about `torch`, try:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

---

## Step 5 — Run the Backend

```bash
uvicorn main:app --reload
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

✅ Backend is now running!

---

## Step 6 — Open the Chatbot

1. In VS Code **Explorer**, right-click `chatbot.html`
2. Click **"Open with Live Server"**
   - If you don't have Live Server: Go to Extensions (`Ctrl+Shift+X`), search `Live Server`, install it
3. The chatbot opens in your browser at `http://127.0.0.1:5500/chatbot.html`

Haven will automatically greet you and the chat is live! 🎉

---

## Step 7 — (Optional) MuRIL Emotion Weights

Without the weights file, emotion detection defaults to **Neutral** — the chatbot still works fully.

To enable real emotion detection:
1. Train the model using `mindmate_integration.py` in Google Colab
2. Download the saved `muril_emotion_model.pth` file
3. Place it at: `backend/models/muril_emotion_model.pth`
4. Restart the backend (`Ctrl+C` then `uvicorn main:app --reload`)

The right sidebar will now show real emotion labels, confidence, and sentiment bars.

---

## How It Works

```
[chatbot.html]  →  POST /start   →  [main.py]  →  Gemini API  →  Greeting
[User types]    →  POST /chat    →  [main.py]  →  Gemini API  →  Reply
                                              →  MuRIL-BERT  →  Emotion + Sentiment
```

- **Conversation memory**: Last 20 messages are kept per session — Haven remembers context
- **Crisis detection**: Certain keywords trigger a safe response with helpline numbers
- **Emotion sidebar**: Updates after every message with emoji, confidence bar, and 8-emotion pills
- **Sentiment sidebar**: Shows polarity (negative↔positive) and subjectivity bars

---

## API Endpoints (for reference)

| Method | Endpoint  | Purpose                        |
|--------|-----------|--------------------------------|
| POST   | `/start`  | Begin session, get greeting    |
| POST   | `/chat`   | Send message, get AI reply     |
| POST   | `/reset`  | Clear conversation memory      |
| GET    | `/health` | Check if backend is running    |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Backend offline" in chat | Make sure `uvicorn main:app --reload` is running in terminal |
| `GEMINI_API_KEY is missing` | Check `backend/.env` — key must be set, no quotes around it |
| `pip install` fails for torch | Use the CPU-only install command in Step 4 |
| Port 8000 already in use | Run `uvicorn main:app --reload --port 8001` and update `BACKEND` in chatbot.html |
| MuRIL download is slow | First run downloads ~500MB — wait for it, it's one-time only |

---

## Crisis Resources

| Service | Number |
|---------|--------|
| Kiran Mental Health Helpline (India) | **9152987821** |
| AASRA | 91-9820466627 |
| iCall (TISS) | 9152987821 |

---

*Built with ❤️ by the MyHaven team — Maitreyi College*