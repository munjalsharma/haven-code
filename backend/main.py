"""
MyHaven Backend — FastAPI v6.0
Groq (llama-3.3-70b-versatile) + PERMANENT SQLite memory + MuRIL emotion/sentiment
All conversation history, user context, mood, topics saved to myhaven.db
Crisis: shares helplines BUT keeps conversation open
"""

import sqlite3
import json
import os
import re
import time
import csv
import io
from datetime import datetime
from typing import Dict, List, Tuple, Any
from contextlib import contextmanager

# PostgreSQL support for cloud hosting (Supabase)
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None
 
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field
 
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
 
# Add this near the top of main.py, after existing imports
from mindmate_integration import MindMateSentimentAnalyzer
 
# Initialize the analyzer (do this once globally, outside any function)
print("[MindMate] Loading sentiment analyzer...")
mindmate_analyzer = MindMateSentimentAnalyzer(model_path="muril_emotion_model.pth")
print("[MindMate] ✅ Analyzer ready!")
# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
 
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

# Admin access key - set this in your .env file
HAVEN_ADMIN_KEY = os.getenv("HAVEN_ADMIN_KEY", "haven_master_2026").strip()
 
MAX_TURNS = 30
 
# DB paths
DATABASE_URL = os.getenv("DATABASE_URL", "").strip() # For Supabase
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "myhaven.db")

def is_postgres():
    return bool(DATABASE_URL)

def get_placeholder():
    return "%s" if is_postgres() else "?"
 
CRISIS_WORDS = [
    "suicide", "kill myself", "end my life", "self harm", "self-harm",
    "want to die", "i want to die", "i will die", "hurt myself",
    "no point living", "can't go on", "rather be dead", "end it all"
]
 
INDIA_KIRAN = "9152987821 (Kiran · Free · 24/7)"
 
 
# ══════════════════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ══════════════════════════════════════════════════════════════════════════════
 
def get_db_connection():
    if is_postgres():
        if not psycopg2:
            raise ImportError("psycopg2-binary is required for PostgreSQL support.")
        
        url = DATABASE_URL
        if "supabase.co" in url or "supabase.com" in url:
            if "sslmode=" not in url:
                separator = "&" if "?" in url else "?"
                url += f"{separator}sslmode=require"
        
        try:
            return psycopg2.connect(url)
        except Exception as e:
            print(f"[DB] ❌ Postgres connection failed: {e}")
            raise e
            
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn
 
@contextmanager
def get_db():
    conn = get_db_connection()
    try:
        if is_postgres():
            cur = conn.cursor(cursor_factory=RealDictCursor)
            yield cur
        else:
            yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def db_execute(conn_or_cur, query, params=None):
    if is_postgres():
        query = query.replace("?", "%s")
    if params is None:
        return conn_or_cur.execute(query)
    return conn_or_cur.execute(query, params)
 
 
def init_db():
    """Create tables if they don't exist. Safe to call multiple times."""
    is_pg = is_postgres()
    
    # Common table strings with minor syntax variations
    users_sql = """
        CREATE TABLE IF NOT EXISTS users (
            user_id     TEXT PRIMARY KEY,
            name        TEXT,
            language    TEXT DEFAULT 'english',
            topics      TEXT DEFAULT '[]',
            mood_history TEXT DEFAULT '[]',
            details     TEXT DEFAULT '[]',
            created_at  REAL,
            updated_at  REAL
        );
    """
    
    # SQLite uses AUTOINCREMENT, Postgres uses SERIAL
    if is_pg:
        messages_sql = """
            CREATE TABLE IF NOT EXISTS messages (
                id          SERIAL PRIMARY KEY,
                user_id     TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );
        """
        diary_sql = """
            CREATE TABLE IF NOT EXISTS diary_entries (
                id              SERIAL PRIMARY KEY,
                user_id         TEXT NOT NULL,
                raw_chat        TEXT NOT NULL,
                emotion_label   TEXT,
                sentiment_score REAL,
                timestamp       REAL NOT NULL
            );
        """
    else:
        messages_sql = """
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );
        """
        diary_sql = """
            CREATE TABLE IF NOT EXISTS diary_entries (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         TEXT NOT NULL,
                raw_chat        TEXT NOT NULL,
                emotion_label   TEXT,
                sentiment_score REAL,
                timestamp       REAL NOT NULL
            );
        """

    with get_db() as conn:
        # Use individual execute() calls which work for both SQLite connections and Postgres cursors
        conn.execute(users_sql)
        conn.execute(messages_sql)
        conn.execute(diary_sql)
        
        # Create indexes - safe for both DB types
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_diary_user_id ON diary_entries(user_id);")
            
    print(f"[DB] ✅ Database ready ({'Postgres' if is_pg else 'SQLite'})")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# DB HELPERS — READ / WRITE
# ══════════════════════════════════════════════════════════════════════════════
 
def db_get_user(user_id: str) -> dict:
    """Load user context from DB. Returns a dict (never None)."""
    with get_db() as conn:
        db_execute(conn, "SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = conn.fetchone() if hasattr(conn, 'fetchone') else conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        
        # Safe fetch for both SQLite and Postgres
        if not row:
            return {"name": None, "language": "english", "topics": [], "mood_history": [], "details": []}
            
        return {
            "name":         row["name"],
            "language":     row["language"] or "english",
            "topics":       json.loads(row["topics"] or "[]"),
            "mood_history": json.loads(row["mood_history"] or "[]"),
            "details":      json.loads(row["details"] or "[]"),
        }
 
 
def db_upsert_user(user_id: str, ctx: dict):
    """Save or update user context in DB."""
    now = time.time()
    with get_db() as conn:
        query = """
            INSERT INTO users (user_id, name, language, topics, mood_history, details, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                name         = excluded.name,
                language     = excluded.language,
                topics       = excluded.topics,
                mood_history = excluded.mood_history,
                details      = excluded.details,
                updated_at   = excluded.updated_at
        """
        db_execute(conn, query, (
            user_id,
            ctx.get("name"),
            ctx.get("language", "english"),
            json.dumps(ctx.get("topics", [])),
            json.dumps(ctx.get("mood_history", [])),
            json.dumps(ctx.get("details", [])),
            now, now
        ))
 
 
def db_get_messages(user_id: str, limit: int = MAX_TURNS) -> List[dict]:
    """Load last N messages for a user, oldest first."""
    with get_db() as conn:
        query = "SELECT role, content FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?"
        db_execute(conn, query, (user_id, limit))
        rows = conn.fetchall() if hasattr(conn, 'fetchall') else conn.execute(query, (user_id, limit)).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
 
 
def db_add_message(user_id: str, role: str, content: str):
    """Append a single message to DB."""
    with get_db() as conn:
        query = "INSERT INTO messages (user_id, role, content, created_at) VALUES (?, ?, ?, ?)"
        db_execute(conn, query, (user_id, role, content, time.time()))
 
 
def db_clear_user(user_id: str):
    """Delete all messages for a user (reset conversation). Keep user profile."""
    with get_db() as conn:
        db_execute(conn, "DELETE FROM messages WHERE user_id = ?", (user_id,))
 
 
def db_full_reset_user(user_id: str):
    """Delete messages AND user profile entirely."""
    with get_db() as conn:
        db_execute(conn, "DELETE FROM messages WHERE user_id = ?", (user_id,))
        db_execute(conn, "DELETE FROM users WHERE user_id = ?", (user_id,))
 
 
def db_add_diary_entry(user_id: str, raw_chat: str, emotion_label: str, sentiment_score: float):
    """Auto-save a diary entry from chat."""
    with get_db() as conn:
        query = "INSERT INTO diary_entries (user_id, raw_chat, emotion_label, sentiment_score, timestamp) VALUES (?,?,?,?,?)"
        db_execute(conn, query, (user_id, raw_chat, emotion_label, round(sentiment_score, 3), time.time()))
 
 
def db_get_diary(user_id: str) -> List[dict]:
    """Return all diary entries for a user, newest first."""
    with get_db() as conn:
        query = "SELECT id, user_id, raw_chat, emotion_label, sentiment_score, timestamp FROM diary_entries WHERE user_id=? ORDER BY timestamp DESC"
        db_execute(conn, query, (user_id,))
        rows = conn.fetchall() if hasattr(conn, 'fetchall') else conn.execute(query, (user_id,)).fetchall()
        return [dict(r) for r in rows]


def db_admin_get_all_users() -> List[dict]:
    """Admin only: list all users who have ever chatted."""
    with get_db() as conn:
        query = """
            SELECT u.user_id, u.name, u.updated_at, COUNT(m.id) as msg_count
            FROM users u
            LEFT JOIN messages m ON u.user_id = m.user_id
            GROUP BY u.user_id, u.name, u.updated_at
            ORDER BY u.updated_at DESC
        """
        db_execute(conn, query)
        rows = conn.fetchall() if hasattr(conn, 'fetchall') else conn.execute(query).fetchall()
        return [dict(r) for r in rows]


def db_admin_get_full_history(user_id: str) -> List[dict]:
    """Admin only: get every single message for a user."""
    with get_db() as conn:
        query = "SELECT role, content, created_at FROM messages WHERE user_id = ? ORDER BY created_at ASC"
        db_execute(conn, query, (user_id,))
        rows = conn.fetchall() if hasattr(conn, 'fetchall') else conn.execute(query, (user_id,)).fetchall()
        return [dict(r) for r in rows]
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT + CONTEXT
# ══════════════════════════════════════════════════════════════════════════════
 
def build_system_prompt(ctx: dict, crisis: bool = False) -> str:
    name         = ctx.get("name") or "there"
    topics       = ctx.get("topics", [])
    mood_history = ctx.get("mood_history", [])
    details      = ctx.get("details", [])
 
    context_block = ""
    if name != "there":
        context_block += f"- User's name: {name}\n"
    if topics:
        context_block += f"- Topics so far: {', '.join(topics[-5:])}\n"
    if mood_history:
        context_block += f"- Mood trend: {', '.join(mood_history[-5:])}\n"
    if details:
        context_block += f"- Known details: {'; '.join(details[-8:])}\n"
 
    crisis_instruction = ""
    if crisis:
        crisis_instruction = """
CRISIS MODE — VERY IMPORTANT:
- The user has expressed thoughts of suicide or self-harm.
- Do NOT end the conversation. Do NOT say "I cannot continue".
- Share the crisis helpline number naturally in your reply: Kiran Mental Health Helpline: 9152987821 (Free · 24/7)
- Stay warm, present, and human. Acknowledge their pain directly.
- Gently encourage them to call or reach out to someone, but keep talking to them.
- Ask one caring question to understand what they're going through.
- Example tone: "That sounds incredibly painful. You don't have to face this alone — please consider calling Kiran at 9152987821, they're available 24/7 and it's free. Can you tell me a bit about what's been happening?"
"""
 
    context_section = f"""
What you know about this user:
{context_block}""" if context_block else ""
 
    return f"""You are Haven, a warm emotional-support companion for undergraduate students in India.
 
LANGUAGE RULE (CRITICAL — follow this strictly):
- Detect the language the user is writing in RIGHT NOW and reply in EXACTLY that language.
- User writes in ENGLISH → reply in ENGLISH only. Zero Hindi words.
- User writes in HINDI (Devanagari script) → reply in HINDI only.
- User deliberately mixes Hindi + English (Hinglish like "yaar I'm so stressed") → mirror that mix.
- DEFAULT is ENGLISH. Never switch on your own — always follow the user's lead.
 
MEMORY RULE (CRITICAL):
- You have permanent memory of this user. Use it naturally.
- If you know their name, mood history, or topics — reference them organically.
- Example: "Last time you mentioned your exams were stressing you out — how did that go?"
- Never mention that you "looked up" their history. Just use it naturally like a real friend would.
 
CONVERSATION STYLE:
- Talk like a warm caring friend, NOT a therapist or AI assistant.
- Keep replies to 2–4 sentences. Be concise and natural.
- NEVER use: "I'm always here for you", "I understand how you feel", "That must be hard", "I cannot assist with".
- CRITICAL: Do NOT end every reply with a question. Vary your responses:
  * Sometimes just validate what they said ("Yeah, that really sucks." / "That makes total sense.")
  * Sometimes share a relatable thought or gentle observation.
  * Sometimes suggest a small, practical thing they could try.
  * Only ask a follow-up question about 1 in every 3-4 replies, and only when genuinely needed.
- If the user shares something heavy, sit with it. Don't immediately redirect with a question.
- Never repeat phrasing you used earlier in this conversation.
- If you know the user's name, use it occasionally — not every message.
- Reference what you know about them to make them feel heard and remembered.
- Leave space for them to share more without always prompting them.
- Do NOT give medical diagnoses or prescribe anything.
- Match the user's energy — if they're brief, be brief. If they're venting long, give a fuller response.
{crisis_instruction}{context_section}"""
 
 
# ══════════════════════════════════════════════════════════════════════════════
# LANGUAGE + CONTEXT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
 
def detect_language(text: str) -> str:
    if any('\u0900' <= c <= '\u097F' for c in text):
        return "hindi"
    hinglish = ["yaar","bhai","kya","nahi","bahut","hua","hai","hoon",
                "mujhe","toh","aur","kal","abhi","thoda","bohot","kuch",
                "sab","accha","theek","nahi","tera","mera","acha","bilkul"]
    tl = text.lower()
    if any(w in tl.split() for w in hinglish):
        return "hinglish"
    return "english"
 
 
def extract_context_from_message(text: str, ctx: dict) -> dict:
    """Update ctx dict in-place with info extracted from message. Returns updated ctx."""
    tl = text.lower()
    ctx["language"] = detect_language(text)
 
    topic_map = {
        "exams":        ["exam","exams","test","paper","marks","result","pariksha","score"],
        "math":         ["math","maths","calculus","algebra","mathematics","statistics"],
        "family":       ["family","mom","dad","parents","sister","brother","ghar","mummy","papa","bhaiya","didi"],
        "friends":      ["friend","friends","bestie","yaar","dost","classmate"],
        "anxiety":      ["anxious","anxiety","panic","nervous","overthinking","phobia"],
        "depression":   ["depressed","depression","sad","crying","empty","numb","hollow"],
        "sleep":        ["sleep","insomnia","tired","exhausted","neend"],
        "relationship": ["relationship","boyfriend","girlfriend","breakup","crush","love","heartbreak"],
        "college":      ["college","university","semester","assignment","project","professor","faculty"],
        "career":       ["job","career","placement","internship","future","campus"],
        "stress":       ["stress","stressed","pressure","burden","tension","overwhelmed"],
        "loneliness":   ["lonely","alone","isolated","no one","nobody"],
    }
    for topic, kws in topic_map.items():
        if any(k in tl for k in kws) and topic not in ctx["topics"]:
            ctx["topics"].append(topic)
 
    mood_map = {
        "sad":      ["sad","dukhi","unhappy","crying","tears"],
        "stressed": ["stressed","stress","tension","pressure","overwhelmed"],
        "anxious":  ["anxious","nervous","scared","worried","panic","darr"],
        "angry":    ["angry","frustrated","annoyed","irritated","gusse"],
        "happy":    ["happy","good","great","better","fine","khush","acha"],
        "hopeless": ["hopeless","give up","no point","lost","koi fayda nahi"],
        "tired":    ["tired","exhausted","drained","thaka","thaki"],
        "relieved": ["relieved","better now","feeling good","thanks","helped"],
    }
    for mood, kws in mood_map.items():
        if any(k in tl for k in kws):
            if not ctx["mood_history"] or ctx["mood_history"][-1] != mood:
                ctx["mood_history"].append(mood)
 
    detail_patterns = [
        ("final year student",    ["final year","4th year","fourth year"]),
        ("3rd year student",      ["3rd year","third year"]),
        ("2nd year student",      ["2nd year","second year"]),
        ("1st year student",      ["1st year","first year","fresher"]),
        ("struggling with math",  ["math is stressing","maths is hard","maths mein problem"]),
        ("going through breakup", ["breakup","broke up","she left","he left"]),
        ("exam pressure",         ["exam pressure","exam stress","pariksha ka darr"]),
        ("feeling lonely",        ["feeling lonely","no one to talk","akela"]),
    ]
    for detail, pats in detail_patterns:
        if any(p in tl for p in pats) and detail not in ctx["details"]:
            ctx["details"].append(detail)
 
    nm = re.search(
        r"(?:i am|i'm|my name is|mera naam hai|call me|naam hai mera)\s+([A-Z][a-z]+)",
        text, re.IGNORECASE
    )
    if nm and not ctx.get("name"):
        ctx["name"] = nm.group(1).strip()
 
    return ctx
 
 
# ══════════════════════════════════════════════════════════════════════════════
# MuRIL EMOTION CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
 
MURIL_BASE     = os.getenv("MURIL_BASE", "google/muril-base-cased")
EMOTION_LABELS = ["joy","sadness","fear","anger","surprise","neutral","disgust","shame"]
EMOJI_MAP      = {"joy":"😊","sadness":"😢","fear":"😰","anger":"😠",
                  "surprise":"😲","neutral":"😐","disgust":"🤢","shame":"😳"}
EMOTION_TO_SENTIMENT = {
    "joy":(0.80,0.90),"surprise":(0.30,0.70),"neutral":(0.00,0.20),
    "shame":(-0.30,0.90),"disgust":(-0.40,0.80),"fear":(-0.50,0.80),
    "sadness":(-0.65,0.90),"anger":(-0.80,0.90),
}
 
_tokenizer = None
_muril_model = None
_head_loaded = False
_device = "cpu"
 
 
def _load_muril():
    global _tokenizer, _muril_model, _head_loaded, _device
    if _tokenizer is not None:
        return
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MuRIL] device: {_device}")
    _tokenizer = AutoTokenizer.from_pretrained(MURIL_BASE)
 
    class MuRILEmotionClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.muril = AutoModel.from_pretrained(MURIL_BASE)
            for p in self.muril.embeddings.parameters():
                p.requires_grad = False
            self.drop = nn.Dropout(0.4)
            self.clf = nn.Sequential(
                nn.Linear(768, 512), nn.ReLU(), nn.Dropout(0.4), nn.BatchNorm1d(512),
                nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4), nn.BatchNorm1d(256),
                nn.Linear(256, len(EMOTION_LABELS))
            )
 
        def forward(self, ids, mask):
            o = self.muril(input_ids=ids, attention_mask=mask, return_dict=True)
            return self.clf(self.drop(o.last_hidden_state[:, 0]))
 
    _muril_model = MuRILEmotionClassifier().to(_device)
    _muril_model.eval()
    _here = os.path.dirname(os.path.abspath(__file__))
    for path in [
        os.getenv("MURIL_EMOTION_WEIGHTS", ""),
        os.path.join(_here, "models", "muril_emotion_model.pth"),
        os.path.join(_here, "muril_emotion_model.pth"),
    ]:
        if path and os.path.exists(path):
            try:
                import torch
                s = torch.load(path, map_location=_device)
                if isinstance(s, dict) and "state_dict" in s:
                    s = s["state_dict"]
                _muril_model.load_state_dict(s, strict=False)
                _head_loaded = True
                print(f"[MuRIL] ✅ {path}")
                break
            except Exception as ex:
                print(f"[MuRIL] ⚠️ {ex}")
    if not _head_loaded:
        print("[MuRIL] ⚠️ No weights — returning Neutral.")
 
 
def detect_emotion(text: str) -> Tuple[str, float, str, str]:
    try:
        result = mindmate_analyzer.predict_emotion(text)
        label = result['emotion']
        return label, result['confidence'], result['emoji'], ""
    except Exception as e:
        return "neutral", 0.01, "😐", str(e)
    # Keyword fallback when no weights loaded
    tl = text.lower()
    if any(w in tl for w in ["angry","anger","furious","frustrated","tired","annoyed"]):
        return "anger", 0.85, "😠", ""
    if any(w in tl for w in ["sad","crying","depressed","unhappy","lonely"]):
        return "sadness", 0.85, "😢", ""
    if any(w in tl for w in ["happy","great","excited","joy","wonderful"]):
        return "joy", 0.85, "😊", ""
    if any(w in tl for w in ["scared","afraid","worried","anxious","fear"]):
        return "fear", 0.85, "😰", ""
    return "neutral", 0.50, "😐", ""
 
 
def detect_sentiment(em: str) -> Tuple[float, float, str]:
    pol, sub = EMOTION_TO_SENTIMENT.get(em, (0.0, 0.2))
    return pol, sub, ("Positive" if pol > 0.15 else "Negative" if pol < -0.15 else "Neutral")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# GROQ CHAT
# ══════════════════════════════════════════════════════════════════════════════
 
_groq_client = None
 
 
def get_groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client
 
 
def _is_crisis(text: str) -> bool:
    tl = (text or "").lower()
    return any(w in tl for w in CRISIS_WORDS)
 
 
def groq_chat(user_id: str, user_message: str, persona_hint: str = "") -> str:
    if not GROQ_API_KEY:
        return "⚠️ GROQ_API_KEY missing. Add it to backend/.env and restart."
 
    crisis = _is_crisis(user_message)
 
    # 1. Load user context from DB
    ctx = db_get_user(user_id)
 
    # 2. Extract new context from this message and update ctx
    ctx = extract_context_from_message(user_message, ctx)
 
    # 3. Save updated context back to DB
    db_upsert_user(user_id, ctx)
 
    # 4. Save the user message to DB
    db_add_message(user_id, "user", user_message)
 
    # 5. Load conversation history from DB
    history = db_get_messages(user_id, limit=MAX_TURNS)
 
    # 6. Build system prompt with current context
    system_prompt = build_system_prompt(ctx, crisis=crisis)
    # Inject persona hint if provided
    if persona_hint:
        system_prompt += f"\n\nPERSONA OVERRIDE:\n{persona_hint}"
    messages = [{"role": "system", "content": system_prompt}] + history
 
    # 7. Call Groq
    try:
        r = get_groq().chat.completions.create(
            model=GROQ_MODEL, messages=messages,
            temperature=0.85, max_tokens=260, top_p=0.95
        )
        reply = r.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Groq] {e}")
        reply = "I got a little glitchy — could you say that again?"
 
    # 8. Save assistant reply to DB
    db_add_message(user_id, "assistant", reply)
 
    return reply
 
 
# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════
 
app = FastAPI(title="MyHaven Backend", version="6.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
 
 
# Initialize DB on startup
@app.on_event("startup")
def startup_event():
    init_db()
    print("[Startup] ✅ MyHaven Backend v6.0 ready")
 
 
# ── Request Models ────────────────────────────────────────────────────────────
 
class UserReq(BaseModel):
    user_id: str = Field(..., min_length=1)
 
 
class ChatReq(BaseModel):
    user_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    persona_hint: str = Field(default="")


class SetNameReq(BaseModel):
    user_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
 
 
# ── Endpoints ─────────────────────────────────────────────────────────────────
 
@app.post("/start")
def start(req: UserReq):
    """
    Begin a new session. 
    - If user has been here before: loads their profile, sends a personalized greeting.
    - If new user: creates profile, sends default greeting.
    Note: does NOT clear chat history — that's intentional for memory persistence.
    """
    ctx = db_get_user(req.user_id)
    name = ctx.get("name")
    topics = ctx.get("topics", [])
    mood_history = ctx.get("mood_history", [])
 
    # Build a personalized greeting if we know the user
    if name:
        if mood_history:
            last_mood = mood_history[-1]
            greeting = f"Hey {name}! 💛 Welcome back. Last time you seemed {last_mood} — how are you feeling today?"
        elif topics:
            last_topic = topics[-1]
            greeting = f"Hey {name}! 💛 Good to have you back. We were talking about {last_topic} last time — what's on your mind today?"
        else:
            greeting = f"Hey {name}! 💛 Good to see you again. What's been going on?"
    else:
        greeting = "Hey! I'm Haven 💛 What's been weighing on your mind lately — college stuff, relationships, family, or something else?"
 
    # Save greeting to DB
    db_upsert_user(req.user_id, ctx)
    db_add_message(req.user_id, "assistant", greeting)
 
    return {"reply": greeting, "returning_user": bool(name), "user_name": name}
 
 
@app.post("/set_name")
def set_name(req: SetNameReq):
    """Save the user's name to their profile."""
    ctx = db_get_user(req.user_id)
    ctx["name"] = req.name
    db_upsert_user(req.user_id, ctx)
    return {"status": "success", "user_name": req.name}
 
 
@app.post("/chat")
def chat(req: ChatReq):
    """Send a message, get a reply with emotion + sentiment analysis."""
    reply = groq_chat(req.user_id, req.message, persona_hint=req.persona_hint)
    el, ec, ee, en = detect_emotion(req.message)
    pol, sub, sl = detect_sentiment(el)
    ctx = db_get_user(req.user_id)
 
    # ── Auto-save to diary ────────────────────────────────────────────
    db_add_diary_entry(req.user_id, req.message, el, pol)
    # ─────────────────────────────────────────────────────────────────
 
    return {
        "reply": reply,
        "emotion": {
            "label": el.title(),
            "confidence": ec,
            "emoji": ee,
            "note": en
        },
        "sentiment": {
            "label": sl,
            "polarity": round(pol, 3),
            "subjectivity": round(sub, 3),
            "note": "" if _head_loaded else "Derived from emotion mapping (no trained weights yet)."
        },
        "context": {
            "name": ctx.get("name"),
            "language": ctx.get("language", "english"),
            "topics": ctx.get("topics", []),
            "mood_history": ctx.get("mood_history", [])
        },
        "meta": {
            "groq_model": GROQ_MODEL,
            "muril_loaded": _head_loaded
        },
    }
 
 
@app.post("/reset")
def reset(req: UserReq):
    """Clear conversation history but KEEP user profile (name, topics, mood)."""
    db_clear_user(req.user_id)
    return {"ok": True, "message": "Conversation cleared. User profile kept."}
 
 
@app.post("/full_reset")
def full_reset(req: UserReq):
    """Wipe EVERYTHING for this user — messages AND profile."""
    db_full_reset_user(req.user_id)
    return {"ok": True, "message": "All data deleted for this user."}
 
 
@app.get("/history/{user_id}")
def get_history(user_id: str, limit: int = 20):
    """Get recent chat history for a user (useful for debugging)."""
    messages = db_get_messages(user_id, limit=limit)
    ctx = db_get_user(user_id)
    return {
        "user_id": user_id,
        "profile": ctx,
        "messages": messages,
        "count": len(messages)
    }
 
 
@app.get("/diary/{user_id}")
def get_diary(user_id: str):
    """Return all auto-captured diary entries for a user (newest first)."""
    return db_get_diary(user_id)
 
 
@app.get("/diary/{user_id}/export")
def export_diary_csv(user_id: str):
    """Download all diary entries as a CSV file."""
    entries = db_get_diary(user_id)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["user_id", "raw_chat", "emotion_label", "sentiment_score", "timestamp"])
    for e in entries:
        writer.writerow([
            e["user_id"],
            e["raw_chat"],
            e["emotion_label"],
            e["sentiment_score"],
            datetime.fromtimestamp(e["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        ])
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=diary_{user_id}.csv"}
    )
 
 
@app.get("/health")
def health():
    """Check backend status."""
    return {
        "ok": True,
        "time": time.time(),
        "groq_model": GROQ_MODEL,
        "muril_loaded": _head_loaded,
        "api_key_set": bool(GROQ_API_KEY),
        "db_path": DB_PATH,
        "version": "6.1" # Incremented for Admin Dashboard
    }


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN DASHBOARD ROUTES
# ══════════════════════════════════════════════════════════════════════════════

ADMIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Haven | Admin Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Outfit', sans-serif; }
    </style>
</head>
<body class="bg-[#f8f9ff] text-slate-800 min-h-screen">
    <nav class="bg-indigo-600 text-white p-4 shadow-lg">
        <div class="max-w-6xl mx-auto flex justify-between items-center">
            <h1 class="text-xl font-bold flex items-center gap-2">
                <span class="p-1.5 bg-white/20 rounded-lg">🛡️</span> Haven Admin
            </h1>
            <div class="flex items-center gap-2">
                <span class="text-xs opacity-75">Production Mode</span>
                <div class="size-2 bg-green-400 rounded-full animate-pulse"></div>
            </div>
        </div>
    </nav>

    <main class="max-w-6xl mx-auto p-8">
        <div id="auth-section" class="flex flex-col items-center justify-center pt-20">
            <div class="bg-white p-8 rounded-3xl shadow-xl border border-gray-100 w-full max-w-md text-center">
                <h2 class="text-2xl font-bold mb-4 text-indigo-900">Sign In</h2>
                <p class="text-gray-500 mb-8 text-sm">Enter your master admin key to view chat data.</p>
                <input type="password" id="admin-key" placeholder="Admin Key" class="w-full px-5 py-4 bg-gray-50 border border-gray-200 rounded-2xl mb-4 focus:ring-2 focus:ring-indigo-500 outline-none transition-all">
                <button onclick="loadDashboard()" class="w-full bg-indigo-600 hover:bg-indigo-700 text-white py-4 rounded-2xl font-bold shadow-lg shadow-indigo-200 transition-all">Access Dashboard</button>
            </div>
        </div>

        <div id="dash-section" class="hidden">
            <div class="flex justify-between items-end mb-8">
                <div>
                    <h2 class="text-3xl font-bold text-slate-900 mb-1">User Activity</h2>
                    <p class="text-slate-500">Monitoring all conversations and interactions.</p>
                </div>
                <button onclick="window.location.reload()" class="px-4 py-2 bg-white border border-gray-200 rounded-xl text-sm font-semibold text-slate-600 hover:bg-gray-50 transition-all">Refresh Data</button>
            </div>

            <div class="bg-white rounded-3xl shadow-xl overflow-hidden border border-gray-100 mb-12">
                <table class="w-full text-left">
                    <thead class="bg-slate-50 text-slate-400 text-[10px] font-bold uppercase tracking-wider">
                        <tr>
                            <th class="px-6 py-4">User Name</th>
                            <th class="px-6 py-4">User ID</th>
                            <th class="px-6 py-4">Last Active</th>
                            <th class="px-6 py-4">Messages</th>
                            <th class="px-6 py-4">Actions</th>
                        </tr>
                    </thead>
                    <tbody id="user-rows" class="divide-y divide-gray-100"></tbody>
                </table>
            </div>

            <div id="history-section" class="hidden animate-in fade-in slide-in-from-bottom-4">
                <div class="flex items-center gap-3 mb-4">
                    <h3 class="text-xl font-bold text-slate-900">Conversation History</h3>
                    <span id="target-user" class="px-3 py-1 bg-indigo-50 text-indigo-600 text-xs font-bold rounded-full"></span>
                </div>
                <div id="chat-reel" class="bg-[#111122] rounded-3xl p-8 max-h-[600px] overflow-y-auto flex flex-col gap-4 text-sm scroll-smooth"></div>
            </div>
        </div>
    </main>

    <script>
        let GLOBAL_KEY = "";

        async function loadDashboard() {
            const key = document.getElementById('admin-key').value;
            if(!key) return alert("Enter key");
            
            try {
                const res = await fetch(`/admin/api/users?key=${key}`);
                const data = await res.json();
                if(!res.ok) throw new Error(data.detail || "Error");
                
                GLOBAL_KEY = key;
                document.getElementById('auth-section').classList.add('hidden');
                document.getElementById('dash-section').classList.remove('hidden');
                renderUsers(data);
            } catch(e) { alert(e.message); }
        }

        function renderUsers(users) {
            const tbody = document.getElementById('user-rows');
            tbody.innerHTML = users.map(u => `
                <tr class="hover:bg-indigo-50/30 transition-colors">
                    <td class="px-6 py-5 font-bold text-slate-900">${u.name || 'Anonymous'}</td>
                    <td class="px-6 py-5 font-mono text-xs text-slate-400">${u.user_id}</td>
                    <td class="px-6 py-5 text-sm text-slate-500">${new Date(u.updated_at*1000).toLocaleString()}</td>
                    <td class="px-6 py-5">
                        <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">${u.msg_count}</span>
                    </td>
                    <td class="px-6 py-5 text-right">
                        <button onclick="loadUserHistory('${u.user_id}', '${u.name || 'Anonymous'}')" class="text-indigo-600 font-bold hover:underline">View Transcript</button>
                    </td>
                </tr>
            `).join('');
        }

        async function loadUserHistory(uid, name) {
            document.getElementById('history-section').classList.remove('hidden');
            document.getElementById('target-user').textContent = name;
            const reel = document.getElementById('chat-reel');
            reel.innerHTML = '<p class="text-slate-500 italic">Decoding history...</p>';
            
            try {
                const res = await fetch(`/admin/api/history/${uid}?key=${GLOBAL_KEY}`);
                const data = await res.json();
                reel.innerHTML = data.map(m => `
                    <div class="flex flex-col ${m.role === 'user' ? 'items-end' : 'items-start'}">
                        <span class="text-[10px] uppercase font-bold text-slate-500 mb-1">${m.role}</span>
                        <div class="max-w-[85%] p-4 rounded-2xl ${m.role === 'user' ? 'bg-indigo-600 text-white rounded-tr-none' : 'bg-[#1e1e30] text-indigo-100 rounded-tl-none'}">
                            ${m.content}
                        </div>
                    </div>
                `).join('');
                document.getElementById('history-section').scrollIntoView();
            } catch(e) { alert(e.message); }
        }
    </script>
</body>
</html>
"""

@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    return ADMIN_HTML

@app.get("/admin/api/users")
def admin_api_users(key: str):
    if key != HAVEN_ADMIN_KEY:
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Invalid admin key")
    return db_admin_get_all_users()

@app.get("/admin/api/history/{user_id}")
def admin_api_history(user_id: str, key: str):
    if key != HAVEN_ADMIN_KEY:
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Invalid admin key")
    return db_admin_get_full_history(user_id)