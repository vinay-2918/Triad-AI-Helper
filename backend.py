from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import fitz  # PyMuPDF
import requests
import psutil
import platform
import datetime
import sqlite3
import secrets
import bcrypt
import json
import os

# ── Load secrets ───────────────────────────────────────────────────────────────
load_dotenv(".env.txt")
GROQ_API_KEY          = os.getenv("GROQ_API_KEY")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
GEMINI_API_KEY        = os.getenv("GEMINI_API_KEY", "")
HF_API_KEY            = os.getenv("HF_API_KEY", "")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in .env.txt")

# ── AI Provider Setup ──────────────────────────────────────────────────────────
#
#  Fallback chain (tried in order until one succeeds):
#
#  1. Groq  — LLaMA 3.3 70B          (primary, fastest)
#  2. Groq  — LLaMA 3.1 8B           (same key, lighter model)
#  3. Groq  — Gemma2 9B              (same key, Google model via Groq)
#  4. Gemini 1.5 Flash               (Google free tier — add GEMINI_API_KEY)
#  5. Hugging Face — Mistral 7B      (completely free — add HF_API_KEY)
#
#  To get free API keys (optional but recommended):
#    Gemini  → https://aistudio.google.com/app/apikey
#    HF      → https://huggingface.co/settings/tokens
#  Add them to your .env.txt:
#    GEMINI_API_KEY=your_key_here
#    HF_API_KEY=your_key_here

groq_client = Groq(api_key=GROQ_API_KEY)

GROQ_MODELS = [
    "llama-3.3-70b-versatile",   # Best quality — primary
    "llama-3.1-8b-instant",      # Lighter/faster — first fallback
    "gemma2-9b-it",              # Google Gemma via Groq — second fallback
]

app = FastAPI(title="Triad AI Backend")

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve index.html at /ui ────────────────────────────────────────────────────
if os.path.exists("index.html"):
    app.mount("/static", StaticFiles(directory="."), name="static")

    @app.get("/ui")
    def serve_ui():
        return FileResponse("index.html")


# ── Database helpers ───────────────────────────────────────────────────────────
DB_PATH = "users.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create tables if they don't exist yet."""
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    NOT NULL UNIQUE,
            email       TEXT    NOT NULL UNIQUE,
            password    TEXT    NOT NULL,
            full_name   TEXT    DEFAULT '',
            avatar_url  TEXT    DEFAULT '',
            created_at  TEXT    DEFAULT (datetime('now')),
            updated_at  TEXT    DEFAULT (datetime('now'))
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            token       TEXT    PRIMARY KEY,
            user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at  TEXT    DEFAULT (datetime('now')),
            expires_at  TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def clear_all_sessions():
    """
    Called once on every server startup.
    Deletes ALL sessions so every browser must log in fresh.
    This is the correct behaviour — login is required each time
    the server is started/restarted.
    """
    conn = get_db()
    deleted = conn.execute("DELETE FROM sessions").rowcount
    conn.commit()
    conn.close()
    if deleted:
        print(f"🔒 Cleared {deleted} old session(s) — all users must log in again.")
    else:
        print("🔒 No existing sessions to clear.")

init_db()
clear_all_sessions()


# ── Auth helpers ───────────────────────────────────────────────────────────────

SESSION_DAYS = 30

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

def check_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())

def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(48)
    expires = (datetime.datetime.utcnow() + datetime.timedelta(days=SESSION_DAYS)).isoformat()
    conn = get_db()
    conn.execute(
        "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
        (token, user_id, expires)
    )
    conn.commit()
    conn.close()
    return token

def get_user_by_token(token: str):
    if not token:
        return None
    conn = get_db()
    row = conn.execute(
        """SELECT u.* FROM users u
           JOIN sessions s ON s.user_id = u.id
           WHERE s.token = ? AND s.expires_at > datetime('now')""",
        (token,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None

def require_auth(authorization: str = Header(default=None)):
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
    user = get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return user


# ── Auth request models ────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    full_name: str = ""

class LoginRequest(BaseModel):
    email: str
    password: str

class UpdateProfileRequest(BaseModel):
    username:   str  | None = None
    email:      str  | None = None
    full_name:  str  | None = None
    avatar_url: str  | None = None

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password:     str


# ── Auth Routes ────────────────────────────────────────────────────────────────

@app.post("/auth/register")
def register(req: RegisterRequest):
    if len(req.username.strip()) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters.")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    if "@" not in req.email:
        raise HTTPException(status_code=400, detail="Invalid email address.")

    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM users WHERE email = ? OR username = ?",
        (req.email.lower(), req.username.lower())
    ).fetchone()
    if existing:
        conn.close()
        raise HTTPException(status_code=409, detail="Email or username already in use.")

    hashed = hash_password(req.password)
    cur = conn.execute(
        "INSERT INTO users (username, email, password, full_name) VALUES (?, ?, ?, ?)",
        (req.username.strip(), req.email.lower().strip(), hashed, req.full_name.strip())
    )
    user_id = cur.lastrowid
    conn.commit()
    conn.close()

    token = create_session(user_id)
    return {
        "token": token,
        "user": {
            "id": user_id,
            "username": req.username.strip(),
            "email": req.email.lower().strip(),
            "full_name": req.full_name.strip(),
            "avatar_url": "",
        }
    }


@app.post("/auth/login")
def login(req: LoginRequest):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM users WHERE email = ?", (req.email.lower().strip(),)
    ).fetchone()
    conn.close()

    if not row or not check_password(req.password, row["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = create_session(row["id"])
    return {
        "token": token,
        "user": {
            "id":         row["id"],
            "username":   row["username"],
            "email":      row["email"],
            "full_name":  row["full_name"],
            "avatar_url": row["avatar_url"],
        }
    }


@app.get("/auth/me")
def me(authorization: str = Header(default=None)):
    user = require_auth(authorization)
    return {
        "id":         user["id"],
        "username":   user["username"],
        "email":      user["email"],
        "full_name":  user["full_name"],
        "avatar_url": user["avatar_url"],
        "created_at": user["created_at"],
    }


@app.post("/auth/logout")
def logout(authorization: str = Header(default=None)):
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        conn = get_db()
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
        conn.close()
    return {"message": "Logged out."}


@app.put("/auth/profile")
def update_profile(req: UpdateProfileRequest, authorization: str = Header(default=None)):
    user = require_auth(authorization)

    updates, params = [], []
    if req.username is not None:
        if len(req.username.strip()) < 3:
            raise HTTPException(status_code=400, detail="Username must be at least 3 characters.")
        # Check uniqueness
        conn = get_db()
        clash = conn.execute(
            "SELECT id FROM users WHERE username = ? AND id != ?",
            (req.username.strip(), user["id"])
        ).fetchone()
        conn.close()
        if clash:
            raise HTTPException(status_code=409, detail="Username already taken.")
        updates.append("username = ?"); params.append(req.username.strip())

    if req.email is not None:
        if "@" not in req.email:
            raise HTTPException(status_code=400, detail="Invalid email address.")
        conn = get_db()
        clash = conn.execute(
            "SELECT id FROM users WHERE email = ? AND id != ?",
            (req.email.lower().strip(), user["id"])
        ).fetchone()
        conn.close()
        if clash:
            raise HTTPException(status_code=409, detail="Email already in use.")
        updates.append("email = ?"); params.append(req.email.lower().strip())

    if req.full_name is not None:
        updates.append("full_name = ?"); params.append(req.full_name.strip())

    if req.avatar_url is not None:
        updates.append("avatar_url = ?"); params.append(req.avatar_url.strip())

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update.")

    updates.append("updated_at = datetime('now')")
    params.append(user["id"])

    conn = get_db()
    conn.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = ?", params)
    conn.commit()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user["id"],)).fetchone()
    conn.close()

    return {
        "id":         row["id"],
        "username":   row["username"],
        "email":      row["email"],
        "full_name":  row["full_name"],
        "avatar_url": row["avatar_url"],
    }


@app.put("/auth/password")
def change_password(req: ChangePasswordRequest, authorization: str = Header(default=None)):
    user = require_auth(authorization)

    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user["id"],)).fetchone()
    conn.close()

    if not check_password(req.current_password, row["password"]):
        raise HTTPException(status_code=401, detail="Current password is incorrect.")
    if len(req.new_password) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters.")

    hashed = hash_password(req.new_password)
    conn = get_db()
    conn.execute("UPDATE users SET password = ?, updated_at = datetime('now') WHERE id = ?", (hashed, user["id"]))
    # Invalidate all other sessions for security
    conn.execute("DELETE FROM sessions WHERE user_id = ? AND token != ?",
                 (user["id"], authorization[7:]))
    conn.commit()
    conn.close()

    return {"message": "Password updated successfully."}


# ── Original request models ─────────────────────────────────────────────────────

class VoiceCommandRequest(BaseModel):
    command: str

class TravelRequest(BaseModel):
    origin: str
    destination: str
    days: int
    travelers: int
    budget: str

class AIQuestionRequest(BaseModel):
    question: str

class LegalTextRequest(BaseModel):
    text: str

class MedicalTextRequest(BaseModel):
    text: str


# ── Utilities ──────────────────────────────────────────────────────────────────

def get_public_ip():
    try:
        return requests.get("https://api.ipify.org", timeout=5).text
    except Exception:
        return None

def get_battery_status():
    try:
        battery = psutil.sensors_battery()
        if battery:
            percent = round(battery.percent)
            status  = "charging" if battery.power_plugged else "not charging"
            return f"Battery is at {percent} percent and it is {status}."
    except Exception:
        pass
    return "Battery status is not available on this device."

def get_system_details():
    s = platform.uname()
    return (f"System: {s.system}, Device: {s.node}, "
            f"Release: {s.release}, Processor: {s.processor or 'unknown'}.")

def get_datetime():
    now = datetime.datetime.now()
    return (f"Today is {now.strftime('%A, %d %B %Y')} "
            f"and the time is {now.strftime('%I:%M %p')}.")

# ── Multi-provider AI with automatic fallback ─────────────────────────────────

def _try_groq(prompt: str, max_tokens: int, model: str) -> str:
    """Attempt a single Groq model call."""
    resp = groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


def _try_gemini(prompt: str, max_tokens: int) -> str:
    """Google Gemini 1.5 Flash — free tier, 15 req/min."""
    if not GEMINI_API_KEY:
        raise ValueError("No GEMINI_API_KEY configured")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.7},
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def _try_huggingface(prompt: str, max_tokens: int) -> str:
    """Hugging Face Inference API — Mistral 7B, completely free."""
    if not HF_API_KEY:
        raise ValueError("No HF_API_KEY configured")
    url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": min(max_tokens, 1024), "temperature": 0.7},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and data:
        text = data[0].get("generated_text", "")
        # Strip the echoed prompt from the response
        return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
    raise ValueError("Unexpected HF response format")


def ai_ask(prompt: str, max_tokens: int = 300) -> str:
    """
    Smart AI caller with automatic fallback chain.
    Tries each provider in order — returns first successful response.

    Order:
      1. Groq LLaMA 3.3 70B  (best quality)
      2. Groq LLaMA 3.1 8B   (same key, lighter)
      3. Groq Gemma2 9B       (same key, Google model)
      4. Google Gemini Flash  (needs GEMINI_API_KEY in .env.txt)
      5. Hugging Face Mistral (needs HF_API_KEY in .env.txt)
    """
    errors = []

    # ── Groq models (all use the same API key) ────────────────────────────────
    for model in GROQ_MODELS:
        try:
            result = _try_groq(prompt, max_tokens, model)
            if result:
                print(f"✅ AI responded via Groq ({model})")
                return result
        except Exception as e:
            errors.append(f"Groq/{model}: {e}")
            print(f"⚠️  Groq/{model} failed: {e}")

    # ── Gemini fallback ────────────────────────────────────────────────────────
    try:
        result = _try_gemini(prompt, max_tokens)
        if result:
            print("✅ AI responded via Google Gemini")
            return result
    except Exception as e:
        errors.append(f"Gemini: {e}")
        print(f"⚠️  Gemini failed: {e}")

    # ── Hugging Face fallback ──────────────────────────────────────────────────
    try:
        result = _try_huggingface(prompt, max_tokens)
        if result:
            print("✅ AI responded via Hugging Face")
            return result
    except Exception as e:
        errors.append(f"HuggingFace: {e}")
        print(f"⚠️  HuggingFace failed: {e}")

    # ── All providers failed ───────────────────────────────────────────────────
    raise RuntimeError(
        "All AI providers are currently unavailable. Please try again in a moment. "
        f"Details: {' | '.join(errors)}"
    )


# Keep groq_ask as an alias so nothing else breaks
def groq_ask(prompt: str, max_tokens: int = 300) -> str:
    return ai_ask(prompt, max_tokens)

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()

def analyze_legal_document(text: str) -> dict:
    plain_prompt = f"""You are a legal expert who explains complex legal documents in simple language for common people in India.

Given the following legal document, explain it in very simple plain English that anyone can understand. Avoid legal jargon. Be clear and friendly.

Legal Document:
{text[:4000]}

Provide a clear plain English explanation in 3-5 paragraphs."""
    plain_english = groq_ask(plain_prompt, max_tokens=1000)

    clause_prompt = f"""You are a legal expert. Analyze the following legal document and break it down clause by clause.

For each clause or section you find:
- Give it a simple title
- Explain what it means in plain language
- Flag if it's IMPORTANT ⚠️ or RISKY 🔴 for the reader

Legal Document:
{text[:4000]}

Format exactly like:
📌 Clause 1: [Title]
Meaning: [Plain explanation]
Status: [Normal / ⚠️ Important / 🔴 Risky]

📌 Clause 2: [Title]
...and so on."""
    clauses = groq_ask(clause_prompt, max_tokens=1500)

    summary_prompt = f"""You are a legal expert. Read this legal document and extract the most important key points.

Legal Document:
{text[:4000]}

Provide:
✅ What you CAN do (rights)
❌ What you CANNOT do (restrictions)
⚠️ Important obligations or deadlines
🔴 Risky or unfair clauses to watch out for
💡 Overall recommendation

Be specific and practical for an Indian reader."""
    key_points = groq_ask(summary_prompt, max_tokens=1000)

    return {
        "plain_english": plain_english,
        "clauses":       clauses,
        "key_points":    key_points,
    }


def analyze_medical_report(text: str) -> dict:
    summary_prompt = f"""You are a friendly doctor explaining a medical report to a patient in India in simple plain English.
Read the following medical report and explain it so anyone can understand. No medical jargon.
Cover: what the report is about, overall health status, and what the patient should know.

Medical Report:
{text[:4000]}

Give a clear summary in 3-5 paragraphs."""
    summary = groq_ask(summary_prompt, max_tokens=1000)

    values_prompt = f"""You are a medical expert. Analyze each test value in the following medical report.

Medical Report:
{text[:4000]}

For each test value found, format exactly like:
🔬 [Test Name]
Value: [reported value]
Normal Range: [standard range]
Status: ✅ Normal / ⚠️ Borderline / 🔴 Abnormal
What it means: [plain explanation in 1-2 sentences]

List all values you find in the report."""
    value_analysis = groq_ask(values_prompt, max_tokens=1500)

    advice_prompt = f"""You are a friendly doctor giving practical health advice based on a medical report for an Indian patient.

Medical Report:
{text[:4000]}

Provide:
💊 Immediate actions needed (if any)
🥗 Diet recommendations
🏃 Lifestyle changes suggested
⚠️ Warning signs to watch for
👨‍⚕️ When to see a doctor
💡 General health tips

Be specific, practical, and encouraging."""
    health_advice = groq_ask(advice_prompt, max_tokens=1000)

    return {
        "summary":        summary,
        "value_analysis": value_analysis,
        "health_advice":  health_advice,
    }


# ── Voice command processing ───────────────────────────────────────────────────

WEBSITES = {
    "google":         "https://google.com",
    "youtube":        "https://youtube.com",
    "facebook":       "https://facebook.com",
    "instagram":      "https://instagram.com",
    "whatsapp":       "https://web.whatsapp.com",
    "gmail":          "https://mail.google.com",
    "amazon":         "https://amazon.in",
    "flipkart":       "https://flipkart.com",
    "spotify":        "https://open.spotify.com",
    "snapchat":       "https://snapchat.com",
    "times of india": "https://timesofindia.indiatimes.com",
    "weather":        "https://weather.com",
    "maps":           "https://maps.google.com",
    "twitter":        "https://twitter.com",
    "linkedin":       "https://linkedin.com",
    "reddit":         "https://reddit.com",
}


def process_voice_command(command: str) -> dict:
    cmd = command.lower().strip()
    response = {"text": "", "action": None, "url": None}

    for site, url in WEBSITES.items():
        if f"open {site}" in cmd or (site in cmd and "open" in cmd):
            response["text"]   = f"Opening {site.title()}."
            response["action"] = "open_url"
            response["url"]    = url
            return response

    if any(k in cmd for k in ["date", "time", "what time", "what day", "today"]):
        response["text"] = get_datetime()
        return response

    if any(k in cmd for k in ["battery", "charging"]):
        response["text"] = get_battery_status()
        return response

    if any(k in cmd for k in ["system", "computer", "device info", "about this"]):
        response["text"] = get_system_details()
        return response

    if any(k in cmd for k in ["ip address", "my ip"]):
        ip = get_public_ip()
        response["text"] = (f"Your public IP address is {ip}."
                            if ip else "Could not fetch your IP address.")
        return response

    if any(k in cmd for k in ["how are you", "what's up", "whats up", "hello", "hi triad"]):
        response["text"] = "I'm doing great! How can I help you today?"
        return response

    if "plan" in cmd and ("trip" in cmd or "travel" in cmd):
        response["text"]   = "Sure! Let me open the Travel Planner for you."
        response["action"] = "switch_tab"
        response["tab"]    = "travel"
        return response

    if any(k in cmd for k in ["legal", "contract", "document", "clause", "agreement"]):
        response["text"]   = "Sure! Let me open the Legal Explainer for you."
        response["action"] = "switch_tab"
        response["tab"]    = "legal"
        return response

    if any(k in cmd for k in ["medical", "report", "blood test", "health report", "prescription"]):
        response["text"]   = "Sure! Let me open the Medical Reader for you."
        response["action"] = "switch_tab"
        response["tab"]    = "medical"
        return response

    prompt = (
        f"You are Triad AI, a friendly voice assistant. "
        f"Answer in 1-2 short sentences suitable for text-to-speech. "
        f"No markdown, no bullet points, no symbols.\n\n"
        f"User said: {command}"
    )
    response["text"] = groq_ask(prompt, max_tokens=150)
    return response


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Triad AI backend is running ✅", "ai": "Groq (LLaMA 3.3 70B) — Free & Fast"}


@app.post("/voice/command")
def voice_command(req: VoiceCommandRequest):
    if not req.command.strip():
        raise HTTPException(status_code=400, detail="Command cannot be empty")
    try:
        return process_voice_command(req.command)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/travel/plan")
def travel_plan(req: TravelRequest):
    if not req.destination.strip() or not req.origin.strip():
        raise HTTPException(status_code=400, detail="Origin and destination are required")
    if req.days < 1 or req.days > 30:
        raise HTTPException(status_code=400, detail="Days must be between 1 and 30")
    try:
        prompt = f"""You are a friendly travel planner for Indian travellers.

Create a detailed {req.days}-day itinerary from {req.origin} to {req.destination}.
Travellers: {req.travelers} | Total budget: ₹{req.budget}

Format each day exactly like this:
Day X — [Theme]
• Morning: ...
• Afternoon: ...
• Evening: ...
• 🍽 Eat at: [restaurant name] — [cuisine type] (~₹ estimated cost)
• 📍 Must visit: [attraction name]
• 💸 Daily spend estimate: ₹...

After all days, add:
💰 Budget Breakdown
- Accommodation: ₹...
- Food: ₹...
- Transport: ₹...
- Activities: ₹...
- Total: ₹...

💡 3 Money-Saving Tips for {req.destination}

🚌 Best way to travel from {req.origin} to {req.destination} on this budget

Use real place names. Be specific and practical. Friendly, conversational tone."""
        itinerary = groq_ask(prompt, max_tokens=2000)
        return {
            "itinerary":   itinerary,
            "destination": req.destination,
            "origin":      req.origin,
            "days":        req.days,
            "travelers":   req.travelers,
            "budget":      req.budget
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/legal/explain-pdf")
async def legal_explain_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    try:
        pdf_bytes = await file.read()
        text = extract_text_from_pdf(pdf_bytes)
        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")
        result = analyze_legal_document(text)
        result["source"]   = "pdf"
        result["filename"] = file.filename
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/legal/explain-text")
def legal_explain_text(req: LegalTextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Legal text cannot be empty.")
    if len(req.text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Text too short.")
    try:
        result = analyze_legal_document(req.text)
        result["source"] = "text"
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/medical/analyze-pdf")
async def medical_analyze_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    try:
        pdf_bytes = await file.read()
        text = extract_text_from_pdf(pdf_bytes)
        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")
        result = analyze_medical_report(text)
        result["source"]   = "pdf"
        result["filename"] = file.filename
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/medical/analyze-text")
def medical_analyze_text(req: MedicalTextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Medical text cannot be empty.")
    if len(req.text.strip()) < 30:
        raise HTTPException(status_code=400, detail="Text too short.")
    try:
        result = analyze_medical_report(req.text)
        result["source"] = "text"
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/ask")
def ask_ai(req: AIQuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        answer = groq_ask(req.question, max_tokens=400)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ────────────────────────────────────────────────────────────────
def free_port(port: int):
    """Kill any process currently listening on the given port (Windows + Linux)."""
    import socket
    import sys

    # Quick check — is the port actually in use?
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        in_use = s.connect_ex(("127.0.0.1", port)) == 0

    if not in_use:
        return  # Port is free, nothing to do

    print(f"⚠️  Port {port} is already in use. Attempting to free it...")

    try:
        if sys.platform == "win32":
            import subprocess
            # Find PID using the port
            result = subprocess.run(
                f'netstat -ano | findstr ":{port} "',
                shell=True, capture_output=True, text=True
            )
            pids = set()
            for line in result.stdout.strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5 and "LISTENING" in line:
                    pids.add(parts[-1])
                elif len(parts) >= 4:
                    pids.add(parts[-1])
            for pid in pids:
                if pid.isdigit() and pid != "0":
                    subprocess.run(f"taskkill /PID {pid} /F", shell=True,
                                   capture_output=True)
                    print(f"✅ Killed process PID {pid} on port {port}")
        else:
            # Linux / macOS
            import subprocess
            result = subprocess.run(
                ["lsof", "-ti", f"tcp:{port}"],
                capture_output=True, text=True
            )
            for pid in result.stdout.strip().splitlines():
                if pid.isdigit():
                    subprocess.run(["kill", "-9", pid])
                    print(f"✅ Killed process PID {pid} on port {port}")
    except Exception as e:
        print(f"⚠️  Could not auto-free port {port}: {e}")
        print(f"    Please manually close the process using port {port} and try again.")

    import time
    time.sleep(1)  # Give OS time to release the port


if __name__ == "__main__":
    import uvicorn

    PORT = int(os.environ.get("PORT", 8000))
    free_port(PORT)

    print(f"\n🚀 Starting Triad AI backend on http://127.0.0.1:{PORT}")
    print("   Press Ctrl+C to stop.\n")

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=PORT,
        reload=False,
        timeout_keep_alive=300,
        workers=1,
        log_level="info"
    )
