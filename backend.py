from fastapi import FastAPI, HTTPException, UploadFile, File, Form
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
import os

# ── Load secrets ───────────────────────────────────────────────────────────────
load_dotenv(".env.txt")
GROQ_API_KEY          = os.getenv("GROQ_API_KEY")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in .env.txt")

# ── Groq setup ─────────────────────────────────────────────────────────────────
client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.3-70b-versatile"   # ✅ Free, fast, very capable

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


# ── Request models ─────────────────────────────────────────────────────────────

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

class MedicalTextRequest(BaseModel):      # ✅ NEW
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

def groq_ask(prompt: str, max_tokens: int = 300) -> str:
    """Helper to call Groq API — free, no rate limit issues, sub-second speed."""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()

def analyze_legal_document(text: str) -> dict:
    """Send legal text to Groq and get full analysis."""

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


# ── ✅ NEW: Medical analysis function ──────────────────────────────────────────

def analyze_medical_report(text: str) -> dict:
    """Send medical report text to Groq and get full analysis."""

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

    # ✅ NEW: Medical voice shortcut
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


# ── Legal Explainer Routes ─────────────────────────────────────────────────────

@app.post("/legal/explain-pdf")
async def legal_explain_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    try:
        pdf_bytes = await file.read()
        text = extract_text_from_pdf(pdf_bytes)
        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF. Make sure it is not a scanned image.")
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
        raise HTTPException(status_code=400, detail="Text too short. Please paste the full legal document.")
    try:
        result = analyze_legal_document(req.text)
        result["source"] = "text"
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──  NEW: Medical Reader Routes 

@app.post("/medical/analyze-pdf")
async def medical_analyze_pdf(file: UploadFile = File(...)):
    """Upload a medical report PDF and get full analysis."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    try:
        pdf_bytes = await file.read()
        text = extract_text_from_pdf(pdf_bytes)
        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF. Make sure it is not a scanned image.")
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
    """Paste medical report text directly and get full analysis."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Medical text cannot be empty.")
    if len(req.text.strip()) < 30:
        raise HTTPException(status_code=400, detail="Text too short. Please paste actual report values.")
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


# ── Entry point 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        timeout_keep_alive=120,
        workers=1
    )