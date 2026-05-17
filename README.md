🤖 Triad AI Helper
A multi-feature AI web application built as a final-year BCA project at M S Ramaiah College of Arts, Science and Commerce, Bengaluru (Bengaluru City University).
Triad AI combines four intelligent tools into a single web app — Voice Assistant, Medical Report Reader, Legal Document Explainer, and Travel Planner — powered by Groq LLaMA 3.3 70B with an automatic multi-provider fallback chain, all completely free to run.

✨ Features
🎙️ Voice Assistant

Wake-word activation — say "Hey Triad" to start
Processes natural language voice commands
Opens websites, tells date/time, checks battery, fetches IP address
Automatically switches to relevant app tabs by voice
Text-to-speech responses in Indian English

🏥 Medical Report Reader

Upload a medical report PDF or paste report text
Get a plain-English summary of your health status
Detailed analysis of every test value with normal ranges
Practical health advice — diet, lifestyle, when to see a doctor
Read aloud support with stop/resume

⚖️ Legal Document Explainer

Upload a legal document PDF or paste legal text
Plain English explanation — no legal jargon
Clause-by-clause breakdown with risk flagging (⚠️ Important / 🔴 Risky)
Key rights, restrictions, obligations, and recommendations
Tailored for Indian legal documents and readers

✈️ Travel Planner

Enter origin, destination, days, travellers, and budget
Generates a day-by-day itinerary with morning/afternoon/evening plans
Real restaurant recommendations and must-visit attractions
Full budget breakdown (accommodation, food, transport, activities)
Money-saving tips and best transport options — Indian traveller focused


🛠️ Tech Stack
LayerTechnologyBackendPython 3.11, FastAPI, UvicornAI (Primary)Groq — LLaMA 3.3 70B VersatileAI (Fallback)Groq LLaMA 3.1 8B → Groq Gemma2 9B → Google Gemini 1.5 Flash → HuggingFace Mistral 7BPDF ProcessingPyMuPDF (fitz)Authbcrypt password hashing, session tokens, SQLiteDatabaseSQLite (users + sessions)FrontendVanilla HTML, CSS, JavaScriptVoiceWeb Speech API (SpeechRecognition + SpeechSynthesis)StylingDesert glassmorphism theme, custom animations

📁 Project Structure
Triad-AI-Helper/
├── backend.py          # FastAPI server — all API routes and AI logic
├── app.js              # Frontend JavaScript — all UI logic and voice engine
├── index.html          # HTML markup — all panels and screens
├── style.css           # All CSS — desert glassmorphism theme
├── init_db.py          # One-time database setup script
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template (copy to .env.txt)
├── .gitignore          # Git ignore rules
└── .dockerignore       # Docker ignore rules

🚀 Getting Started (Local Setup)
Prerequisites

Python 3.11 or 3.12
A free Groq API key — console.groq.com/keys

1. Clone the repository
bashgit clone https://github.com/vinay-2918/Triad-AI-Helper.git
cd Triad-AI-Helper
2. Install dependencies
bashpip install -r requirements.txt
3. Set up environment variables
bash# Copy the example file
cp .env.example .env.txt

# Open .env.txt and add your actual API keys
Your .env.txt should look like:
GROQ_API_KEY=your_groq_key_here
GEMINI_API_KEY=your_gemini_key_here        # optional
HF_API_KEY=your_huggingface_key_here       # optional
GOOGLE_PLACES_API_KEY=your_places_key_here # optional
4. Initialise the database
bashpython init_db.py
5. Start the server
bashpython backend.py
6. Open the app
Visit http://localhost:8000/ui in your browser.
Register an account, log in, and start using Triad AI.

🔑 Getting Free API Keys
KeyWhere to get itRequired?GROQ_API_KEYconsole.groq.com/keys✅ YesGEMINI_API_KEYaistudio.google.com/app/apikeyOptionalHF_API_KEYhuggingface.co/settings/tokensOptionalGOOGLE_PLACES_API_KEYconsole.cloud.google.comOptional
Only GROQ_API_KEY is required to run the app. The others act as fallback providers if Groq is unavailable.

🌐 Deployment (Render)
This project is configured for free deployment on Render.

Push this repo to GitHub
Go to render.com → New → Web Service
Connect your GitHub repo
Set Start Command: uvicorn backend:app --host 0.0.0.0 --port $PORT
Add environment variables in the Render dashboard:

GROQ_API_KEY
GOOGLE_PLACES_API_KEY


Click Deploy


Note: Render's free tier spins down after 15 minutes of inactivity. The first request after idle takes ~30–60 seconds to wake up.


👥 Team
NameUSNVinayU18EV23S0338ShashankU18EV23S0333ChinmayeeU18EV23S0321
Project Guide: Prof. Haripriya G S
Institution: M S Ramaiah College of Arts, Science and Commerce, Bengaluru
University: Bengaluru City University
Programme: Bachelor of Computer Applications (BCA) — Final Year, 2026

📄 License
This project is developed for academic purposes as part of the BCA final-year curriculum at M S Ramaiah College of Arts, Science and Commerce, Bengaluru.


Built with ❤️ by Team Triad — Bengaluru, 2026
