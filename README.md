# 🌐 Triad AI

> An AI-powered multi-module web application for Voice, Medical, Legal, and Travel assistance.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Groq](https://img.shields.io/badge/Groq-LLaMA3.3--70B-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 About

**Triad AI** is a multi-functional AI-powered web application developed. It integrates four intelligent modules into a single unified interface, making AI accessible for everyday real-world use cases.

---

## ✨ Features

| Module | Description |
|---|---|
| 🎙️ Voice Assistant | Wake-word activated voice interaction using Web Speech API |
| 🏥 Medical Report Reader | Uploads and explains medical PDF reports in simple language |
| ⚖️ Legal Document Explainer | Simplifies complex legal documents into plain summaries |
| ✈️ Travel Planner | Generates AI-powered travel itineraries and recommendations |

---

## 🛠️ Tech Stack

- **Backend** — Python, FastAPI, Uvicorn
- **AI Model** — Groq LLaMA 3.3 70B
- **PDF Processing** — PyMuPDF
- **Frontend** — HTML, CSS, JavaScript (Single File)
- **Deployment** — Render.com

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Set up environment variables
Create a `.env` file in the root folder:
GROQ_API_KEY=your_groq_api_key_here
### 3. Run the application
```bash
uvicorn main:app --reload
```
### 4. Open in browser
http://localhost:8000

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key from console.groq.com |

---

## 📁 Project Structure
triad-ai/
│
├── main.py              # FastAPI backend
├── index.html           # Frontend (single file)
├── requirements.txt     # Python dependencies
├── .env                 # API keys (never push this)
├── .env.example         # Template for environment variables
└── README.md            # Project documentation

---

## 👨‍💻 Developer

**Vinay**
BCA Final Year — M S Ramaiah College, Bengaluru
AI/ML Intern @ GET SKILLED
---
## 📄 License
This project is for academic purposes.
