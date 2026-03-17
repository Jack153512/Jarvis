# J.A.R.V.I.S

A local AI assistant with voice, chat, and tools (CAD, web search, system info, file ops, web automation). Runs fully offline using Ollama.

---

## What you need

- **Node.js** (v18 or newer) — [nodejs.org](https://nodejs.org)
- **Python 3.11+** — [python.org](https://www.python.org/downloads/)
- **Ollama** — [ollama.ai](https://ollama.ai) (for the local LLM)

---

## Setup (first time)

### 1. Clone and open the project

```bash
git clone https://github.com/Jack153512/Jarvis.git
cd Jarvis
```

### 2. Backend (Python)

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Windows (CMD):
# .venv\Scripts\activate.bat
# Mac/Linux:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Install Ollama and pull the model:**

1. Install from [ollama.ai](https://ollama.ai/download).
2. In a terminal: `ollama pull qwen2.5:14b-instruct`

### 3. Frontend (Node)

```bash
npm install
```

---

## Run the app

```bash
npm run dev
```

This starts the Vite dev server, Electron window, and the Python backend. Use the app from the Electron window.

---

## Features

| Feature | Description |
|---------|-------------|
| **Chat** | Text input with streaming AI responses |
| **Web search** | Search the web (DuckDuckGo/ddgs, no API key) |
| **System info** | OS, CPU, RAM, GPU info on demand |
| **CAD / 3D** | Generate 3D models via Shap-E or build123d |
| **File ops** | Read, write, list files in the project |
| **Web automation** | Browser control via Playwright |
| **TTS** | Edge TTS for spoken responses |
| **Memory** | Personal and project memory injected into prompts |

The assistant uses a structured prompt (identity, capabilities, constraints, context), tool result feedback, and a sliding context window (last 40 messages).

---

## Optional: extra setup

- **Speech-to-text (STT)** — Uses faster-whisper (offline, multilingual); model auto-downloads on first use.
- **CAD / 3D** — Shap-E models in `backend/shap_e_model_cache/` (download on first use).
- **Image generation** — Local diffusers; first run downloads models (~1.7 GB). GPU (CUDA) recommended.
- **Web automation** — Run `playwright install` once after `pip install`.

Settings (identity, LLM, TTS, tool permissions) are in `backend/settings.json`. That file is local-only and not in the repo.

---

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start app (dev mode) |
| `npm run build` | Build frontend for production |
| `npm start` | Run Electron (after build) |
| `npm test` | Run Vitest (frontend tests) |
| `pytest` | Run backend tests |

---

## Tech stack

- **Frontend:** React 19, Vite 8, Electron 41, Three.js, Tailwind
- **Backend:** FastAPI, Socket.IO, Ollama (qwen2.5:14b-instruct)
- **Tools:** ddgs (web search), psutil (system info), build123d, Playwright

---

## Troubleshooting

- **"Python backend failed to start"** — Activate the same `.venv` you used for `pip install`, or ensure Python 3.11+ is available.
- **"Ollama / model not found"** — Install Ollama and run `ollama pull qwen2.5:14b-instruct`.
- **Port 5173 in use** — Another app is using it; close it or change the port in the Vite config.
- **Web search rate limit** — Results are cached; rapid repeated queries may wait briefly.

---

*Work in progress.*
