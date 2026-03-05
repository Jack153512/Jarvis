# J.A.R.V.I.S

A local AI assistant with voice, chat, and optional tools (CAD, image gen, web automation). Runs fully offline using Ollama.

---

## What you need

- **Node.js** (v18 or newer) — [nodejs.org](https://nodejs.org)
- **Python 3.11** — [python.org](https://www.python.org/downloads/) (on Windows, the `py -3.11` launcher is used)
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
2. In a terminal: `ollama pull qwen2.5-coder:7b-instruct`

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

## Optional: extra features

- **Speech-to-text (STT)** — Uses faster-whisper (offline, multilingual); model auto-downloads on first use.
- **Text-to-speech (TTS)** — Uses Edge TTS (online); works after dependencies are installed.
- **CAD / 3D** — Needs `build123d`, `cadquery`; Shap-E 3D uses models in `backend/shap_e_model_cache/` (download on first use or add your own).
- **Image generation** — Uses local diffusers; first run downloads models (~1.7 GB). GPU (CUDA) recommended.
- **Web automation** — Run `playwright install` once after `pip install`.

Settings (identity, LLM, TTS, tools) are in `backend/settings.json`. That file is local-only and not in the repo.

---

## Scripts

| Command       | Description                    |
|--------------|--------------------------------|
| `npm run dev` | Start app (dev mode)          |
| `npm run build` | Build frontend for production |
| `npm start`  | Run Electron (after build)    |

---

## Troubleshooting

- **“Python backend failed to start”** — Activate the same `.venv` you used for `pip install`, or install Python 3.11 and run again (Windows uses `py -3.11`).
- **“Ollama / model not found”** — Install Ollama and run `ollama pull qwen2.5-coder:7b-instruct`.
- **Port 5173 in use** — Another app is using it; close it or change the port in the Vite config.
