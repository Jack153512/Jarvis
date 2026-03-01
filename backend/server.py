"""
J.A.R.V.I.S Backend Server - Local LLM Version

FastAPI + Socket.IO server for the J.A.R.V.I.S desktop application.
Fully offline, no paid APIs required.

Uses:
- Ollama with Qwen2.5-Coder-7B for AI
- edge-tts for text-to-speech
- Vosk for speech-to-text (optional)
"""

import sys
import asyncio
import logging
from contextlib import asynccontextmanager
import os
import copy
import time


# ── Electron pipe safety (stderr only) ───────────────────────────────────────
# When Python is spawned as an Electron child process, sys.stderr is a Windows
# named pipe that raises "OSError: [Errno 22] Invalid argument" when tqdm calls
# stderr.flush() inside its progress-bar __init__.
# We ONLY wrap stderr — NOT stdout — because wrapping stdout would cause Python
# to buffer all print() output (isatty() → False → block buffering), which
# makes the backend appear silent in the Electron terminal.
class _SafeStderr:
    """Transparent stderr wrapper that silences OSError on flush."""
    def __init__(self, stream):
        self._s = stream

    def write(self, data):
        try:
            return self._s.write(data)
        except OSError:
            return 0

    def flush(self):
        try:
            self._s.flush()
        except OSError:
            pass

    def isatty(self):
        return False  # tells tqdm not to use ANSI cursor codes

    def __getattr__(self, name):
        return getattr(self._s, name)

sys.stderr = _SafeStderr(sys.stderr)
# ─────────────────────────────────────────────────────────────────────────────

# Suppress TensorFlow's hardware-scanning startup noise.
# TF initialises (JIT, oneDNN, CUDA scan) the moment numpy is first touched,
# which adds 8-18 s to any import that pulls numpy. These two variables cut
# that init cost significantly and silence the log spam entirely.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")       # errors only
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")       # skip oneDNN JIT


def _ensure_utf8_stdio():
    """Ensure stdout/stderr can print Unicode on Windows consoles.

    Some Windows terminals default to cp1252/cp437, which can raise
    UnicodeEncodeError when printing Vietnamese or other non-ASCII text.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
            continue
        except Exception:
            pass

        try:
            import io
            buf = getattr(stream, "buffer", None)
            if buf is None:
                continue
            wrapped = io.TextIOWrapper(buf, encoding="utf-8", errors="replace", line_buffering=True)
            setattr(sys, stream_name, wrapped)
        except Exception:
            pass


_ensure_utf8_stdio()

# Fix for asyncio subprocess support on Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import socketio
import uvicorn
from fastapi import FastAPI
import json
import signal
from pathlib import Path
import tempfile
import shutil
from typing import Optional

from memory import JarvisMemory

# Ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Lazy import jarvis - only import when needed to speed up startup
# import jarvis

# Create servers
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    logger=False,
    engineio_logger=False,
)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("jarvis.backend")


def _fire_and_forget(coro):
    try:
        task = asyncio.create_task(coro)
    except Exception:
        return

    def _done(t: asyncio.Task):
        try:
            t.result()
        except Exception:
            pass

    try:
        task.add_done_callback(_done)
    except Exception:
        pass


def _emit(event: str, data, room=None):
    if room is None:
        _fire_and_forget(sio.emit(event, data))
    else:
        _fire_and_forget(sio.emit(event, data, room=room))


def _make_safe_cad_work_dir() -> str:
    return tempfile.mkdtemp(prefix="jarvis_cad_work_")


def _copy_cad_artifacts_to_target(result: dict, target_dir: str) -> dict:
    try:
        if not result or not isinstance(result, dict):
            return result
        file_path = result.get('file_path')
        if not file_path or not os.path.exists(file_path):
            return result

        os.makedirs(target_dir, exist_ok=True)
        src = Path(file_path)
        dest = Path(target_dir) / src.name
        shutil.copy2(src, dest)

        src_obj = src.with_suffix('.obj')
        if src_obj.exists():
            try:
                shutil.copy2(src_obj, dest.with_suffix('.obj'))
            except Exception:
                pass

        result['file_path'] = str(dest)
        return result
    except OSError as e:
        if getattr(e, 'errno', None) == 22:
            fallback_dir = Path(tempfile.gettempdir()) / 'jarvis_cad'
            try:
                fallback_dir.mkdir(parents=True, exist_ok=True)
                src = Path(result.get('file_path'))
                fallback_path = fallback_dir / src.name
                shutil.copy2(src, fallback_path)
                result['file_path'] = str(fallback_path)
                return result
            except Exception:
                return result
        return result
    except Exception:
        return result


async def _startup():
    global standalone_cad_agent, standalone_web_agent
    logger.info("========================================")
    logger.info("Server startup event - /status ready")
    logger.info("Starting Jarvis Backend...")
    logger.info("Python: %s", sys.version)
    logger.info("========================================")

    init_agents_on_startup = os.environ.get('INIT_AGENTS_ON_STARTUP') == '1'

    try:
        _trigger_llm_warmup()
    except Exception:
        pass

    if init_agents_on_startup:
        # Initialize standalone CAD agent for direct generation
        try:
            from cad_agent_shape import ShapECadAgent

            def on_cad_status(status):
                asyncio.create_task(sio.emit('cad_status', status))

            def on_cad_thought(thought):
                asyncio.create_task(sio.emit('cad_thought', {'text': thought}))

            standalone_cad_agent = ShapECadAgent(
                on_thought=on_cad_thought,
                on_status=on_cad_status
            )
            logger.info("Shap-E CAD Agent initialized")

            # Preload models in background to avoid first-request delay
            async def preload_models():
                logger.info("Preloading Shap-E neural models (this may take a moment)...")
                success = await standalone_cad_agent._load_models()
                if success:
                    logger.info("Shap-E models preloaded and ready!")
                else:
                    logger.warning("Shap-E models failed to preload")

            if os.environ.get('DISABLE_SHAPE_PRELOAD') != '1':
                asyncio.create_task(preload_models())

        except Exception as e:
            logger.exception("Failed to initialize CAD agent: %s", e)
            standalone_cad_agent = None

        # Initialize standalone Web agent
        try:
            from web_agent import WebAgent
            standalone_web_agent = WebAgent()
            logger.info("Web Agent initialized (browser-use + Ollama)")
        except Exception as e:
            logger.exception("Failed to initialize Web agent: %s", e)
            standalone_web_agent = None
    else:
        logger.info("Lazy init enabled: CAD/Web agents will initialize on first use")


async def _shutdown():
    global audio_loop, loop_task
    logger.info("Server shutdown...")

    if audio_loop:
        try:
            audio_loop.stop()
        except Exception:
            logger.exception("Error while stopping audio_loop")
        finally:
            audio_loop = None

    if loop_task and not loop_task.done():
        loop_task.cancel()
    loop_task = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await _startup()
    try:
        yield
    finally:
        await _shutdown()


app = FastAPI(lifespan=lifespan)
app_socketio = socketio.ASGIApp(sio, app)

# Shutdown handler
def signal_handler(sig, frame):
    logger.info("Caught signal %s. Exiting...", sig)
    if audio_loop:
        try:
            audio_loop.stop() 
        except:
            pass
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Global state
audio_loop = None
loop_task = None
standalone_cad_agent = None  # For CAD generation without audio loop
standalone_web_agent = None  # For web browsing without audio loop
_llm_warmup_task = None

MEMORY = JarvisMemory(os.environ.get("JARVIS_MEMORY_DB_PATH") or "jarvis_memory.db")

# Active conversation tracking (server-side)
_active_conv_id: Optional[str] = None

# Settings
SETTINGS_FILE = str((Path(__file__).resolve().parent / "settings.json").resolve())
DEFAULT_SETTINGS = {
    "identity": {
        "assistant_name": "JARVIS",
        "user_name": "User",
    },
    "personality": {
        "enabled": False,
        "auto_project_naming": False,
        "humor_in_logs": False,
        "delight_moments": False,
        "humor_rate": 0.06,
        "delight_rate": 0.02,
        "min_delight_interval_s": 600,
    },
    "llm": {
        "provider": "ollama",
        "base_url": "http://127.0.0.1:11434",
        "model": "qwen2.5-coder:7b-instruct",
        "code_model": "qwen2.5-coder:7b-instruct"
    },
    "tts": {
        "enabled": True,
        "voice": "en-US-GuyNeural",
        "voice_en": "en-US-GuyNeural",
        "voice_vi": "vi-VN-HoaiMyNeural",
        "auto_detect": True,
        "rate_en": "+0%",
        "pitch_en": "+0Hz",
        "rate_vi": "+0%",
        "pitch_vi": "+0Hz"
    },
    "stt": {
        "provider": "browser",  # "browser" or "vosk"
        "model_path": "./models/vosk-model-small-en-us"
    },
    "tool_permissions": {
        "generate_cad": True,
        "iterate_cad": False,
        "run_web_agent": True,
        "write_file": False,
        "read_file": True,
        "read_directory": True,
        "create_project": True,
        "switch_project": True,
        "list_projects": True
    }
}

SETTINGS = copy.deepcopy(DEFAULT_SETTINGS)

SETTINGS_LOCK = asyncio.Lock()


def load_settings():
    """Load settings from file."""
    global SETTINGS
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                # Deep merge
                for key, value in loaded.items():
                    if isinstance(value, dict) and key in SETTINGS:
                        SETTINGS[key].update(value)
                    else:
                        SETTINGS[key] = value
            logger.info("Loaded settings")
        except Exception as e:
            logger.exception("Error loading settings: %s", e)


def save_settings():
    """Save settings to file."""
    try:
        tmp_path = f"{SETTINGS_FILE}.tmp"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(SETTINGS, f, indent=4, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, SETTINGS_FILE)
        logger.info("Settings saved")
    except Exception as e:
        logger.exception("Error saving settings: %s", e)


async def _persist_identity_update(identity_patch: dict):
    """Persist identity changes (e.g., learned user_name) to settings.json.

    This is invoked from the AudioLoop via callback, so it must never raise.
    """
    try:
        if not isinstance(identity_patch, dict):
            return

        async with SETTINGS_LOCK:
            ident = SETTINGS.get("identity")
            if not isinstance(ident, dict):
                ident = {}
                SETTINGS["identity"] = ident

            if "user_name" in identity_patch:
                ident["user_name"] = identity_patch.get("user_name")
            if "assistant_name" in identity_patch:
                ident["assistant_name"] = identity_patch.get("assistant_name")

            _sanitize_identity_settings()
            snapshot = copy.deepcopy(SETTINGS)

            try:
                ident2 = snapshot.get("identity") if isinstance(snapshot, dict) else None
                if isinstance(ident2, dict):
                    u = ident2.get("user_name")
                    a = ident2.get("assistant_name")
                    if isinstance(u, str) and u.strip() and u.strip() != DEFAULT_SETTINGS["identity"]["user_name"]:
                        MEMORY.set_user_name(u.strip())
                    if isinstance(a, str) and a.strip() and a.strip() != DEFAULT_SETTINGS["identity"]["assistant_name"]:
                        MEMORY.set_assistant_name(a.strip())
            except Exception:
                pass

        try:
            await asyncio.to_thread(save_settings_snapshot, snapshot)
        except Exception:
            logger.exception("Failed to save identity update")

        try:
            _emit('settings', snapshot)
        except Exception:
            pass
    except Exception:
        logger.exception("_persist_identity_update failed")


def _sanitize_identity_settings() -> bool:
    """Ensure assistant/user identity are distinct and valid.

    Returns True if SETTINGS was mutated.
    """
    try:
        mutated = False
        ident = SETTINGS.get("identity")
        if not isinstance(ident, dict):
            ident = {}
            mutated = True

        assistant_name = str(ident.get("assistant_name") or DEFAULT_SETTINGS["identity"]["assistant_name"]).strip()
        user_name = str(ident.get("user_name") or DEFAULT_SETTINGS["identity"]["user_name"]).strip()

        if not assistant_name:
            assistant_name = DEFAULT_SETTINGS["identity"]["assistant_name"]
            mutated = True

        if not user_name:
            user_name = DEFAULT_SETTINGS["identity"]["user_name"]
            mutated = True

        low_user = user_name.lower()
        low_assistant = assistant_name.lower()

        if low_user in {"jarvis", "j.a.r.v.i.s", "j.a.r.v.i.s."}:
            user_name = DEFAULT_SETTINGS["identity"]["user_name"]
            mutated = True

        if low_user == low_assistant:
            user_name = DEFAULT_SETTINGS["identity"]["user_name"]
            mutated = True

        if ident.get("assistant_name") != assistant_name:
            ident["assistant_name"] = assistant_name
            mutated = True

        if ident.get("user_name") != user_name:
            ident["user_name"] = user_name
            mutated = True

        if SETTINGS.get("identity") is not ident:
            SETTINGS["identity"] = ident
            mutated = True

        return mutated
    except Exception:
        return False


def save_settings_snapshot(snapshot):
    """Save an explicit settings snapshot to file.

    This avoids races where SETTINGS mutates between releasing SETTINGS_LOCK
    and writing to disk.
    """
    try:
        tmp_path = f"{SETTINGS_FILE}.tmp"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=4, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, SETTINGS_FILE)
        logger.info("Settings saved")
    except Exception as e:
        logger.exception("Error saving settings: %s", e)


# Load settings on startup
load_settings()

try:
    stored_user = MEMORY.get_user_name()
    stored_assistant = MEMORY.get_assistant_name()
    if stored_user or stored_assistant:
        ident = SETTINGS.get("identity")
        if not isinstance(ident, dict):
            ident = {}
            SETTINGS["identity"] = ident
        if stored_user:
            ident["user_name"] = stored_user
        if stored_assistant:
            ident["assistant_name"] = stored_assistant
except Exception:
    pass

try:
    if _sanitize_identity_settings():
        save_settings()
except Exception:
    pass

try:
    llm_base_url = SETTINGS.get("llm", {}).get("base_url")
    if isinstance(llm_base_url, str) and "localhost" in llm_base_url:
        SETTINGS["llm"]["base_url"] = llm_base_url.replace("localhost", "127.0.0.1")
except Exception:
    pass

logger.info("Server module loaded - /status endpoint available")


def _trigger_llm_warmup():
    global _llm_warmup_task
    try:
        if _llm_warmup_task and (not _llm_warmup_task.done()):
            return
    except Exception:
        pass

    async def _do():
        try:
            from local_llm import LocalLLM, LLMConfig
            llm_cfg = SETTINGS.get("llm", {}) if isinstance(SETTINGS, dict) else {}
            base_url = llm_cfg.get("base_url") if isinstance(llm_cfg, dict) else None
            model = llm_cfg.get("model") if isinstance(llm_cfg, dict) else None

            cfg = LLMConfig(
                base_url=base_url or "http://127.0.0.1:11434",
                model=model or "qwen2.5-coder:7b-instruct",
                temperature=0.0,
                context_length=2048,
                timeout=30.0,
            )
            llm = LocalLLM(cfg)
            try:
                await llm.warmup()
            finally:
                try:
                    await llm.close()
                except Exception:
                    pass
        except Exception:
            return

    try:
        _llm_warmup_task = asyncio.create_task(_do())
        _llm_warmup_task.add_done_callback(lambda t: (t.exception() if not t.cancelled() else None))
    except Exception:
        _llm_warmup_task = None


@app.get("/status")
async def status():
    """Health check endpoint."""
    return {"status": "running", "service": "J.A.R.V.I.S Backend (Local)"}


@sio.event
async def connect(sid, environ):
    """Handle client connection."""
    logger.info("Client connected: %s", sid)
    try:
        await sio.emit('status', {'msg': 'Connected to J.A.R.V.I.S Backend'}, room=sid)
    except Exception:
        pass

    try:
        _trigger_llm_warmup()
    except Exception:
        pass


@sio.event
async def disconnect(sid, data=None):
    """Handle client disconnection."""
    logger.info("Client disconnected: %s", sid)


@sio.event
async def start_audio(sid, data=None):
    """Start the J.A.R.V.I.S audio loop."""
    global audio_loop, loop_task
    
    # Lazy import jarvis - only import when needed
    import jarvis
    
    logger.info("Starting Audio Loop...")
    
    device_index = None
    device_name = None
    if data:
        device_index = data.get('device_index')
        device_name = data.get('device_name')
        
    logger.info("Input device: Name=%r, Index=%r", device_name, device_index)
    
    if audio_loop:
        if loop_task and (loop_task.done() or loop_task.cancelled()):
            audio_loop = None
            loop_task = None
        else:
            await sio.emit('status', {'msg': 'J.A.R.V.I.S Already Running'})
            return

    # Callbacks
    def on_audio_data(data):
        # Handle TTS streaming events and visualization data
        if isinstance(data, dict):
            event_type = data.get('type', '')
            
            if event_type == 'tts_chunk':
                # Streaming TTS chunk - send immediately for playback
                _emit('tts_chunk', data)
            elif event_type == 'tts_end':
                # End of TTS stream
                _emit('tts_end', data)
            elif event_type == 'tts_stop':
                # TTS interrupted
                _emit('tts_stop', data)
            elif event_type == 'tts':
                # Legacy single audio (backwards compatibility)
                _emit('tts_audio', data)
            else:
                # Unknown dict type - log and ignore
                logger.warning("Unknown audio event type: %s", event_type)
        else:
            # Visualization data (raw audio samples)
            _emit('audio_data', {'data': list(data) if hasattr(data, '__iter__') else []})
    
    def on_cad_data(data):
        print(f"[SERVER] CAD data: {len(data.get('data', ''))} bytes")
        _emit('cad_data', data)

    def on_web_data(data):
        _emit('browser_frame', data)
        
    def on_transcription(data):
        _emit('transcription', data)

    def on_tool_confirmation(data):
        print(f"[SERVER] Tool confirmation: {data.get('tool')}")
        _emit('tool_confirmation_request', data)

    def on_tool_activity(data):
        _emit('tool_activity', data)

    def on_cad_status(status):
        if isinstance(status, dict):
            _emit('cad_status', status)
        else:
            _emit('cad_status', {'status': status})

    def on_cad_thought(thought_text):
        _emit('cad_thought', {'text': thought_text})

    def on_cad_spec(spec):
        # Emit design specification for frontend visibility (Two-Stage CAD)
        print(f"[SERVER] CAD Spec: {len(spec.get('components', []))} components")
        _emit('cad_spec', spec)
    
    def on_project_update(project_name):
        print(f"[SERVER] Project: {project_name}")
        _emit('project_update', {'project': project_name})

    def on_recommendation(data):
        _emit('recommendation', data)

    def on_identity_update(patch):
        try:
            asyncio.create_task(_persist_identity_update(patch))
        except Exception:
            pass

    def on_error(msg):
        print(f"[SERVER] Error: {msg}")
        _emit('error', {'msg': msg})

    try:
        tts_cfg = SETTINGS.get("tts", {}) or {}
        tts_voice = tts_cfg.get("voice", "andrew")
        tts_voice_en = tts_cfg.get("voice_en") or tts_voice
        tts_voice_vi = tts_cfg.get("voice_vi")
        tts_auto_detect = tts_cfg.get("auto_detect", True)
        tts_rate_en = tts_cfg.get("rate_en", "+0%")
        tts_pitch_en = tts_cfg.get("pitch_en", "+0Hz")
        tts_rate_vi = tts_cfg.get("rate_vi", "+0%")
        tts_pitch_vi = tts_cfg.get("pitch_vi", "+0Hz")

        identity_cfg = SETTINGS.get("identity", {}) or {}
        user_name = identity_cfg.get("user_name") if isinstance(identity_cfg, dict) else None
        assistant_name = identity_cfg.get("assistant_name") if isinstance(identity_cfg, dict) else None
        personality_cfg = SETTINGS.get("personality", {}) if isinstance(SETTINGS, dict) else {}
        audio_loop = jarvis.AudioLoop(
            video_mode="none", 
            on_audio_data=on_audio_data,
            on_cad_data=on_cad_data,
            on_web_data=on_web_data,
            on_transcription=on_transcription,
            on_recommendation=on_recommendation,
            on_tool_confirmation=on_tool_confirmation,
            on_tool_activity=on_tool_activity,
            on_cad_status=on_cad_status,
            on_cad_thought=on_cad_thought,
            on_cad_spec=on_cad_spec,  # Two-Stage CAD design specification
            on_project_update=on_project_update,
            on_identity_update=on_identity_update,
            on_error=on_error,
            input_device_index=device_index,
            input_device_name=device_name,
            user_name=user_name,
            assistant_name=assistant_name,
            personality=personality_cfg,
            tts_voice=tts_voice,
            tts_voice_en=tts_voice_en,
            tts_voice_vi=tts_voice_vi,
            tts_rate_en=tts_rate_en,
            tts_pitch_en=tts_pitch_en,
            tts_rate_vi=tts_rate_vi,
            tts_pitch_vi=tts_pitch_vi,
            tts_auto_detect=tts_auto_detect
        )

        # Apply permissions
        audio_loop.update_permissions(SETTINGS["tool_permissions"])
        
        # Check initial mute state
        if data and data.get('muted', False):
            audio_loop.set_paused(True)

        # Start the loop
        loop_task = asyncio.create_task(audio_loop.run())
        
        def handle_loop_exit(task):
            try:
                task.result()
            except asyncio.CancelledError:
                print("[SERVER] Audio Loop Cancelled")
            except Exception as e:
                print(f"[SERVER] Audio Loop Error: {e}")
        
        loop_task.add_done_callback(handle_loop_exit)
        
        await sio.emit('status', {'msg': 'J.A.R.V.I.S Started'})
        
    except Exception as e:
        print(f"[SERVER] CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        await sio.emit('error', {'msg': f"Failed to start: {str(e)}"})
        audio_loop = None


@sio.event
async def stop_audio(sid, data=None):
    """Stop the audio loop."""
    global audio_loop
    try:
        if audio_loop:
            audio_loop.stop()
            audio_loop = None
        await sio.emit('status', {'msg': 'J.A.R.V.I.S Stopped'}, room=sid)
    except Exception:
        logger.exception("stop_audio failed")


@sio.event
async def pause_audio(sid, data=None):
    """Pause audio input."""
    if audio_loop:
        audio_loop.set_paused(True)
        await sio.emit('status', {'msg': 'Audio Paused'})


@sio.event
async def resume_audio(sid, data=None):
    """Resume audio input."""
    if audio_loop:
        audio_loop.set_paused(False)
        await sio.emit('status', {'msg': 'Audio Resumed'})


@sio.event
async def stop_tts(sid, data=None):
    """Stop TTS playback immediately."""
    try:
        if audio_loop:
            print("[SERVER] Stopping TTS...")
            await audio_loop.stop_speaking()
        await sio.emit('tts_stop', {}, room=sid)
    except Exception:
        logger.exception("stop_tts failed")


@sio.event
async def confirm_tool(sid, data):
    """Handle tool confirmation response."""
    request_id = data.get('id')
    confirmed = data.get('confirmed', False)
    
    print(f"[SERVER] Tool confirmation: {request_id} = {confirmed}")
    
    if audio_loop:
        audio_loop.resolve_tool_confirmation(request_id, confirmed)


@sio.event
async def shutdown(sid, data=None):
    """Graceful shutdown."""
    global audio_loop, loop_task
    
    print("[SERVER] Shutdown requested...")
    
    if audio_loop:
        audio_loop.stop()
        audio_loop = None
    
    if loop_task and not loop_task.done():
        loop_task.cancel()
        loop_task = None
    
    os._exit(0)


@sio.event
async def user_input(sid, data):
    """Handle text input from user."""
    text = data.get('text')
    print(f"[SERVER] User input: {text}")
    
    if not audio_loop:
        await sio.emit('error', {'msg': 'J.A.R.V.I.S not running'})
        return

    if text:
        # Process through the audio loop
        await audio_loop.process_text_input(text)


@sio.event
async def video_frame(sid, data):
    """Handle video frame from frontend."""
    try:
        image_data = (data or {}).get('image') if isinstance(data, dict) else None
        if image_data and audio_loop:
            _fire_and_forget(audio_loop.send_frame(image_data))
    except Exception:
        logger.exception("video_frame failed")


@sio.event
async def iterate_cad(sid, data):
    """Handle CAD iteration request."""
    global standalone_cad_agent
    prompt = data.get('prompt')
    print(f"[SERVER] CAD iterate: {prompt}")
    
    # Use audio_loop's CAD agent if available, otherwise use standalone
    cad_agent = None
    output_dir = None
    target_dir = None
    
    if audio_loop and audio_loop.cad_agent:
        cad_agent = audio_loop.cad_agent
        target_dir = str(audio_loop.project_manager.get_current_project_path() / "cad")
    elif standalone_cad_agent:
        cad_agent = standalone_cad_agent
        target_dir = str(Path("projects/temp/cad").absolute())
        os.makedirs(target_dir, exist_ok=True)
    else:
        try:
            from cad_agent_shape import ShapECadAgent

            def on_cad_status(status):
                asyncio.create_task(sio.emit('cad_status', status))

            def on_cad_thought(thought):
                asyncio.create_task(sio.emit('cad_thought', {'text': thought}))

            standalone_cad_agent = ShapECadAgent(
                on_thought=on_cad_thought,
                on_status=on_cad_status
            )
            cad_agent = standalone_cad_agent
            target_dir = str(Path("projects/temp/cad").absolute())
            os.makedirs(target_dir, exist_ok=True)

            if os.environ.get('DISABLE_SHAPE_PRELOAD') != '1':
                asyncio.create_task(standalone_cad_agent._load_models())
        except Exception as e:
            print(f"[SERVER] Failed to initialize CAD agent: {e}")
            await sio.emit('error', {'msg': f'Failed to initialize CAD agent: {e}'})

    if not cad_agent:
        await sio.emit('error', {'msg': 'CAD Agent not available'})
        return

    output_dir = _make_safe_cad_work_dir()

    try:
        await sio.emit('cad_status', {'status': 'generating'})
        
        result = await cad_agent.iterate_prototype(prompt, output_dir=output_dir)

        if (not audio_loop) and target_dir:
            result = _copy_cad_artifacts_to_target(result, target_dir)
        
        if result:
            await sio.emit('cad_data', result)
            if audio_loop and 'file_path' in result:
                audio_loop.project_manager.save_cad_artifact(result['file_path'], prompt)
            await sio.emit('status', {'msg': 'Design updated'})
        else:
            last_error = getattr(cad_agent, 'last_error', None)
            if last_error:
                await sio.emit('error', {'msg': f'Failed to update design: {last_error}'})
            else:
                await sio.emit('error', {'msg': 'Failed to update design'})
            
    except Exception as e:
        print(f"[SERVER] CAD iterate error: {e}")
        await sio.emit('error', {'msg': f"Error: {str(e)}"})


@sio.event
async def generate_cad(sid, data):
    """Handle CAD generation request."""
    global standalone_cad_agent
    prompt = data.get('prompt')
    print(f"[SERVER] CAD generate: {prompt}")
    
    # Use audio_loop's CAD agent if available, otherwise use standalone
    cad_agent = None
    output_dir = None
    target_dir = None
    
    if audio_loop and audio_loop.cad_agent:
        cad_agent = audio_loop.cad_agent
        target_dir = str(audio_loop.project_manager.get_current_project_path() / "cad")
    elif standalone_cad_agent:
        cad_agent = standalone_cad_agent
        target_dir = str(Path("projects/temp/cad").absolute())
        os.makedirs(target_dir, exist_ok=True)
    else:
        try:
            from cad_agent_shape import ShapECadAgent

            def on_cad_status(status):
                asyncio.create_task(sio.emit('cad_status', status))

            def on_cad_thought(thought):
                asyncio.create_task(sio.emit('cad_thought', {'text': thought}))

            standalone_cad_agent = ShapECadAgent(
                on_thought=on_cad_thought,
                on_status=on_cad_status
            )
            cad_agent = standalone_cad_agent
            target_dir = str(Path("projects/temp/cad").absolute())
            os.makedirs(target_dir, exist_ok=True)

            if os.environ.get('DISABLE_SHAPE_PRELOAD') != '1':
                asyncio.create_task(standalone_cad_agent._load_models())
        except Exception as e:
            print(f"[SERVER] Failed to initialize CAD agent: {e}")
            await sio.emit('error', {'msg': f'Failed to initialize CAD agent: {e}'})

    if not cad_agent:
        await sio.emit('error', {'msg': 'CAD Agent not available'})
        return

    output_dir = _make_safe_cad_work_dir()

    try:
        await sio.emit('cad_status', {'status': 'generating'})
        
        result = await cad_agent.generate_prototype(prompt, output_dir=output_dir)

        if audio_loop and result and 'file_path' in result:
            saved_path = audio_loop.project_manager.save_cad_artifact(result['file_path'], prompt)
            if saved_path:
                result['file_path'] = saved_path
        elif target_dir:
            result = _copy_cad_artifacts_to_target(result, target_dir)
        
        if result:
            await sio.emit('cad_data', result)
            await sio.emit('status', {'msg': 'Design generated'})
        else:
            last_error = getattr(cad_agent, 'last_error', None)
            if last_error:
                await sio.emit('error', {'msg': f'Failed to generate design: {last_error}'})
            else:
                await sio.emit('error', {'msg': 'Failed to generate design'})
            
    except Exception as e:
        print(f"[SERVER] CAD generate error: {e}")
        await sio.emit('error', {'msg': f"Error: {str(e)}"})


_image_generation_lock = asyncio.Lock()
# Hard cap: if model load + inference takes longer than this, the backend sends
# a timeout error instead of hanging forever.
_IMAGE_GEN_TIMEOUT = 420  # seconds (7 min covers first-time model download)

@sio.event
async def generate_image(sid, data):
    """Generate an image using local diffusers (no external server needed)."""
    import traceback

    # Reject duplicate requests that arrive while a generation is already running.
    # Without this guard, every "Try Again" click spawns a new concurrent download
    # attempt, all stalling each other at 0 % indefinitely.
    if _image_generation_lock.locked():
        await sio.emit('image_result', {
            'status': 'error',
            'message': 'A generation is already in progress. Please wait.',
        }, to=sid)
        return

    prompt = data.get('prompt', '')
    negative = data.get('negative_prompt', '')
    size = data.get('size', '512x512')
    steps = int(data.get('steps', 12))
    seed = data.get('seed', None)

    print(f"[SERVER] Image generate: {prompt!r} size={size} steps={steps}", flush=True)
    await sio.emit('image_status', {'status': 'generating'}, to=sid)

    async with _image_generation_lock:
        try:
            # Import is inside try/except so any ImportError is surfaced to the
            # client instead of silently killing the event handler.
            from image_gen import generate_image as _generate_image

            result = await asyncio.wait_for(
                _generate_image(
                    prompt=prompt,
                    negative_prompt=negative,
                    size=size,
                    steps=steps,
                    seed=seed,
                ),
                timeout=_IMAGE_GEN_TIMEOUT,
            )
            await sio.emit('image_result', {**result, 'status': 'done'}, to=sid)
            await sio.emit('status', {'msg': 'Image generated'}, to=sid)
        except asyncio.TimeoutError:
            msg = f"Image generation timed out after {_IMAGE_GEN_TIMEOUT}s. The model may still be downloading — try again in a minute."
            print(f"[SERVER] Image generate timeout", flush=True)
            await sio.emit('image_result', {'status': 'error', 'message': msg}, to=sid)
        except Exception as e:
            tb = traceback.format_exc()
            short = str(e) or type(e).__name__
            print(f"[SERVER] Image generate error:\n{tb}", flush=True)
            await sio.emit('image_result', {'status': 'error', 'message': short, 'traceback': tb}, to=sid)


@sio.event
async def prompt_web_agent(sid, data):
    """Handle web agent request."""
    global standalone_web_agent
    prompt = data.get('prompt')
    print(f"[SERVER] Web agent: {prompt}")
    
    # Use audio_loop's web agent if available, otherwise use standalone
    web_agent = None
    if audio_loop and audio_loop.web_agent:
        web_agent = audio_loop.web_agent
    elif standalone_web_agent:
        web_agent = standalone_web_agent
    else:
        try:
            from web_agent import WebAgent
            standalone_web_agent = WebAgent()
            web_agent = standalone_web_agent
        except Exception as e:
            print(f"[SERVER] Failed to initialize Web agent: {e}")

    if not web_agent:
        await sio.emit('error', {'msg': 'Web Agent not available'})
        return

    try:
        await sio.emit('status', {'msg': 'Web Agent running...'})
        await sio.emit('web_agent_status', {'status': 'running', 'task': prompt})
        
        async def update_callback(image_b64, log):
            await sio.emit('browser_frame', {'image': image_b64, 'log': log})
        
        result = await web_agent.run_task(prompt, update_callback=update_callback)
        
        await sio.emit('web_agent_status', {'status': 'completed', 'result': result})
        await sio.emit('web_agent_result', {'result': result, 'task': prompt})
        await sio.emit('status', {'msg': f'Web Agent completed'})
        
    except Exception as e:
        print(f"[SERVER] Web agent error: {e}")
        import traceback
        traceback.print_exc()
        await sio.emit('web_agent_status', {'status': 'error', 'error': str(e)})
        await sio.emit('error', {'msg': f"Web Agent Error: {str(e)}"})


@sio.event
async def web_research(sid, data):
    """Handle web research request - specialized for gathering information."""
    global standalone_web_agent
    topic = data.get('topic') or data.get('prompt')
    sources = data.get('sources', 3)
    print(f"[SERVER] Web research: {topic}")
    
    # Use audio_loop's web agent if available, otherwise use standalone
    web_agent = None
    if audio_loop and audio_loop.web_agent:
        web_agent = audio_loop.web_agent
    elif standalone_web_agent:
        web_agent = standalone_web_agent
    else:
        try:
            from web_agent import WebAgent
            standalone_web_agent = WebAgent()
            web_agent = standalone_web_agent
        except Exception as e:
            print(f"[SERVER] Failed to initialize Web agent: {e}")

    if not web_agent:
        await sio.emit('error', {'msg': 'Web Agent not available'})
        return
        
    try:
        await sio.emit('status', {'msg': f'Researching: {topic[:30]}...'})
        await sio.emit('web_agent_status', {'status': 'researching', 'topic': topic})
        
        async def update_callback(image_b64, log):
            await sio.emit('browser_frame', {'image': image_b64, 'log': log})
        
        # Use research-optimized prompt
        research_prompt = f"""Research the topic: "{topic}"

Instructions:
1. Search Google for relevant information
2. Visit up to {sources} authoritative sources
3. Extract and summarize key findings
4. When finished, provide a comprehensive summary

Focus on finding accurate, up-to-date information."""

        result = await web_agent.run_task(research_prompt, update_callback=update_callback, max_steps=20)
        
        await sio.emit('web_agent_status', {'status': 'completed'})
        await sio.emit('web_research_result', {
            'topic': topic,
            'result': result,
            'sources': sources
        })
        await sio.emit('status', {'msg': 'Research completed'})
        
    except Exception as e:
        print(f"[SERVER] Web research error: {e}")
        import traceback
        traceback.print_exc()
        await sio.emit('web_agent_status', {'status': 'error', 'error': str(e)})
        await sio.emit('error', {'msg': f"Research Error: {str(e)}"})


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


@sio.event
async def get_conversations(sid, data=None):
    """Return list of all conversation metadata."""
    try:
        convs = MEMORY.get_conversations()
        await sio.emit('conversations_list', {'conversations': convs}, to=sid)
    except Exception as e:
        print(f"[SERVER] get_conversations error: {e}")
        await sio.emit('conversations_list', {'conversations': []}, to=sid)


@sio.event
async def create_conversation(sid, data=None):
    """Create a new conversation and switch to it."""
    global _active_conv_id, audio_loop
    try:
        import uuid
        conv_id = str(uuid.uuid4())
        title = (data or {}).get('title', 'New Chat')
        now = _now_iso()
        MEMORY.create_conversation(conv_id, title, now)

        # Save LLM history for the previous conversation, then reset
        if audio_loop and _active_conv_id:
            hist = audio_loop.export_llm_history()
            MEMORY.save_llm_history(_active_conv_id, hist)
            audio_loop.reset_llm_context()

        _active_conv_id = conv_id
        convs = MEMORY.get_conversations()
        await sio.emit('conversation_created', {
            'conversation': {'id': conv_id, 'title': title, 'created_at': now, 'updated_at': now},
            'conversations': convs,
        }, to=sid)
    except Exception as e:
        print(f"[SERVER] create_conversation error: {e}")


@sio.event
async def load_conversation(sid, data=None):
    """Switch to an existing conversation, restoring its LLM context."""
    global _active_conv_id, audio_loop
    try:
        conv_id = (data or {}).get('id', '')
        if not conv_id:
            return

        # Save current context before switching
        if audio_loop and _active_conv_id and _active_conv_id != conv_id:
            hist = audio_loop.export_llm_history()
            MEMORY.save_llm_history(_active_conv_id, hist)
            audio_loop.reset_llm_context()

        # Load target conversation's LLM history
        if audio_loop:
            target_hist = MEMORY.get_llm_history(conv_id)
            if target_hist:
                audio_loop.import_llm_history(target_hist)

        _active_conv_id = conv_id
        msgs = MEMORY.get_conversation_messages(conv_id)
        conv = MEMORY.get_conversation(conv_id)
        await sio.emit('conversation_loaded', {
            'conversation': conv,
            'messages': msgs,
        }, to=sid)
    except Exception as e:
        print(f"[SERVER] load_conversation error: {e}")


@sio.event
async def update_conversation_title(sid, data=None):
    """Rename a conversation."""
    try:
        conv_id = (data or {}).get('id', '')
        title = str((data or {}).get('title', '')).strip()
        if conv_id and title:
            MEMORY.update_conversation_title(conv_id, title)
            convs = MEMORY.get_conversations()
            await sio.emit('conversations_list', {'conversations': convs}, to=sid)
    except Exception as e:
        print(f"[SERVER] update_conversation_title error: {e}")


@sio.event
async def delete_conversation(sid, data=None):
    """Delete a conversation; if it was active, create a fresh one."""
    global _active_conv_id, audio_loop
    try:
        conv_id = (data or {}).get('id', '')
        if not conv_id:
            return
        MEMORY.delete_conversation(conv_id)

        was_active = (_active_conv_id == conv_id)
        if was_active:
            _active_conv_id = None
            if audio_loop:
                audio_loop.reset_llm_context()

        convs = MEMORY.get_conversations()
        await sio.emit('conversation_deleted', {
            'id': conv_id,
            'was_active': was_active,
            'conversations': convs,
        }, to=sid)
    except Exception as e:
        print(f"[SERVER] delete_conversation error: {e}")


@sio.event
async def add_conversation_message(sid, data=None):
    """Persist a message to the active conversation."""
    global _active_conv_id
    try:
        conv_id = str((data or {}).get('conversation_id') or _active_conv_id or '').strip()
        sender = str((data or {}).get('sender', '')).strip()
        text = str((data or {}).get('text', '')).strip()
        timestamp = str((data or {}).get('timestamp', _now_iso())).strip()
        if conv_id and sender and text:
            MEMORY.add_conversation_message(conv_id, sender, text, timestamp)
    except Exception as e:
        print(f"[SERVER] add_conversation_message error: {e}")


@sio.event
async def get_chat_history(sid, data=None):
    """Return persisted chat messages to the requesting client."""
    try:
        limit = int((data or {}).get('limit', 200))
        msgs = MEMORY.get_chat_history(limit=limit)
        await sio.emit('chat_history', {'messages': msgs}, to=sid)
    except Exception as e:
        print(f"[SERVER] get_chat_history error: {e}")
        await sio.emit('chat_history', {'messages': []}, to=sid)


@sio.event
async def add_chat_message(sid, data):
    """Persist a single chat message sent from the frontend."""
    try:
        sender = str(data.get('sender', '')).strip()
        text = str(data.get('text', '')).strip()
        timestamp = str(data.get('timestamp', '')).strip()
        session = str(data.get('session', '')).strip()
        if sender and text:
            MEMORY.add_chat_message(sender, text, timestamp, session)
    except Exception as e:
        print(f"[SERVER] add_chat_message error: {e}")


@sio.event
async def clear_chat_history(sid, data=None):
    """Delete all persisted chat messages and notify the client."""
    try:
        MEMORY.clear_chat_history()
        await sio.emit('chat_history', {'messages': []}, to=sid)
    except Exception as e:
        print(f"[SERVER] clear_chat_history error: {e}")


@sio.event
async def save_memory(sid, data):
    """Save conversation to file."""
    try:
        messages = data.get('messages', [])
        if not messages:
            return
        
        memory_dir = Path("long_term_memory")
        memory_dir.mkdir(exist_ok=True)
        
        from datetime import datetime
        provided_name = data.get('filename')
        
        if provided_name:
            if not provided_name.endswith('.txt'):
                provided_name += '.txt'
            filename = memory_dir / Path(provided_name).name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = memory_dir / f"memory_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            for msg in messages:
                sender = msg.get('sender', 'Unknown')
                text = msg.get('text', '')
                f.write(f"[{sender}]: {text}\n")
        
        print(f"[SERVER] Memory saved: {filename}")
        await sio.emit('status', {'msg': 'Memory Saved'})
        
    except Exception as e:
        print(f"[SERVER] Save memory error: {e}")
        await sio.emit('error', {'msg': f"Failed to save: {str(e)}"})


@sio.event
async def upload_memory(sid, data):
    """Load memory context."""
    payload = data or {}
    memory_text = payload.get('memory', '') if isinstance(payload, dict) else ''
    if not memory_text:
        return

    if not audio_loop:
        await sio.emit('error', {'msg': 'J.A.R.V.I.S not running'})
        return

    try:
        print("[SERVER] Loading memory context...")
        context_msg = (
            "System Notification: The user uploaded a long-term memory file. "
            "Load the following context into your understanding. "
            "The format is a text log of previous conversations:\n\n"
            + memory_text
        )

        if hasattr(audio_loop, "llm") and hasattr(audio_loop.llm, "add_message"):
            audio_loop.llm.add_message("system", context_msg)
        else:
            await audio_loop.process_text_input(context_msg)

        await sio.emit('status', {'msg': 'Memory Loaded'})
    except Exception as e:
        print(f"[SERVER] Upload memory error: {type(e).__name__}: {e}")
        await sio.emit('error', {'msg': f"Failed to upload memory: {str(e)}"})


@sio.event
async def recommendation_feedback(sid, data=None):
    payload = data or {}
    if not isinstance(payload, dict):
        payload = {}

    outcome = payload.get('outcome')
    rec_id = payload.get('rec_id')
    note = payload.get('note')

    if not audio_loop:
        await sio.emit('error', {'msg': 'J.A.R.V.I.S not running'}, room=sid)
        return

    try:
        res = audio_loop.record_feedback(outcome=str(outcome or ''), rec_id=str(rec_id) if rec_id else None, note=note)
        await sio.emit('recommendation_feedback_result', res, room=sid)
    except Exception as e:
        logger.exception("recommendation_feedback failed: %s", e)
        await sio.emit('recommendation_feedback_result', {'ok': False, 'error': str(e)}, room=sid)


@sio.event
async def get_learning_summary(sid, data=None):
    if not audio_loop:
        await sio.emit('learning_summary', {'error': 'J.A.R.V.I.S not running'}, room=sid)
        return

    try:
        summary = audio_loop.get_learning_summary()
        await sio.emit('learning_summary', summary, room=sid)
    except Exception as e:
        logger.exception("get_learning_summary failed: %s", e)
        await sio.emit('learning_summary', {'error': str(e)}, room=sid)


@sio.event
async def get_settings(sid, data=None):
    try:
        async with SETTINGS_LOCK:
            snapshot = copy.deepcopy(SETTINGS)
        _emit('settings', snapshot, room=sid)
    except Exception as e:
        logger.exception("get_settings failed: %s", e)


@sio.event
async def update_settings(sid, data=None):
    try:
        payload = data or {}
        if isinstance(payload, dict):
            keys = list(payload.keys())
        else:
            keys = []
        print(f"[SERVER] Updating settings: {keys}")

        async with SETTINGS_LOCK:
            if isinstance(payload, dict) and "tool_permissions" in payload and isinstance(payload.get("tool_permissions"), dict):
                SETTINGS["tool_permissions"].update(payload["tool_permissions"])

            if isinstance(payload, dict) and "llm" in payload:
                print("[SERVER] Ignoring client LLM settings update (model selection is managed internally).")

            if isinstance(payload, dict) and "tts" in payload and isinstance(payload.get("tts"), dict):
                SETTINGS["tts"].update(payload["tts"])

            if isinstance(payload, dict) and "stt" in payload and isinstance(payload.get("stt"), dict):
                SETTINGS["stt"].update(payload["stt"])

            if isinstance(payload, dict) and "image" in payload and isinstance(payload.get("image"), dict):
                if not isinstance(SETTINGS.get("image"), dict):
                    SETTINGS["image"] = {}
                SETTINGS["image"].update(payload["image"])

            if isinstance(payload, dict) and "personality" in payload and isinstance(payload.get("personality"), dict):
                if not isinstance(SETTINGS.get("personality"), dict):
                    SETTINGS["personality"] = {}
                SETTINGS["personality"].update(payload["personality"])

            snapshot = copy.deepcopy(SETTINGS)

        if audio_loop:
            try:
                audio_loop.update_permissions(snapshot.get("tool_permissions", {}))
            except Exception:
                logger.exception("Failed to update permissions")

        if audio_loop and hasattr(audio_loop, "tts"):
            try:
                tts_cfg = snapshot.get("tts", {}) or {}
                voice = tts_cfg.get("voice")
                voice_en = tts_cfg.get("voice_en") or voice
                voice_vi = tts_cfg.get("voice_vi")
                rate_en = tts_cfg.get("rate_en")
                pitch_en = tts_cfg.get("pitch_en")
                rate_vi = tts_cfg.get("rate_vi")
                pitch_vi = tts_cfg.get("pitch_vi")
                auto_detect = tts_cfg.get("auto_detect")

                if voice_en:
                    audio_loop.tts_voice_en = voice_en
                if voice_vi:
                    audio_loop.tts_voice_vi = voice_vi
                if rate_en is not None:
                    audio_loop.tts_rate_en = rate_en
                if pitch_en is not None:
                    audio_loop.tts_pitch_en = pitch_en
                if rate_vi is not None:
                    audio_loop.tts_rate_vi = rate_vi
                if pitch_vi is not None:
                    audio_loop.tts_pitch_vi = pitch_vi
                if auto_detect is not None:
                    audio_loop.tts_auto_detect = bool(auto_detect)

                if voice:
                    try:
                        audio_loop.tts.set_voice(voice)
                    except Exception as e:
                        print(f"[SERVER] Failed to set TTS voice: {e}")
            except Exception:
                logger.exception("Failed to apply TTS settings")

        if audio_loop and hasattr(audio_loop, "set_personality_config"):
            try:
                audio_loop.set_personality_config(snapshot.get("personality", {}))
            except Exception:
                logger.exception("Failed to apply personality settings")

        try:
            await asyncio.to_thread(save_settings_snapshot, snapshot)
        except Exception:
            logger.exception("Failed to save settings")

        _emit('settings', snapshot, room=sid)
    except Exception as e:
        logger.exception("update_settings failed: %s", e)


@sio.event
async def set_identity(sid, data=None):
    try:
        payload = data or {}
        if not isinstance(payload, dict):
            return

        patch = {}
        if "identity" in payload and isinstance(payload.get("identity"), dict):
            patch.update(payload.get("identity") or {})
        else:
            patch.update(payload)

        await _persist_identity_update(patch)

        if audio_loop:
            try:
                ident = SETTINGS.get("identity") if isinstance(SETTINGS, dict) else None
                if isinstance(ident, dict):
                    if isinstance(ident.get("user_name"), str):
                        audio_loop.user_name = ident.get("user_name")
                    if isinstance(ident.get("assistant_name"), str):
                        audio_loop.assistant_name = ident.get("assistant_name")
            except Exception:
                pass
    except Exception as e:
        logger.exception("set_identity failed: %s", e)


@sio.event
async def forget_user_name(sid, data=None):
    try:
        try:
            MEMORY.delete_personal("user_name")
        except Exception:
            pass

        await _persist_identity_update({"user_name": DEFAULT_SETTINGS["identity"]["user_name"]})
    except Exception as e:
        logger.exception("forget_user_name failed: %s", e)


@sio.event
async def get_tool_permissions(sid, data=None):
    """Get tool permissions (legacy)."""
    try:
        async with SETTINGS_LOCK:
            perms = copy.deepcopy(SETTINGS.get("tool_permissions", {}))
        _emit('tool_permissions', perms, room=sid)
    except Exception as e:
        logger.exception("get_tool_permissions failed: %s", e)


@sio.event
async def update_tool_permissions(sid, data=None):
    """Update tool permissions (legacy)."""
    try:
        payload = data or {}
        async with SETTINGS_LOCK:
            if isinstance(payload, dict):
                SETTINGS["tool_permissions"].update(payload)
            perms = copy.deepcopy(SETTINGS.get("tool_permissions", {}))

        if audio_loop:
            try:
                audio_loop.update_permissions(perms)
            except Exception:
                logger.exception("Failed to update permissions")

        try:
            await asyncio.to_thread(save_settings_snapshot, {**SETTINGS, "tool_permissions": perms})
        except Exception:
            logger.exception("Failed to save settings")

        _emit('tool_permissions', perms, room=sid)
    except Exception as e:
        logger.exception("update_tool_permissions failed: %s", e)


if __name__ == "__main__":
    print("[SERVER] Starting uvicorn server on http://127.0.0.1:8000")
    print("[SERVER] /status endpoint should be available immediately")
    uvicorn.run(
        "server:app_socketio",
        host="127.0.0.1",
        port=8000,
        access_log=False,
        reload=False,
        loop="asyncio",
    )
