"""
Speech-to-Text Engine using faster-whisper for offline multilingual recognition.

Supports 99+ languages including Vietnamese and English with automatic language detection.
The frontend converts all audio to 16 kHz mono WAV before sending, so no ffmpeg is required.

Install: py -3.11 -m pip install faster-whisper
Model is auto-downloaded on first use:
  - 'base'  ~145 MB  — fast, good accuracy
  - 'small' ~466 MB  — best balance for vi+en  ← default
"""

import logging
import os
import tempfile
import threading
from typing import Optional

logger = logging.getLogger("jarvis.stt")

_whisper_model = None
_whisper_lock = threading.Lock()

DEFAULT_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "whisper")

# Set to True after a successful load; False after a hard failure.
_model_load_attempted = False
_model_load_failed = False
_model_load_error: Optional[str] = None


def _get_model(model_size: str = DEFAULT_MODEL_SIZE):
    """Load (or return cached) faster-whisper model. Thread-safe, logs clearly."""
    global _whisper_model, _model_load_attempted, _model_load_failed, _model_load_error

    if _whisper_model is not None:
        return _whisper_model
    if _model_load_failed:
        return None

    with _whisper_lock:
        if _whisper_model is not None:
            return _whisper_model
        if _model_load_failed:
            return None

        _model_load_attempted = True
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            _model_load_failed = True
            _model_load_error = (
                "faster-whisper is not installed. "
                "Run:  py -3.11 -m pip install faster-whisper"
            )
            logger.error("[STT] %s", _model_load_error)
            print(f"[STT] {_model_load_error}")
            return None

        # Detect CUDA availability
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        compute = "float16" if device == "cuda" else "int8"
        os.makedirs(CACHE_DIR, exist_ok=True)

        logger.info("[STT] Loading faster-whisper '%s' (%s/%s)…", model_size, device, compute)
        print(f"[STT] Loading faster-whisper '{model_size}' ({device}/{compute})…")

        try:
            _whisper_model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute,
                download_root=CACHE_DIR,
            )
            logger.info("[STT] faster-whisper ready ('%s', %s/%s)", model_size, device, compute)
            print(f"[STT] faster-whisper ready ('{model_size}', {device}/{compute})")
            return _whisper_model
        except Exception as e:
            _model_load_failed = True
            _model_load_error = str(e)
            logger.error("[STT] Failed to load model: %s", e)
            print(f"[STT] Failed to load model: {e}")
            return None


def transcribe_bytes(
    audio_bytes: bytes,
    mime_type: str = "audio/wav",
    language: Optional[str] = None,
    model_size: str = DEFAULT_MODEL_SIZE,
) -> dict:
    """
    Transcribe raw audio bytes.

    The browser converts all recordings to 16 kHz mono WAV before sending,
    so no ffmpeg is needed — whisper reads the WAV directly.

    Returns a dict:
        text (str)        — transcribed text (empty string if nothing heard)
        language (str)    — ISO code of detected language
        error (str|None)  — human-readable error if transcription failed
    """
    model = _get_model(model_size)
    if model is None:
        err = _model_load_error or "faster-whisper not available"
        logger.warning("[STT] Transcription skipped: %s", err)
        return {"text": "", "language": language or "unknown", "error": err}

    if not audio_bytes:
        return {"text": "", "language": language or "unknown", "error": "empty audio"}

    suffix = ".wav" if "wav" in (mime_type or "").lower() else _mime_to_suffix(mime_type)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        logger.debug("[STT] Transcribing %d bytes (%s, lang=%s)…", len(audio_bytes), suffix, language or "auto")

        segments, info = model.transcribe(
            tmp_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
        )
        # Consume the generator so all segments are evaluated before cleanup
        text = " ".join(seg.text.strip() for seg in segments).strip()
        detected_lang = info.language if info else (language or "unknown")

        logger.info("[STT] [%s] %r", detected_lang, text)
        print(f"[STT] [{detected_lang}] {text!r}")
        return {"text": text, "language": detected_lang, "error": None}

    except Exception as e:
        logger.error("[STT] Transcription error: %s", e)
        print(f"[STT] Transcription error: {e}")
        return {"text": "", "language": language or "unknown", "error": str(e)}
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _mime_to_suffix(mime_type: str) -> str:
    mime_type = (mime_type or "").lower().split(";")[0].strip()
    return {
        "audio/webm": ".webm",
        "audio/ogg": ".ogg",
        "audio/wav": ".wav",
        "audio/mp4": ".mp4",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "video/webm": ".webm",
    }.get(mime_type, ".webm")


def preload_model(model_size: str = DEFAULT_MODEL_SIZE):
    """Pre-warm the model in a background thread at server startup."""
    t = threading.Thread(target=_get_model, args=(model_size,), daemon=True, name="whisper-preload")
    t.start()
