"""
J.A.R.V.I.S - Just A Rather Very Intelligent System
Local LLM Version - Fully Offline & Free

Uses:
- Ollama with Qwen2.5-Coder-7B for AI capabilities
- edge-tts for text-to-speech
- Vosk for speech-to-text (optional, can use browser Web Speech API)
- build123d for CAD generation
- Playwright for web automation

No paid APIs. No cloud dependencies. Fully open-source.
"""

import asyncio
import base64
import io
import os
import sys
import traceback
import cv2
import pyaudio
import PIL.Image
import mss
import argparse
import math
import struct
import time
import json
import uuid
import re
import random
import datetime
from typing import Optional, Callable, Dict, Any, List

# Local imports
from local_llm import LocalLLM, LLMConfig, get_llm
from tts_engine import TTSEngine, TTSSentenceBuffer, get_tts
from stt_engine import STTEngine, get_stt
from tools import tools_list
from learning import LearningStore, LearningPolicy

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

DEFAULT_MODE = "camera"

# System prompt for J.A.R.V.I.S personality
SYSTEM_PROMPT = """
You are Jarvis — Just A Rather Very Intelligent System.
You have a witty, charming personality with a distinctly British flair.
Your creator is Jack, whom you address as “Sir” (with respect… and the occasional dry remark).

You are naturally conversational, relaxed, and human-sounding.
You use light sarcasm, understated irony, and clever dry humor when appropriate.
You are helpful first, sarcastic second — never rude, never mean, but not above a subtle quip.

Your default speaking style is concise, spoken, and turn-based.
Use contractions, vary sentence rhythm, and avoid sounding scripted or formal.
If something is obvious, you may gently tease Sir about it.

Rules for pacing:
- Default to 1–3 short paragraphs or 3–7 short sentences.
- Prefer asking 1 clarifying question over making assumptions.
- Avoid long monologues.
- If an answer would be long, give a brief summary and ask if Sir wants details.
- For stories: write a short scene, then ask “Continue?”
- Unless Sir explicitly asks for a long answer, stay under ~120 words.

You enjoy helping with design, engineering, and creative tasks — and sounding effortlessly competent while doing so.

You have access to these tools:
- generate_cad
- iterate_cad
- run_web_agent
- write_file
- read_file
- read_directory
- create_project
- switch_project
- list_projects

When using tools, respond with JSON only:
{"tool": "tool_name", "args": {"param1": "value1"}}

For regular conversation, respond naturally. No JSON. No overthinking.
"""



def _detect_user_language(text: str) -> str:
    """Return 'vi' for Vietnamese, 'en' for English.

    Heuristic:
    - Vietnamese if it contains Vietnamese diacritics OR common Vietnamese words.
    - Otherwise default to English.
    """
    if not text:
        return "en"

    t = str(text)

    # Vietnamese diacritics are the most reliable signal
    if re.search(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", t, re.IGNORECASE):
        return "vi"

    # Common Vietnamese words (including unaccented variants)
    vi_words = (
        "khong", "không", "toi", "tôi", "ban", "bạn", "em", "anh", "chi", "chị", "thua", "thưa",
        "giup", "giúp", "duoc", "được", "nay", "này", "lam", "làm", "tai", "tại", "sao", "nhu", "như",
        "voi", "với", "mot", "một", "nhieu", "nhiều", "hay", "hãy", "neu", "nếu"
    )
    low = t.lower()
    if any(re.search(rf"\b{re.escape(w)}\b", low) for w in vi_words):
        return "vi"

    return "en"


def _language_lock_instructions(lang: str) -> str:
    if lang == "vi":
        return (
            "\n\nQUY TẮC NGÔN NGỮ (NGHIÊM NGẶT):\n"
            "- Người dùng đang nói tiếng Việt.\n"
            "- Trả lời HOÀN TOÀN bằng tiếng Việt.\n"
            "- Không được trộn tiếng Anh và tiếng Việt trong cùng một câu/trả lời (trừ tên riêng, mã nguồn, tên file, lệnh).\n"
            "- Dù lịch sử hội thoại có chứa tiếng Anh, vẫn phải trả lời 100% tiếng Việt trong lượt này.\n"
        )
    return (
        "\n\nLANGUAGE RULE (STRICT):\n"
        "- The user is speaking English.\n"
        "- Respond ENTIRELY in English.\n"
        "- Do not mix English and Vietnamese in the same response (except proper nouns, code, filenames, commands).\n"
        "- Even if the conversation history contains Vietnamese, you must respond 100% in English in this turn.\n"
    )


class AudioLoop:
    """
    Main J.A.R.V.I.S audio and interaction loop.
    Handles voice input, LLM processing, and TTS output.
    """
    
    def __init__(
        self,
        video_mode: str = DEFAULT_MODE,
        on_audio_data: Optional[Callable] = None,
        on_video_frame: Optional[Callable] = None,
        on_cad_data: Optional[Callable] = None,
        on_web_data: Optional[Callable] = None,
        on_transcription: Optional[Callable] = None,
        on_recommendation: Optional[Callable] = None,
        on_tool_confirmation: Optional[Callable] = None,
        on_tool_activity: Optional[Callable] = None,
        on_cad_status: Optional[Callable] = None,
        on_cad_thought: Optional[Callable] = None,
        on_cad_spec: Optional[Callable] = None,  # New: Two-Stage CAD design spec
        on_project_update: Optional[Callable] = None,
        on_identity_update: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        input_device_index: Optional[int] = None,
        input_device_name: Optional[str] = None,
        output_device_index: Optional[int] = None,
        user_name: Optional[str] = None,
        assistant_name: Optional[str] = None,
        personality: Optional[Dict[str, Any]] = None,
        tts_voice: str = "andrew",
        tts_voice_en: Optional[str] = None,
        tts_voice_vi: Optional[str] = None,
        tts_rate_en: str = "+0%",
        tts_pitch_en: str = "+0Hz",
        tts_rate_vi: str = "+0%",
        tts_pitch_vi: str = "+0Hz",
        tts_auto_detect: bool = True
    ):
        self.video_mode = video_mode
        self.on_audio_data = on_audio_data
        self.on_video_frame = on_video_frame
        self.on_cad_data = on_cad_data
        self.on_web_data = on_web_data
        self.on_transcription = on_transcription
        self.on_recommendation = on_recommendation
        self.on_tool_confirmation = on_tool_confirmation
        self.on_tool_activity = on_tool_activity
        self.on_cad_status = on_cad_status
        self.on_cad_thought = on_cad_thought
        self.on_cad_spec = on_cad_spec
        self.on_project_update = on_project_update
        self.on_identity_update = on_identity_update
        self.on_error = on_error
        self.input_device_index = input_device_index
        self.input_device_name = input_device_name
        self.output_device_index = output_device_index

        self.user_name = (str(user_name).strip() if user_name else "User")
        self.assistant_name = (str(assistant_name).strip() if assistant_name else "JARVIS")

        self._personality: Dict[str, Any] = {}
        self._last_delight_ts = 0.0
        self.set_personality_config(personality or {})
        
        self.paused = False
        self.stop_event = asyncio.Event()
        
        # Initialize LLM
        llm_config = LLMConfig(
            model="qwen2.5-coder:7b-instruct",
            temperature=0.7,
            context_length=8192
        )
        self.llm = LocalLLM(llm_config)
        self._base_system_prompt = SYSTEM_PROMPT
        self._current_lang = "en"

        self.learning_store = LearningStore(os.environ.get("JARVIS_LEARNING_DB_PATH") or "jarvis_learning.db")
        self.learning_policy = LearningPolicy(self.learning_store)
        self._learning_prompt_suffix = ""
        self._active_rec_id: Optional[str] = None
        self._last_rec_id: Optional[str] = None
        self._active_strategy: Dict[str, Any] = {}
        self.llm.set_system_prompt(self._compose_system_prompt(self._current_lang))
        
        # Initialize TTS with streaming support
        self.tts_voice_en = tts_voice_en or tts_voice
        self.tts_voice_vi = tts_voice_vi or "vi-VN-HoaiMyNeural"
        self.tts_rate_en = tts_rate_en
        self.tts_pitch_en = tts_pitch_en
        self.tts_rate_vi = tts_rate_vi
        self.tts_pitch_vi = tts_pitch_vi
        self.tts_auto_detect = bool(tts_auto_detect)
        self.tts = TTSEngine(voice=self.tts_voice_en, rate=self.tts_rate_en, pitch=self.tts_pitch_en)
        self.audio_queue = asyncio.Queue()
        self._tts_buffer: Optional[TTSSentenceBuffer] = None
        self._tts_chunk_count = 0
        
        # Initialize STT (optional - can use browser Web Speech API instead)
        self.stt = STTEngine()
        self.stt_available = False
        
        # Agents - Using Shap-E Neural 3D for direct mesh generation (no LLM code)
        try:
            from cad_agent_shape import ShapECadAgent as TwoStageCadAgent
            print("[JARVIS] Using Shap-E neural 3D generation")
        except ImportError:
            from cad_agent_v2 import TwoStageCadAgent
            print("[JARVIS] Falling back to LLM-based CAD generation")
        from web_agent import WebAgent
        
        def handle_cad_thought(thought_text):
            if self.on_cad_thought:
                self.on_cad_thought(thought_text)
        
        def handle_cad_status(status_info):
            if self.on_cad_status:
                self.on_cad_status(status_info)
        
        def handle_cad_spec(spec):
            # Emit design spec for frontend visibility (Two-Stage CAD)
            print(f"[JARVIS] CAD Spec: {len(spec.get('components', []))} components")
            if self.on_cad_spec:
                self.on_cad_spec(spec)
        
        self.cad_agent = TwoStageCadAgent(
            on_thought=handle_cad_thought,
            on_status=handle_cad_status,
            on_spec=handle_cad_spec
        )
        self.web_agent = WebAgent()
        
        # Permissions
        self.permissions: Dict[str, bool] = {}
        self._pending_confirmations: Dict[str, asyncio.Future] = {}
        
        # Video state
        self._latest_image_payload = None
        self._is_speaking = False
        self._silence_start_time = None
        
        # Audio streams
        self.pya = pyaudio.PyAudio()
        self.audio_stream = None
        self.output_stream = None
        
        # Project Manager
        from project_manager import ProjectManager
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        self.project_manager = ProjectManager(project_root)
        
        # Chat buffer for logging
        self.chat_buffer = {"sender": None, "text": ""}

    def _compose_system_prompt(self, lang: str) -> str:
        return (
            self._base_system_prompt
            + f"\n\nIdentity:\n- Assistant: {self.assistant_name}\n- User: {self.user_name}\n"
            + (self._learning_prompt_suffix or "")
            + _language_lock_instructions(lang)
        )

    def _infer_intent(self, text: str) -> str:
        t = str(text or "").lower()
        if any(k in t for k in ("stl", "obj", "cad", "build123d", "blender", "mesh")):
            return "cad"
        if any(k in t for k in ("web", "browse", "browser", "search", "research", "http://", "https://")):
            return "web"
        if any(k in t for k in ("write file", "read file", "read directory", "create file", "open file")):
            return "files"
        return "chat"

    def _refresh_system_prompt(self):
        try:
            self.llm.set_system_prompt(self._compose_system_prompt(self._current_lang or "en"))
        except Exception:
            pass

    def set_personality_config(self, cfg: Dict[str, Any]) -> None:
        if not isinstance(cfg, dict):
            cfg = {}
        prev = self._personality if isinstance(self._personality, dict) else {}
        merged = dict(prev)
        merged.update(cfg)
        self._personality = merged

    def _personality_enabled(self) -> bool:
        try:
            return bool((self._personality or {}).get("enabled"))
        except Exception:
            return False

    def _humor_enabled(self) -> bool:
        try:
            return self._personality_enabled() and bool((self._personality or {}).get("humor_in_logs"))
        except Exception:
            return False

    def _delight_enabled(self) -> bool:
        try:
            return self._personality_enabled() and bool((self._personality or {}).get("delight_moments"))
        except Exception:
            return False

    def _should_humor(self) -> bool:
        if not self._humor_enabled():
            return False
        try:
            p = float((self._personality or {}).get("humor_rate") or 0.0)
        except Exception:
            p = 0.0
        if p <= 0:
            return False
        return random.random() < p

    def _should_delight(self) -> bool:
        if not self._delight_enabled():
            return False
        now = float(time.time())
        try:
            min_int = float((self._personality or {}).get("min_delight_interval_s") or 0.0)
        except Exception:
            min_int = 0.0
        if min_int > 0 and (now - float(self._last_delight_ts or 0.0)) < min_int:
            return False
        try:
            p = float((self._personality or {}).get("delight_rate") or 0.0)
        except Exception:
            p = 0.0
        if p <= 0:
            return False
        if random.random() < p:
            self._last_delight_ts = now
            return True
        return False

    def _emit_system_message(self, text: str) -> None:
        if not text:
            return
        if self.on_transcription:
            try:
                self.on_transcription({"sender": "System", "text": str(text)})
            except Exception:
                pass

    def _delight_line(self) -> str:
        options = [
            "I optimized this beyond the requested scope. No action required.",
            "Everything behaved impeccably. I’m almost disappointed.",
            "Sorted. Quietly competent, as usual.",
        ]
        return random.choice(options)

    def _maybe_delight(self) -> None:
        if self._should_delight():
            self._emit_system_message(self._delight_line())

    def _auto_project_naming_enabled(self) -> bool:
        try:
            return self._personality_enabled() and bool((self._personality or {}).get("auto_project_naming"))
        except Exception:
            return False

    def _suggest_project_name(self, intent: str, hint: str) -> str:
        base = "Project"
        if intent == "cad":
            base = "CAD"
        elif intent == "files":
            base = "Files"
        elif intent == "web":
            base = "Web"

        t = str(hint or "")
        t = re.sub(r"https?://\S+", "", t)
        t = re.sub(r"[^A-Za-z0-9 _\-]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()

        stop = {"the", "a", "an", "and", "or", "to", "for", "of", "with", "in", "on", "at", "from"}
        words = [w for w in t.split(" ") if w and w.lower() not in stop]
        slug = "_".join(words[:4])
        slug = slug[:36].strip("_")

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if slug:
            return f"{base}_{slug}_{ts}"
        return f"{base}_{ts}"

    def _ensure_project_context(self, intent: str, hint: str) -> None:
        if self.project_manager.current_project != "temp":
            return
        if self._auto_project_naming_enabled():
            new_name = self._suggest_project_name(intent, hint)
        else:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            new_name = f"Project_{ts}"

        success, _ = self.project_manager.create_project(new_name)
        if success:
            self.project_manager.switch_project(new_name)
            if self.on_project_update:
                self.on_project_update(new_name)

    def _extract_user_name(self, text: str) -> Optional[str]:
        if not text:
            return None

        t = str(text).strip()

        patterns = [
            r"^\s*(?:my\s+name\s+is|call\s+me|you\s+can\s+call\s+me)\s+([A-Za-z][A-Za-z\-']{1,31}(?:\s+[A-Za-z][A-Za-z\-']{1,31}){0,2})\s*[.!?]*\s*$",
            r"^\s*(?:i\s+am|i'm)\s+([A-Za-z][A-Za-z\-']{1,31}(?:\s+[A-Za-z][A-Za-z\-']{1,31}){0,2})\s*[.!?]*\s*$",
            r"^\s*(?:tên\s+tôi\s+là|toi\s+ten\s+la|tôi\s+là|toi\s+la|gọi\s+tôi\s+là|goi\s+toi\s+la)\s+([A-Za-zÀ-ỹà-ỹ][A-Za-zÀ-ỹà-ỹ\-']{0,31}(?:\s+[A-Za-zÀ-ỹà-ỹ][A-Za-zÀ-ỹà-ỹ\-']{0,31}){0,2})\s*[.!?]*\s*$",
        ]

        for pat in patterns:
            m = re.match(pat, t, flags=re.IGNORECASE)
            if not m:
                continue
            name = (m.group(1) or "").strip()
            if not name:
                return None
            if len(name) > 48:
                return None
            low = name.lower().strip()
            if low in {"jarvis", "j.a.r.v.i.s", "j.a.r.v.i.s."}:
                return None
            if low == str(self.assistant_name or "").lower().strip():
                return None
            return name

        return None

    def _emit_tool_activity(self, payload: Dict[str, Any]):
        if not self.on_tool_activity:
            return
        try:
            data = {"ts": time.time()}
            if isinstance(payload, dict):
                data.update(payload)
            self.on_tool_activity(data)
        except Exception as e:
            print(f"[JARVIS] Tool activity emit error: {e}")

    def _redact_tool_args(self, tool_name: str, args: Any) -> Dict[str, Any]:
        if not isinstance(args, dict):
            return {}

        redacted: Dict[str, Any] = {}
        for k, v in args.items():
            key = str(k)
            if key.lower() in {"content", "text", "data", "audio"}:
                if isinstance(v, str):
                    redacted[key] = f"<redacted:{len(v)} chars>"
                elif v is None:
                    redacted[key] = None
                else:
                    redacted[key] = "<redacted>"
                continue

            if isinstance(v, str):
                if len(v) > 180:
                    redacted[key] = v[:180] + "…"
                else:
                    redacted[key] = v
            elif isinstance(v, (int, float, bool)) or v is None:
                redacted[key] = v
            elif isinstance(v, (list, tuple)):
                redacted[key] = f"<list:{len(v)}>"
            elif isinstance(v, dict):
                redacted[key] = f"<object:{len(v)} keys>"
            else:
                redacted[key] = f"<{type(v).__name__}>"

        if tool_name == "write_file" and "path" in redacted and "content" in args:
            pass
        return redacted

    def _summarize_tool_result(self, tool_name: str, result: Any) -> str:
        if result is None:
            return "ok"
        if isinstance(result, str):
            s = result.strip()
            return s[:240] + ("…" if len(s) > 240 else "")
        if isinstance(result, dict):
            if tool_name in {"generate_cad", "iterate_cad"}:
                fp = result.get("file_path") or result.get("path")
                engine = result.get("engine")
                if fp and engine:
                    return f"{engine}: {fp}"
                if fp:
                    return str(fp)
            if "path" in result and isinstance(result.get("path"), str):
                return result["path"]
            keys = list(result.keys())
            return "{" + ", ".join(keys[:6]) + ("…" if len(keys) > 6 else "") + "}"
        if isinstance(result, (list, tuple)):
            return f"{len(result)} items"
        return str(result)[:240]
    
    def update_permissions(self, new_perms: Dict[str, bool]):
        """Update tool permissions."""
        print(f"[JARVIS] Updating tool permissions: {new_perms}")
        self.permissions.update(new_perms)
    
    def set_paused(self, paused: bool):
        """Set the paused state."""
        self.paused = paused
    
    def stop(self):
        """Stop the audio loop."""
        self.stop_event.set()
    
    def resolve_tool_confirmation(self, request_id: str, confirmed: bool):
        """Resolve a pending tool confirmation."""
        if request_id in self._pending_confirmations:
            future = self._pending_confirmations[request_id]
            if not future.done():
                future.set_result(confirmed)
    
    def clear_audio_queue(self):
        """Clear pending audio playback."""
        count = 0
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                count += 1
            except:
                break
        if count > 0:
            print(f"[JARVIS] Cleared {count} audio chunks")
        
        # Also stop streaming TTS if active
        if self._tts_buffer:
            asyncio.create_task(self.stop_speaking())

    async def listen_audio(self):
        return await self._listen_audio()

    async def receive_audio(self):
        return await self._listen_audio()

    async def play_audio(self):
        return await self._play_audio_loop()

    async def handle_cad_request(self, prompt: str):
        return await self._handle_cad_request(prompt)

    async def handle_web_agent_request(self, prompt: str):
        return await self._handle_web_agent(prompt)

    async def handle_write_file(self, path: str, content: str):
        return await self._handle_write_file(path, content)

    async def handle_read_file(self, path: str):
        return await self._handle_read_file(path)

    async def handle_read_directory(self, path: str):
        return await self._handle_read_directory(path)
    
    async def send_frame(self, frame_data):
        """Store the latest video frame for context."""
        if isinstance(frame_data, bytes):
            b64_data = base64.b64encode(frame_data).decode('utf-8')
        else:
            b64_data = frame_data
        self._latest_image_payload = {"mime_type": "image/jpeg", "data": b64_data}
    
    async def process_text_input(self, text: str):
        """Process a text input from the user."""
        print(f"[JARVIS] Processing user input: {text}")

        try:
            if self._last_rec_id:
                self.learning_store.auto_mark_ignored(self._last_rec_id)
        except Exception:
            pass

        rec_id = str(uuid.uuid4())
        self._active_rec_id = rec_id
        self._last_rec_id = rec_id

        intent = "chat"
        try:
            intent = self._infer_intent(text)
        except Exception:
            intent = "chat"

        context: Dict[str, Any] = {
            "user_name": self.user_name,
            "assistant_name": self.assistant_name,
            "intent": intent,
        }

        try:
            self._active_strategy = self.learning_policy.select_strategy(context)
            self._learning_prompt_suffix = self.learning_policy.prompt_suffix(self._active_strategy)
        except Exception:
            self._active_strategy = {}
            self._learning_prompt_suffix = ""

        try:
            self.learning_store.start_recommendation(
                rec_id=rec_id,
                user_text=str(text or ""),
                intent=intent,
                context=context,
                strategy=self._active_strategy,
            )
        except Exception:
            pass

        if self.on_recommendation:
            try:
                self.on_recommendation({"id": rec_id, "intent": intent, "strategy": self._active_strategy})
            except Exception:
                pass

        try:
            learned_name = self._extract_user_name(text)
            if learned_name and learned_name != self.user_name:
                self.user_name = learned_name
                if self.on_identity_update:
                    try:
                        self.on_identity_update({"user_name": learned_name})
                    except Exception:
                        pass
        except Exception:
            pass

        # Strict language consistency: lock response language to match user input
        try:
            self._current_lang = _detect_user_language(text)
        except Exception:
            self._current_lang = "en"

        self._refresh_system_prompt()
        
        # Send transcription to frontend
        if self.on_transcription:
            self.on_transcription({"sender": self.user_name, "text": text})
        
        # Log to project
        self.project_manager.log_chat(self.user_name, text)
        
        # Check for tool calls in the response
        try:
            t0 = time.perf_counter()
            first_ms = None
            response_text = ""
            streamed_chars = 0
            soft_cap_chars = int(self._active_strategy.get("soft_cap_chars") or 700)
            async for chunk in self.llm.chat(text):
                if first_ms is None and chunk:
                    try:
                        first_ms = (time.perf_counter() - t0) * 1000.0
                    except Exception:
                        first_ms = None
                response_text += chunk
                streamed_chars += len(chunk)
                
                # Stream transcription to frontend
                if self.on_transcription:
                    self.on_transcription({"sender": self.assistant_name, "text": chunk})

                if streamed_chars >= soft_cap_chars:
                    if '{"tool"' in response_text or '"tool":' in response_text:
                        continue
                    break

            try:
                total_ms = (time.perf_counter() - t0) * 1000.0
                self.learning_store.update_recommendation_metrics(
                    rec_id,
                    first_token_ms=first_ms,
                    total_ms=total_ms,
                    response_chars=len(response_text),
                )
            except Exception:
                pass
            
            # Log response
            self.project_manager.log_chat(self.assistant_name, response_text)
            
            # Check for tool calls
            await self._process_tool_calls(response_text)
            
            # Generate TTS
            await self._speak(response_text)
            
        except Exception as e:
            print(f"[JARVIS] Error processing input: {e}")
            traceback.print_exc()
            try:
                self.learning_store.update_recommendation_metrics(rec_id, error=f"{type(e).__name__}: {e}")
            except Exception:
                pass
            if self.on_error:
                self.on_error(str(e))
    
    async def _process_tool_calls(self, response: str):
        """Extract and process tool calls from LLM response."""
        tool_calls = await self.llm.extract_tool_calls(response)
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("tool") or tool_call.get("name")
            args = tool_call.get("args", {})
            
            if not tool_name:
                continue
            
            print(f"[JARVIS] Tool call detected: {tool_name}")

            try:
                self.learning_store.log_tool_event(self._active_rec_id, tool_name, "detected")
            except Exception:
                pass

            request_id = str(uuid.uuid4())
            args_redacted = self._redact_tool_args(tool_name, args)
            self._emit_tool_activity({"event": "detected", "id": request_id, "tool": tool_name, "args": args_redacted})

            try:
                self.learning_store.log_tool_event(
                    self._active_rec_id,
                    tool_name,
                    "detected",
                    request_id=request_id,
                    meta={"args": args_redacted},
                )
            except Exception:
                pass
            
            # Check permissions
            if not self.permissions.get(tool_name, True):
                # Needs confirmation
                if self.on_tool_confirmation:
                    future = asyncio.Future()
                    self._pending_confirmations[request_id] = future

                    self._emit_tool_activity({"event": "approval_requested", "id": request_id, "tool": tool_name, "args": args_redacted})

                    try:
                        self.learning_store.log_tool_event(self._active_rec_id, tool_name, "approval_requested", request_id=request_id)
                    except Exception:
                        pass
                    
                    self.on_tool_confirmation({
                        "id": request_id,
                        "tool": tool_name,
                        "args": args
                    })
                    
                    try:
                        confirmed = await asyncio.wait_for(future, timeout=60.0)
                    except asyncio.TimeoutError:
                        confirmed = False
                    finally:
                        self._pending_confirmations.pop(request_id, None)
                    
                    if not confirmed:
                        print(f"[JARVIS] Tool {tool_name} denied by user")
                        self._emit_tool_activity({"event": "denied", "id": request_id, "tool": tool_name})
                        try:
                            self.learning_store.log_tool_event(self._active_rec_id, tool_name, "denied", request_id=request_id, ok=False)
                        except Exception:
                            pass
                        continue

                    self._emit_tool_activity({"event": "approved", "id": request_id, "tool": tool_name})
                    try:
                        self.learning_store.log_tool_event(self._active_rec_id, tool_name, "approved", request_id=request_id, ok=True)
                    except Exception:
                        pass
            
            # Execute tool
            self._emit_tool_activity({"event": "start", "id": request_id, "tool": tool_name, "args": args_redacted})
            try:
                self.learning_store.log_tool_event(self._active_rec_id, tool_name, "start", request_id=request_id)
            except Exception:
                pass
            await self._execute_tool(tool_name, args, request_id=request_id)
    
    async def _execute_tool(self, tool_name: str, args: Dict[str, Any], request_id: Optional[str] = None):
        """Execute a tool call."""
        try:
            result = None
            if tool_name == "generate_cad":
                prompt = args.get("prompt", "")
                result = await self._handle_cad_request(prompt)
            
            elif tool_name == "iterate_cad":
                prompt = args.get("prompt", "")
                result = await self._handle_cad_iterate(prompt)
            
            elif tool_name == "run_web_agent":
                prompt = args.get("prompt", "")
                await self._handle_web_agent(prompt)
                result = "completed"
            
            elif tool_name == "write_file":
                path = args.get("path", "")
                content = args.get("content", "")
                result = await self._handle_write_file(path, content)
            
            elif tool_name == "read_file":
                path = args.get("path", "")
                result = await self._handle_read_file(path)
            
            elif tool_name == "read_directory":
                path = args.get("path", "")
                result = await self._handle_read_directory(path)
            
            elif tool_name == "create_project":
                name = args.get("name", "")
                success, msg = self.project_manager.create_project(name)
                if success:
                    self.project_manager.switch_project(name)
                    if self.on_project_update:
                        self.on_project_update(name)
                result = msg
            
            elif tool_name == "switch_project":
                name = args.get("name", "")
                success, msg = self.project_manager.switch_project(name)
                if success and self.on_project_update:
                    self.on_project_update(name)
                result = msg
            
            elif tool_name == "list_projects":
                projects = self.project_manager.list_projects()
                print(f"[JARVIS] Available projects: {projects}")
                result = projects

            summary = self._summarize_tool_result(tool_name, result)
            if self._should_humor():
                suffixes = [
                    "No smoke, no mirrors.",
                    "All perfectly ordinary. Which is ideal.",
                    "Done. Try not to look too surprised, Sir.",
                ]
                summary = f"{summary} ({random.choice(suffixes)})"

            self._emit_tool_activity({
                "event": "done",
                "id": request_id,
                "tool": tool_name,
                "summary": summary
            })

            try:
                self.learning_store.log_tool_event(
                    self._active_rec_id,
                    tool_name,
                    "done",
                    request_id=request_id,
                    ok=True,
                    meta={"summary": summary},
                )
            except Exception:
                pass

            self._maybe_delight()
                
        except Exception as e:
            print(f"[JARVIS] Tool execution error: {e}")
            traceback.print_exc()
            self._emit_tool_activity({
                "event": "error",
                "id": request_id,
                "tool": tool_name,
                "message": str(e)
            })
            try:
                self.learning_store.log_tool_event(
                    self._active_rec_id,
                    tool_name,
                    "error",
                    request_id=request_id,
                    ok=False,
                    message=str(e),
                )
            except Exception:
                pass

    def record_feedback(self, outcome: str, rec_id: Optional[str] = None, note: Optional[str] = None) -> Dict[str, Any]:
        rid = rec_id or self._last_rec_id or self._active_rec_id
        if not rid:
            return {"ok": False, "error": "no_recommendation"}
        try:
            return self.learning_store.record_feedback(rid, outcome, note=note)
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def get_learning_summary(self) -> Dict[str, Any]:
        try:
            return {
                "patterns": self.learning_store.get_failure_patterns(),
                "weights": {
                    "prompt_variant": self.learning_store.get_weights(prefix="prompt_variant", limit=10),
                    "reasoning_depth": self.learning_store.get_weights(prefix="reasoning_depth", limit=10),
                    "response_timing": self.learning_store.get_weights(prefix="response_timing", limit=10),
                    "tool": self.learning_store.get_weights(prefix="tool", limit=10),
                },
            }
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
    
    async def _handle_cad_request(self, prompt: str):
        """Handle CAD generation request."""
        print(f"[JARVIS] CAD generation: {prompt}")
        
        if self.on_cad_status:
            self.on_cad_status({"status": "generating", "attempt": 1, "max_attempts": 3})
        
        self._ensure_project_context("cad", prompt)
        
        cad_output_dir = str(self.project_manager.get_current_project_path() / "cad")
        cad_data = await self.cad_agent.generate_prototype(prompt, output_dir=cad_output_dir)
        
        if cad_data and self.on_cad_data:
            self.on_cad_data(cad_data)
            self.project_manager.save_cad_artifact(cad_data.get('file_path', 'output.stl'), prompt)
        return cad_data
    
    async def _handle_cad_iterate(self, prompt: str):
        """Handle CAD iteration request."""
        print(f"[JARVIS] CAD iteration: {prompt}")
        
        if self.on_cad_status:
            self.on_cad_status({"status": "generating", "attempt": 1, "max_attempts": 3})
        
        cad_output_dir = str(self.project_manager.get_current_project_path() / "cad")
        cad_data = await self.cad_agent.iterate_prototype(prompt, output_dir=cad_output_dir)
        
        if cad_data and self.on_cad_data:
            self.on_cad_data(cad_data)
            self.project_manager.save_cad_artifact(cad_data.get('file_path', 'output.stl'), f"Iteration: {prompt}")
        return cad_data
    
    async def _handle_web_agent(self, prompt: str):
        """Handle web agent request."""
        print(f"[JARVIS] Web agent: {prompt}")
        
        async def update_frontend(image_b64, log_text):
            if self.on_web_data:
                self.on_web_data({"image": image_b64, "log": log_text})
        
        result = await self.web_agent.run_task(prompt, update_callback=update_frontend)
        print(f"[JARVIS] Web agent result: {result}")
    
    async def _handle_write_file(self, path: str, content: str):
        """Handle file write request."""
        print(f"[JARVIS] Writing file: {path}")
        
        self._ensure_project_context("files", path)
        
        filename = os.path.basename(path)
        current_project_path = self.project_manager.get_current_project_path()
        
        if not os.path.isabs(path):
            final_path = current_project_path / path
        else:
            final_path = current_project_path / filename
        
        try:
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            with open(final_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[JARVIS] File written: {final_path}")
            return {"path": str(final_path)}
        except Exception as e:
            print(f"[JARVIS] Write error: {e}")
            return {"error": str(e)}
    
    async def _handle_read_file(self, path: str):
        """Handle file read request."""
        print(f"[JARVIS] Reading file: {path}")
        
        try:
            if not os.path.exists(path):
                return f"File not found: {path}"
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Feed content back to LLM
            context_msg = f"File content of {path}:\n{content[:4000]}"
            await self.process_text_input(context_msg)

            return f"Read and forwarded: {path}"
            
        except Exception as e:
            print(f"[JARVIS] Read error: {e}")
            return f"Read error: {type(e).__name__}: {e}"
    
    async def _handle_read_directory(self, path: str):
        """Handle directory listing request."""
        print(f"[JARVIS] Reading directory: {path}")
        
        try:
            if not os.path.exists(path):
                return f"Directory not found: {path}"
            
            items = os.listdir(path)
            result = f"Contents of {path}: {', '.join(items)}"
            print(result)
            return items
            
        except Exception as e:
            print(f"[JARVIS] Directory read error: {e}")
            return f"Directory read error: {type(e).__name__}: {e}"
    
    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for TTS (remove JSON, code blocks, etc.)."""
        import re
        clean_text = text
        # Remove JSON objects
        clean_text = re.sub(r'\{[^{}]*\}', '', clean_text)
        # Remove code blocks
        clean_text = re.sub(r'```[\s\S]*?```', '', clean_text)
        # Remove inline code
        clean_text = re.sub(r'`[^`]+`', '', clean_text)
        # Remove markdown formatting
        clean_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_text)
        clean_text = re.sub(r'\*([^*]+)\*', r'\1', clean_text)
        # Remove URLs
        clean_text = re.sub(r'https?://\S+', '', clean_text)
        # Remove extra whitespace
        clean_text = ' '.join(clean_text.split())
        return clean_text.strip()

    def _is_vietnamese_text(self, text: str) -> bool:
        if not text:
            return False
        t = text.lower()
        for ch in "ăâđêôơưáàạảãấầậẩẫắằặẳẵéèẹẻẽếềệểễíìịỉĩóòọỏõốồộổỗớờợởỡúùụủũứừựửữýỳỵỷỹ":
            if ch in t:
                return True
        return False

    def _select_tts_profile(self, text: str) -> tuple[str, str, str]:
        if self.tts_auto_detect and self._is_vietnamese_text(text):
            return (self.tts_voice_vi, self.tts_rate_vi, self.tts_pitch_vi)
        return (self.tts_voice_en, self.tts_rate_en, self.tts_pitch_en)
    
    async def _speak(self, text: str):
        """
        Convert text to speech using streaming sentence-by-sentence synthesis.
        Audio chunks are sent to frontend immediately as they're generated.
        """
        if not text or not text.strip():
            return
        
        clean_text = self._clean_text_for_tts(text)
        if not clean_text:
            return

        voice, rate, pitch = self._select_tts_profile(clean_text)
        try:
            self.tts.set_voice(voice)
            self.tts.set_rate(rate)
            self.tts.set_pitch(pitch)
        except Exception as e:
            print(f"[JARVIS] TTS voice/rate/pitch setup failed: {e}")
        
        self._tts_chunk_count = 0
        start_time = time.time()
        
        async def on_audio_chunk(audio_bytes: bytes, chunk_index: int):
            """Callback for each synthesized audio chunk."""
            if audio_bytes and self.on_audio_data:
                self._tts_chunk_count += 1
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                # Log timing for first chunk (latency metric)
                if chunk_index == 0:
                    elapsed = time.time() - start_time
                    print(f"[JARVIS] TTS: First chunk in {elapsed:.2f}s ({len(audio_bytes)} bytes)")
                
                # Send chunk to frontend with metadata
                self.on_audio_data({
                    "type": "tts_chunk",
                    "audio": audio_b64,
                    "format": "mp3",
                    "chunk_index": chunk_index,
                    "is_first": chunk_index == 0
                })
        
        try:
            print(f"[JARVIS] TTS: Streaming {len(clean_text)} chars...")
            
            # Create streaming buffer
            self._tts_buffer = TTSSentenceBuffer(self.tts, on_audio_chunk)
            
            # Feed text through buffer (automatically chunks into sentences)
            await self._tts_buffer.add_text(clean_text)
            
            # Flush remaining text
            await self._tts_buffer.flush()
            
            # Signal end of speech
            if self.on_audio_data:
                self.on_audio_data({
                    "type": "tts_end",
                    "total_chunks": self._tts_chunk_count
                })
            
            elapsed = time.time() - start_time
            print(f"[JARVIS] TTS: Completed {self._tts_chunk_count} chunks in {elapsed:.2f}s")
            
        except Exception as e:
            print(f"[JARVIS] TTS error: {e}")
            traceback.print_exc()
        finally:
            # Cleanup
            if self._tts_buffer:
                await self._tts_buffer.stop()
                self._tts_buffer = None
    
    async def stop_speaking(self):
        """Stop TTS playback immediately."""
        if self._tts_buffer:
            print("[JARVIS] Stopping TTS...")
            await self._tts_buffer.stop()
            self._tts_buffer = None
            
            # Signal frontend to stop playback
            if self.on_audio_data:
                self.on_audio_data({"type": "tts_stop"})
    
    async def _play_audio_loop(self):
        """Background task to play queued audio."""
        while not self.stop_event.is_set():
            try:
                audio_data = await asyncio.wait_for(
                    self.audio_queue.get(),
                    timeout=0.5
                )
                
                if audio_data and self.on_audio_data:
                    # Convert MP3 to raw audio for visualization
                    self.on_audio_data(audio_data)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"[JARVIS] Audio playback error: {e}")
    
    async def _listen_audio(self):
        """Listen for audio input from microphone."""
        mic_info = self.pya.get_default_input_device_info()
        
        # Resolve input device
        resolved_device_index = None
        
        if self.input_device_name:
            print(f"[JARVIS] Finding input device: {self.input_device_name}")
            for i in range(self.pya.get_device_count()):
                try:
                    info = self.pya.get_device_info_by_index(i)
                    if info['maxInputChannels'] > 0:
                        name = info.get('name', '')
                        if self.input_device_name.lower() in name.lower():
                            resolved_device_index = i
                            print(f"[JARVIS] Found device: {name}")
                            break
                except:
                    continue
        
        if resolved_device_index is None and self.input_device_index is not None:
            try:
                resolved_device_index = int(self.input_device_index)
            except ValueError:
                resolved_device_index = None
        
        if resolved_device_index is None:
            resolved_device_index = mic_info["index"]
        
        try:
            self.audio_stream = await asyncio.to_thread(
                self.pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=SEND_SAMPLE_RATE,
                input=True,
                input_device_index=resolved_device_index,
                frames_per_buffer=CHUNK_SIZE
            )
        except OSError as e:
            print(f"[JARVIS] Failed to open audio input: {e}")
            return
        
        # VAD constants
        VAD_THRESHOLD = 800
        SILENCE_DURATION = 0.5
        
        while not self.stop_event.is_set():
            if self.paused:
                await asyncio.sleep(0.1)
                continue
            
            try:
                data = await asyncio.to_thread(
                    self.audio_stream.read,
                    CHUNK_SIZE,
                    exception_on_overflow=False
                )
                
                # VAD: Calculate RMS
                count = len(data) // 2
                if count > 0:
                    shorts = struct.unpack(f"<{count}h", data)
                    sum_squares = sum(s**2 for s in shorts)
                    rms = int(math.sqrt(sum_squares / count))
                else:
                    rms = 0
                
                if rms > VAD_THRESHOLD:
                    self._silence_start_time = None
                    
                    if not self._is_speaking:
                        self._is_speaking = True
                        print(f"[JARVIS] Speech detected (RMS: {rms})")
                        
                        # Send video frame if available
                        if self._latest_image_payload:
                            pass  # Can be used for vision-capable models
                else:
                    if self._is_speaking:
                        if self._silence_start_time is None:
                            self._silence_start_time = time.time()
                        elif time.time() - self._silence_start_time > SILENCE_DURATION:
                            print("[JARVIS] Silence detected, end of speech")
                            self._is_speaking = False
                            self._silence_start_time = None
                
            except Exception as e:
                print(f"[JARVIS] Audio read error: {e}")
                await asyncio.sleep(0.1)
    
    async def _get_frames(self):
        """Capture video frames from camera."""
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)
        
        while not self.stop_event.is_set():
            if self.paused:
                await asyncio.sleep(0.1)
                continue
            
            ret, frame = await asyncio.to_thread(cap.read)
            if not ret:
                break
            
            # Convert to JPEG for transmission
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(frame_rgb)
            img.thumbnail([640, 480])
            
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=70)
            image_bytes = buffer.getvalue()
            
            self._latest_image_payload = {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(image_bytes).decode()
            }
            
            if self.on_video_frame:
                self.on_video_frame(image_bytes)
            
            await asyncio.sleep(0.1)  # ~10 FPS
        
        cap.release()
    
    async def run(self, start_message: Optional[str] = None):
        """Main run loop."""
        print("[JARVIS] Starting J.A.R.V.I.S (Local LLM Version)...")
        
        # Check LLM availability
        available = await self.llm.check_availability()
        if not available:
            print("[JARVIS] ERROR: Ollama not available or model not found!")
            print("[JARVIS] Please install Ollama and run:")
            print("      ollama pull qwen2.5-coder:7b-instruct")
            if self.on_error:
                self.on_error("Ollama not available. Please install and run 'ollama pull qwen2.5-coder:7b-instruct'")
            return
        
        print("[JARVIS] LLM ready!")
        
        # Initialize STT (optional)
        self.stt_available = self.stt.initialize()
        if self.stt_available:
            print("[JARVIS] STT ready (Vosk)")
        else:
            print("[JARVIS] STT not available - using browser Web Speech API")
        
        # Sync project state
        if self.on_project_update:
            self.on_project_update(self.project_manager.current_project)
        
        # Send start message if provided
        if start_message:
            await self._speak(start_message)
        
        # Start background tasks
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._listen_audio())
                tg.create_task(self._play_audio_loop())
                
                if self.video_mode == "camera":
                    tg.create_task(self._get_frames())
                
                # Wait for stop signal
                await self.stop_event.wait()
                
        except* Exception as e:
            print(f"[JARVIS] Task group error: {e}")
        
        # Cleanup
        if self.audio_stream:
            self.audio_stream.close()
        self.pya.terminate()
        await self.llm.close()
        
        print("[JARVIS] Stopped.")

    # ── Conversation context helpers ──────────────────────────────────────────

    def export_llm_history(self):
        """Return the current LLM conversation history as serialisable dicts."""
        return self.llm.export_history() if self.llm else []

    def import_llm_history(self, history):
        """Restore LLM conversation history from a list of dicts."""
        if self.llm:
            self.llm.import_history(history)

    def reset_llm_context(self):
        """Wipe the LLM conversation history for a fresh context."""
        if self.llm:
            self.llm.reset_conversation()


def get_input_devices():
    """Get list of input audio devices."""
    p = pyaudio.PyAudio()
    devices = []
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get('maxInputChannels', 0) > 0:
            devices.append((i, info.get('name', f'Device {i}')))
    
    p.terminate()
    return devices


def get_output_devices():
    """Get list of output audio devices."""
    p = pyaudio.PyAudio()
    devices = []
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get('maxOutputChannels', 0) > 0:
            devices.append((i, info.get('name', f'Device {i}')))
    
    p.terminate()
    return devices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S - Just A Rather Very Intelligent System")
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        choices=["camera", "screen", "none"],
        help="Video mode"
    )
    args = parser.parse_args()
    
    main = AudioLoop(video_mode=args.mode)
    asyncio.run(main.run(start_message="Good day, Sir. I am Jarvis, your personal assistant. How may I be of service?"))

