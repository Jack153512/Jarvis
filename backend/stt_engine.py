"""
Speech-to-Text Engine using Vosk for offline recognition.

Vosk is a free, open-source speech recognition toolkit that works
completely offline. It's lightweight and supports many languages.

Recommended model: vosk-model-small-en-us (~40MB)
For better accuracy: vosk-model-en-us-0.22 (~1.8GB)

Install: pip install vosk
Download model: https://alphacephei.com/vosk/models
"""

import asyncio
import json
import os
import queue
import struct
import wave
from typing import Optional, Callable, AsyncGenerator
from dataclasses import dataclass
import threading

try:
    from vosk import Model, KaldiRecognizer, SetLogLevel
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    print("[STT] Vosk not installed. Run: pip install vosk")

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("[STT] PyAudio not installed. Run: pip install pyaudio")


@dataclass
class STTConfig:
    model_path: str = "models/vosk-model-small-en-us"
    sample_rate: int = 16000
    chunk_size: int = 4096


class STTEngine:
    """
    Speech-to-Text engine using Vosk for offline recognition.
    """
    
    # URLs for downloading models
    MODEL_URLS = {
        "small": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
        "medium": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",
        "large": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip"
    }
    
    def __init__(self, config: Optional[STTConfig] = None):
        """
        Initialize the STT engine.
        
        Args:
            config: STT configuration
        """
        self.config = config or STTConfig()
        self.model: Optional[Model] = None
        self.recognizer: Optional[KaldiRecognizer] = None
        self._is_listening = False
        self._audio_queue: queue.Queue = queue.Queue()
        self._listen_thread: Optional[threading.Thread] = None
        
        if VOSK_AVAILABLE:
            SetLogLevel(-1)  # Suppress Vosk logs
    
    def _find_model_path(self) -> Optional[str]:
        """Find the Vosk model, checking multiple locations."""
        paths_to_check = [
            self.config.model_path,
            os.path.join(os.path.dirname(__file__), "models", "vosk-model-small-en-us"),
            os.path.join(os.path.dirname(__file__), "..", "models", "vosk-model-small-en-us"),
            os.path.expanduser("~/.vosk/vosk-model-small-en-us"),
            "vosk-model-small-en-us",
        ]
        
        for path in paths_to_check:
            if os.path.exists(path) and os.path.isdir(path):
                return path
        
        return None
    
    def initialize(self) -> bool:
        """
        Initialize the Vosk model.
        
        Returns:
            True if successful, False otherwise
        """
        if not VOSK_AVAILABLE:
            print("[STT] Vosk library not available")
            return False
        
        model_path = self._find_model_path()
        
        if model_path is None:
            print(f"[STT] Model not found. Please download from:")
            print(f"      {self.MODEL_URLS['small']}")
            print(f"      Extract to: {self.config.model_path}")
            return False
        
        try:
            print(f"[STT] Loading model from: {model_path}")
            self.model = Model(model_path)
            self.recognizer = KaldiRecognizer(self.model, self.config.sample_rate)
            self.recognizer.SetWords(True)
            print("[STT] Model loaded successfully")
            return True
        except Exception as e:
            print(f"[STT] Failed to load model: {e}")
            return False
    
    def transcribe_audio(self, audio_data: bytes) -> str:
        """
        Transcribe audio data.
        
        Args:
            audio_data: Raw PCM audio bytes (16-bit, mono, 16kHz)
            
        Returns:
            Transcribed text
        """
        if self.recognizer is None:
            if not self.initialize():
                return ""
        
        # Process audio
        if self.recognizer.AcceptWaveform(audio_data):
            result = json.loads(self.recognizer.Result())
            return result.get("text", "")
        else:
            # Partial result
            partial = json.loads(self.recognizer.PartialResult())
            return partial.get("partial", "")
    
    def transcribe_file(self, audio_path: str) -> str:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to audio file (WAV format recommended)
            
        Returns:
            Transcribed text
        """
        if self.recognizer is None:
            if not self.initialize():
                return ""
        
        try:
            with wave.open(audio_path, "rb") as wf:
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                    print("[STT] Audio must be mono 16-bit WAV")
                    return ""
                
                # Create a new recognizer for this file's sample rate
                rec = KaldiRecognizer(self.model, wf.getframerate())
                rec.SetWords(True)
                
                results = []
                while True:
                    data = wf.readframes(4096)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        if result.get("text"):
                            results.append(result["text"])
                
                # Get final result
                final = json.loads(rec.FinalResult())
                if final.get("text"):
                    results.append(final["text"])
                
                return " ".join(results)
                
        except Exception as e:
            print(f"[STT] Error transcribing file: {e}")
            return ""
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for streaming audio."""
        self._audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    async def start_listening(
        self, 
        on_text: Callable[[str], None],
        on_partial: Optional[Callable[[str], None]] = None,
        device_index: Optional[int] = None
    ):
        """
        Start listening for speech in the background.
        
        Args:
            on_text: Callback for final transcription
            on_partial: Optional callback for partial results
            device_index: Microphone device index
        """
        if not PYAUDIO_AVAILABLE:
            print("[STT] PyAudio not available for live recognition")
            return
        
        if self.recognizer is None:
            if not self.initialize():
                return
        
        self._is_listening = True
        
        # Start audio stream in separate thread
        def audio_thread():
            pa = pyaudio.PyAudio()
            
            try:
                stream = pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.config.sample_rate,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=self.config.chunk_size,
                    stream_callback=self._audio_callback
                )
                
                stream.start_stream()
                
                while self._is_listening:
                    threading._sleep(0.1)
                
                stream.stop_stream()
                stream.close()
                
            except Exception as e:
                print(f"[STT] Audio stream error: {e}")
            finally:
                pa.terminate()
        
        self._listen_thread = threading.Thread(target=audio_thread, daemon=True)
        self._listen_thread.start()
        
        # Process audio in async loop
        while self._is_listening:
            try:
                # Get audio data with timeout
                try:
                    data = self._audio_queue.get(timeout=0.1)
                except queue.Empty:
                    await asyncio.sleep(0.05)
                    continue
                
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "").strip()
                    if text and on_text:
                        on_text(text)
                else:
                    partial = json.loads(self.recognizer.PartialResult())
                    text = partial.get("partial", "").strip()
                    if text and on_partial:
                        on_partial(text)
                
                await asyncio.sleep(0.01)
                
            except Exception as e:
                print(f"[STT] Processing error: {e}")
                await asyncio.sleep(0.1)
    
    def stop_listening(self):
        """Stop the listening loop."""
        self._is_listening = False
        if self._listen_thread:
            self._listen_thread.join(timeout=1.0)
            self._listen_thread = None
        
        # Clear the queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
    
    def reset(self):
        """Reset the recognizer state."""
        if self.recognizer and self.model:
            self.recognizer = KaldiRecognizer(self.model, self.config.sample_rate)
            self.recognizer.SetWords(True)


class WebSTT:
    """
    Alternative STT using browser's Web Speech API.
    This class provides a bridge for using browser-based speech recognition
    when Vosk is not available.
    """
    
    def __init__(self):
        self.is_listening = False
    
    def get_frontend_code(self) -> str:
        """
        Get JavaScript code for Web Speech API integration.
        To be injected in the frontend.
        """
        return """
        class WebSpeechRecognition {
            constructor(socket) {
                this.socket = socket;
                this.recognition = null;
                this.isListening = false;
                
                if ('webkitSpeechRecognition' in window) {
                    this.recognition = new webkitSpeechRecognition();
                } else if ('SpeechRecognition' in window) {
                    this.recognition = new SpeechRecognition();
                } else {
                    console.error('Web Speech API not supported');
                    return;
                }
                
                this.recognition.continuous = true;
                this.recognition.interimResults = true;
                this.recognition.lang = 'en-US';
                
                this.recognition.onresult = (event) => {
                    let finalTranscript = '';
                    let interimTranscript = '';
                    
                    for (let i = event.resultIndex; i < event.results.length; i++) {
                        const transcript = event.results[i][0].transcript;
                        if (event.results[i].isFinal) {
                            finalTranscript += transcript;
                        } else {
                            interimTranscript += transcript;
                        }
                    }
                    
                    if (finalTranscript) {
                        this.socket.emit('stt_result', { text: finalTranscript, final: true });
                    } else if (interimTranscript) {
                        this.socket.emit('stt_result', { text: interimTranscript, final: false });
                    }
                };
                
                this.recognition.onerror = (event) => {
                    console.error('Speech recognition error:', event.error);
                    if (event.error !== 'no-speech') {
                        this.socket.emit('stt_error', { error: event.error });
                    }
                };
                
                this.recognition.onend = () => {
                    if (this.isListening) {
                        // Restart if still supposed to be listening
                        this.recognition.start();
                    }
                };
            }
            
            start() {
                if (this.recognition && !this.isListening) {
                    this.isListening = true;
                    this.recognition.start();
                }
            }
            
            stop() {
                if (this.recognition && this.isListening) {
                    this.isListening = false;
                    this.recognition.stop();
                }
            }
        }
        """


# Singleton instance
_default_stt: Optional[STTEngine] = None


def get_stt(config: Optional[STTConfig] = None) -> STTEngine:
    """Get the default STTEngine instance."""
    global _default_stt
    if _default_stt is None:
        _default_stt = STTEngine(config)
    return _default_stt


async def test_stt():
    """Test the STT engine."""
    print("[Test] Initializing STT engine...")
    stt = STTEngine()
    
    if not stt.initialize():
        print("[Test] STT initialization failed - model not found")
        print("[Test] Download from: https://alphacephei.com/vosk/models")
        return False
    
    print("[Test] STT engine ready!")
    print("[Test] To test live recognition, call start_listening()")
    
    return True


if __name__ == "__main__":
    asyncio.run(test_stt())

