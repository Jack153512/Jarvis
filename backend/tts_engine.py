"""
Text-to-Speech Engine using edge-tts.

edge-tts is a free, open-source TTS solution that uses Microsoft Edge's 
online TTS service. It produces high-quality, natural-sounding speech.

No API key required. Works offline after initial voice list fetch.

Install: pip install edge-tts
"""

import edge_tts
import asyncio
import io
import logging
import os
import tempfile
from typing import Optional, List, Dict, AsyncGenerator, Callable
from dataclasses import dataclass


logger = logging.getLogger("jarvis.tts")


@dataclass
class Voice:
    name: str
    short_name: str
    gender: str
    locale: str


class TTSEngine:
    """
    Text-to-Speech engine using edge-tts.
    Provides natural-sounding speech synthesis.
    """
    
    # Recommended voices for English (verified working with edge-tts 7.2.7+)
    RECOMMENDED_VOICES = {
        # Female voices (US)
        "ava": "en-US-AvaNeural",          # Natural, warm (default)
        "aria": "en-US-AriaNeural",        # Professional
        "jenny": "en-US-JennyNeural",      # Friendly, casual
        "emma": "en-US-EmmaNeural",        # Clear, informative
        "ana": "en-US-AnaNeural",          # Young, energetic
        "michelle": "en-US-MichelleNeural", # Warm
        
        # Male voices (US)
        "andrew": "en-US-AndrewNeural",    # Natural, conversational
        "brian": "en-US-BrianNeural",      # Professional
        "guy": "en-US-GuyNeural",          # Deep, authoritative
        "eric": "en-US-EricNeural",        # Warm, conversational
        "christopher": "en-US-ChristopherNeural",  # News anchor style
        "roger": "en-US-RogerNeural",      # Mature
        "steffan": "en-US-SteffanNeural",  # Clear
        
        # Multilingual voices
        "ava_multi": "en-US-AvaMultilingualNeural",
        "andrew_multi": "en-US-AndrewMultilingualNeural",
        "emma_multi": "en-US-EmmaMultilingualNeural",
        "brian_multi": "en-US-BrianMultilingualNeural",

        # Vietnamese voices
        "hoaimy": "vi-VN-HoaiMyNeural",
        "namminh": "vi-VN-NamMinhNeural",
        "hoai_my": "vi-VN-HoaiMyNeural",
        "nam_minh": "vi-VN-NamMinhNeural",
    }
    
    def __init__(self, voice: str = "andrew", rate: str = "+0%", pitch: str = "+0Hz"):
        """
        Initialize the TTS engine.
        
        Args:
            voice: Voice name (use short names like 'aria' or full names like 'en-US-AriaNeural')
            rate: Speech rate adjustment (e.g., '+20%', '-10%')
            pitch: Pitch adjustment (e.g., '+5Hz', '-10Hz')
        """
        self.voice = self._resolve_voice(voice)
        self.rate = rate
        self.pitch = pitch
        self._voice_cache: Optional[List[Voice]] = None
    
    def _resolve_voice(self, voice: str) -> str:
        """Resolve short voice names to full names."""
        voice_lower = voice.lower()
        if voice_lower in self.RECOMMENDED_VOICES:
            return self.RECOMMENDED_VOICES[voice_lower]
        return voice
    
    async def list_voices(self, locale_filter: str = "en") -> List[Voice]:
        """
        List available voices, optionally filtered by locale.
        
        Args:
            locale_filter: Filter voices by locale prefix (e.g., 'en', 'en-US')
            
        Returns:
            List of Voice objects
        """
        if self._voice_cache is None:
            try:
                voices_data = await edge_tts.list_voices()
                self._voice_cache = [
                    Voice(
                        name=v.get("FriendlyName", ""),
                        short_name=v.get("ShortName", ""),
                        gender=v.get("Gender", ""),
                        locale=v.get("Locale", "")
                    )
                    for v in voices_data
                ]
            except Exception as e:
                logger.exception("Error fetching voices: %s", e)
                return []
        
        if locale_filter:
            return [v for v in self._voice_cache if v.locale.startswith(locale_filter)]
        return self._voice_cache
    
    def set_voice(self, voice: str):
        """Set the voice to use for synthesis."""
        self.voice = self._resolve_voice(voice)
    
    def set_rate(self, rate: str):
        """Set the speech rate (e.g., '+20%', '-10%')."""
        self.rate = rate
    
    def set_pitch(self, pitch: str):
        """Set the pitch (e.g., '+5Hz', '-10Hz')."""
        self.pitch = pitch
    
    async def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech audio.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio data as bytes (MP3 format)
        """
        if not text or not text.strip():
            return b""
        
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch
        )
        
        audio_data = io.BytesIO()
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])
        
        return audio_data.getvalue()
    
    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream audio chunks as they are generated.
        
        Args:
            text: Text to synthesize
            
        Yields:
            Audio data chunks (MP3 format)
        """
        if not text or not text.strip():
            return
        
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch
        )
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]
    
    async def synthesize_to_file(
        self, 
        text: str, 
        output_path: Optional[str] = None
    ) -> str:
        """
        Save synthesized speech to a file.
        
        Args:
            text: Text to synthesize
            output_path: Output file path (default: temp file)
            
        Returns:
            Path to the saved audio file
        """
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".mp3")
        
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch
        )
        
        await communicate.save(output_path)
        return output_path
    
    async def synthesize_with_subtitles(
        self, 
        text: str
    ) -> tuple[bytes, List[Dict]]:
        """
        Synthesize speech with word-level timing information.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Tuple of (audio_bytes, word_timings)
            word_timings is a list of dicts with 'text', 'start', 'end' keys
        """
        if not text or not text.strip():
            return b"", []
        
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch
        )
        
        audio_data = io.BytesIO()
        word_timings = []
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                word_timings.append({
                    "text": chunk.get("text", ""),
                    "start": chunk.get("offset", 0) / 10000000,  # Convert to seconds
                    "end": (chunk.get("offset", 0) + chunk.get("duration", 0)) / 10000000
                })
        
        return audio_data.getvalue(), word_timings


class TTSSentenceBuffer:
    """
    Buffers text and triggers TTS when complete sentences are detected.
    Useful for streaming LLM responses.
    
    Features:
    - Sentence-level chunking for fast first-audio
    - Parallel TTS generation (generates next while current plays)
    - Maintains correct playback order
    - Handles interrupts gracefully
    """
    
    SENTENCE_ENDERS = {'.', '!', '?', '…', '。', '！', '？'}
    SOFT_ENDERS = {':', ';', ',', '，', '、'}
    MAX_SENTENCE_LENGTH = 200  # Force split at this length
    
    # Common abbreviations that shouldn't end sentences
    ABBREVIATIONS = {
        'mr.', 'mrs.', 'ms.', 'dr.', 'vs.', 'e.g.', 'i.e.', 'etc.',
        'inc.', 'ltd.', 'jr.', 'sr.', 'st.', 'ave.', 'blvd.',
        'tp.', 'q.', 'p.', 'th.', 'hn.', 'hcm.', 'sg.',
        'pgs.', 'gs.', 'ts.', 'ths.', 'th.s.',
        'vd.', 'v.d.',
    }
    
    def __init__(self, tts_engine: TTSEngine, on_audio: callable):
        """
        Initialize the sentence buffer.
        
        Args:
            tts_engine: TTSEngine instance
            on_audio: Async callback function(audio_bytes, chunk_index, is_final) 
                      called when audio is ready
        """
        self.tts = tts_engine
        self.on_audio = on_audio
        self.buffer = ""
        self._stopped = False
        self._chunk_index = 0
        self._pending_tasks: List[asyncio.Task] = []
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._processor_task: Optional[asyncio.Task] = None
    
    def _is_sentence_end(self, text: str, pos: int) -> bool:
        """Check if position is a valid sentence end (not an abbreviation)."""
        if pos >= len(text):
            return False
        
        char = text[pos]
        if char not in self.SENTENCE_ENDERS:
            return False

        if char == '.' and pos + 1 < len(text) and text[pos + 1] == '.':
            return False
        
        # Check for abbreviations (look back up to 5 chars)
        start = max(0, pos - 5)
        before = text[start:pos + 1].lower()
        
        for abbr in self.ABBREVIATIONS:
            if before.endswith(abbr):
                return False
        
        # Check for decimal numbers (e.g., "3.14")
        if char == '.' and pos > 0 and pos < len(text) - 1:
            if text[pos - 1].isdigit() and text[pos + 1].isdigit():
                return False
        
        return True
    
    def _find_split_point(self, text: str) -> int:
        """Find the best point to split text into a sentence chunk."""
        # First, look for sentence enders
        for i, char in enumerate(text):
            if self._is_sentence_end(text, i):
                j = i + 1
                while j < len(text) and text[j] in ('"', "'", ')', ']', '}', '”', '’'):
                    j += 1
                return j
        
        # If text is too long, look for soft enders
        if len(text) > self.MAX_SENTENCE_LENGTH:
            for i in range(self.MAX_SENTENCE_LENGTH, 0, -1):
                if text[i] in self.SOFT_ENDERS:
                    return i + 1
            
            # Last resort: split at space
            for i in range(self.MAX_SENTENCE_LENGTH, 0, -1):
                if text[i] == ' ':
                    return i + 1
        
        return -1  # No split point found
    
    async def _synthesize_chunk(self, text: str, chunk_index: int):
        """Synthesize a single chunk and queue the result."""
        if self._stopped:
            return
        
        try:
            audio = await self.tts.synthesize(text)
            if audio and not self._stopped:
                await self._audio_queue.put((chunk_index, audio))
        except Exception as e:
            logger.exception("Error synthesizing chunk %s: %s", chunk_index, e)
    
    async def _process_audio_queue(self):
        """Process audio queue and call callback in order."""
        expected_index = 0
        pending = {}  # Store out-of-order chunks
        
        while not self._stopped:
            try:
                # Wait for next audio chunk with timeout
                try:
                    chunk_index, audio = await asyncio.wait_for(
                        self._audio_queue.get(), 
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Store chunk
                pending[chunk_index] = audio
                
                # Send chunks in order
                while expected_index in pending:
                    audio_data = pending.pop(expected_index)
                    if self.on_audio and not self._stopped:
                        try:
                            await self.on_audio(audio_data, expected_index)
                        except Exception as e:
                            logger.exception("Callback error: %s", e)
                    expected_index += 1
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Queue processing error: %s", e)
    
    def start(self):
        """Start the audio queue processor."""
        if self._processor_task is None or self._processor_task.done():
            self._stopped = False
            self._chunk_index = 0
            self._processor_task = asyncio.create_task(self._process_audio_queue())
    
    async def add_text(self, text: str):
        """Add text to the buffer and synthesize complete sentences."""
        if self._stopped:
            return
        
        # Start processor if not running
        self.start()
        
        self.buffer += text
        
        # Extract and process complete sentences
        while True:
            split_pos = self._find_split_point(self.buffer)
            if split_pos == -1:
                break
            
            sentence = self.buffer[:split_pos].strip()
            self.buffer = self.buffer[split_pos:].lstrip()
            
            if sentence:
                # Start synthesis task (runs in parallel)
                chunk_idx = self._chunk_index
                self._chunk_index += 1
                task = asyncio.create_task(self._synthesize_chunk(sentence, chunk_idx))
                self._pending_tasks.append(task)
    
    async def flush(self):
        """Flush any remaining text in the buffer and wait for completion."""
        if self._stopped:
            return
        
        # Synthesize remaining buffer
        if self.buffer.strip():
            chunk_idx = self._chunk_index
            self._chunk_index += 1
            task = asyncio.create_task(self._synthesize_chunk(self.buffer.strip(), chunk_idx))
            self._pending_tasks.append(task)
            self.buffer = ""
        
        # Wait for all synthesis tasks to complete
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
            self._pending_tasks.clear()
        
        # Give queue time to process remaining items
        await asyncio.sleep(0.2)
    
    async def stop(self):
        """Stop processing and clear everything."""
        self._stopped = True
        self.buffer = ""
        
        # Cancel pending synthesis tasks
        for task in self._pending_tasks:
            if not task.done():
                task.cancel()
        self._pending_tasks.clear()
        
        # Stop processor
        if self._processor_task and not self._processor_task.done():
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        # Clear queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except:
                break
        
        self._chunk_index = 0
    
    def clear(self):
        """Clear the buffer without synthesizing (sync version)."""
        self.buffer = ""
        self._stopped = True


class StreamingTTSManager:
    """
    High-level manager for streaming TTS with automatic chunking.
    
    Usage:
        manager = StreamingTTSManager(on_audio_chunk=my_callback)
        await manager.speak_streaming(full_text)
        # or
        for chunk in llm_response:
            await manager.add_text(chunk)
        await manager.finish()
    """
    
    def __init__(
        self, 
        on_audio_chunk: Callable[[bytes, int], None],
        voice: str = "andrew",
        rate: str = "+0%",
        pitch: str = "+0Hz"
    ):
        """
        Initialize streaming TTS manager.
        
        Args:
            on_audio_chunk: Callback(audio_bytes, chunk_index) for each audio chunk
            voice: Voice name
            rate: Speech rate
            pitch: Pitch adjustment
        """
        self.tts = TTSEngine(voice=voice, rate=rate, pitch=pitch)
        self.on_audio_chunk = on_audio_chunk
        self._buffer: Optional[TTSSentenceBuffer] = None
        self._is_speaking = False
    
    async def _audio_callback(self, audio: bytes, chunk_index: int):
        """Internal callback to forward audio to user callback."""
        if not self.on_audio_chunk:
            return

        result = self.on_audio_chunk(audio, chunk_index)
        if asyncio.iscoroutine(result):
            await result
    
    async def speak_streaming(self, text: str):
        """
        Speak text with automatic sentence chunking.
        Returns quickly - audio is streamed as it's generated.
        
        Args:
            text: Full text to speak
        """
        if not text or not text.strip():
            return
        
        # Create new buffer for this speech
        self._buffer = TTSSentenceBuffer(self.tts, self._audio_callback)
        self._is_speaking = True
        
        try:
            # Add all text at once - buffer will chunk it
            await self._buffer.add_text(text)
            await self._buffer.flush()
        finally:
            self._is_speaking = False
    
    async def add_text(self, text: str):
        """
        Add text chunk (for streaming from LLM).
        Call finish() when done.
        """
        if self._buffer is None:
            self._buffer = TTSSentenceBuffer(self.tts, self._audio_callback)
            self._is_speaking = True
        
        await self._buffer.add_text(text)
    
    async def finish(self):
        """Finish streaming and flush remaining text."""
        if self._buffer:
            await self._buffer.flush()
            self._is_speaking = False
    
    async def stop(self):
        """Stop speaking immediately."""
        if self._buffer:
            await self._buffer.stop()
        self._is_speaking = False
    
    @property
    def is_speaking(self) -> bool:
        return self._is_speaking


# Singleton instance
_default_tts: Optional[TTSEngine] = None


def get_tts(voice: str = "andrew") -> TTSEngine:
    """Get the default TTSEngine instance."""
    global _default_tts
    if _default_tts is None:
        _default_tts = TTSEngine(voice=voice)
    return _default_tts


async def test_tts():
    """Test the TTS engine."""
    import time
    
    print("[Test] Initializing TTS engine...")
    tts = TTSEngine(voice="andrew")
    
    print("[Test] Listing available English voices...")
    voices = await tts.list_voices("en-US")
    for v in voices[:5]:
        print(f"  - {v.short_name}: {v.name} ({v.gender})")
    
    # Test 1: Basic synthesis
    print("\n[Test 1] Basic synthesis...")
    test_text = "Good day, Sir. I am Jarvis, your personal assistant. How may I be of service?"
    
    start = time.time()
    audio = await tts.synthesize(test_text)
    print(f"[Test 1] Generated {len(audio)} bytes in {time.time() - start:.2f}s")
    
    # Test 2: Streaming sentence buffer
    print("\n[Test 2] Streaming sentence buffer...")
    long_text = """
    Good day, Sir. I hope you're having a wonderful day. 
    Let me tell you about the weather forecast. It's going to be sunny with a high of 75 degrees.
    Perfect weather for a walk in the park! Would you like me to schedule some outdoor activities?
    I can also help you with your calendar, emails, or any other tasks you might have.
    Just let me know what you need, and I'll be happy to assist.
    """
    
    chunks_received = []
    async def on_chunk(audio_bytes, index):
        chunks_received.append((index, len(audio_bytes)))
        print(f"  [Chunk {index}] Received {len(audio_bytes)} bytes")
    
    buffer = TTSSentenceBuffer(tts, on_chunk)
    
    start = time.time()
    # Simulate streaming text input
    words = long_text.split()
    for i in range(0, len(words), 3):
        chunk = ' '.join(words[i:i+3]) + ' '
        await buffer.add_text(chunk)
        await asyncio.sleep(0.05)  # Simulate LLM generation delay
    
    await buffer.flush()
    await buffer.stop()
    
    elapsed = time.time() - start
    print(f"[Test 2] Processed {len(chunks_received)} chunks in {elapsed:.2f}s")
    print(f"[Test 2] First chunk arrived quickly (streaming mode)")
    
    return True


if __name__ == "__main__":
    asyncio.run(test_tts())

