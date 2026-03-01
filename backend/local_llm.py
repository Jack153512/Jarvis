"""
Local LLM Integration Module using Ollama.

Recommended Models (in order of preference for this project):
1. deepseek-r1:14b - Best for CAD (97.3% MATH, superior reasoning) - RECOMMENDED
2. qwen2.5-coder:14b-instruct - Fallback (88.4% HumanEval)
3. qwen2.5-coder:7b-instruct - For limited VRAM

To install Ollama and the recommended model:
    1. Install Ollama: https://ollama.ai/download
    2. Run: ollama pull deepseek-r1:14b
"""

import aiohttp
import asyncio
import json
import re
import sys
from typing import AsyncGenerator, Optional, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Message:
    role: str  # "system", "user", "assistant"
    content: str


@dataclass 
class LLMConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "deepseek-r1:14b"  # 97.3% MATH, best reasoning for CAD
    temperature: float = 0.2  # Lower for deterministic geometric output
    context_length: int = 32768  # DeepSeek-R1 supports 32K context
    timeout: float = 240.0  # More time for reasoning model


class LocalLLM:
    """
    Async client for Ollama-based local LLM.
    Optimized for code generation and CAD workflows.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        if self.config.base_url and "localhost" in self.config.base_url:
            self.config.base_url = self.config.base_url.replace("localhost", "127.0.0.1")
        self.conversation_history: List[Message] = []
        self.system_prompt: Optional[str] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._warmup_lock = asyncio.Lock()
        self._warmed_up = False
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def set_system_prompt(self, prompt: str):
        """Set the system prompt for the conversation."""
        self.system_prompt = prompt
    
    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []

    def export_history(self) -> List[Dict[str, str]]:
        """Serialize conversation history to plain dicts for storage."""
        return [{"role": m.role, "content": m.content} for m in self.conversation_history]

    def import_history(self, history: List[Dict[str, str]]) -> None:
        """Restore conversation history from serialized dicts."""
        self.conversation_history = [
            Message(role=str(m.get("role", "user")), content=str(m.get("content", "")))
            for m in history
            if isinstance(m, dict)
        ]

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.conversation_history.append(Message(role=role, content=content))
    
    def _normalize_chunk(self, chunk: str, buffer: dict) -> str:
        """
        Normalize LLM output chunk by handling model-specific formats.
        
        DeepSeek-R1 outputs reasoning in <think>...</think> blocks before
        the actual response. This method strips those blocks for clean output.
        
        Args:
            chunk: Raw chunk from LLM
            buffer: Mutable dict to track state across chunks
            
        Returns:
            Normalized chunk with think blocks removed
        """
        if not chunk:
            return ""
        
        # Add chunk to buffer for processing
        buffer['text'] = buffer.get('text', '') + chunk
        text = buffer['text']
        
        # Check if we're inside a <think> block
        if buffer.get('in_think', False):
            # Look for closing </think>
            end_pos = text.find('</think>')
            if end_pos != -1:
                # Found end of think block - skip everything up to and including </think>
                buffer['text'] = text[end_pos + 8:]  # 8 = len('</think>')
                buffer['in_think'] = False
                # Return any content after the think block
                return buffer['text']
            else:
                # Still inside think block, consume everything
                buffer['text'] = ''
                return ''
        else:
            # Check for start of <think> block
            start_pos = text.find('<think>')
            if start_pos != -1:
                # Found start of think block
                before_think = text[:start_pos]
                buffer['text'] = text[start_pos + 7:]  # 7 = len('<think>')
                buffer['in_think'] = True
                
                # Check if there's also an end tag in the remaining buffer
                end_pos = buffer['text'].find('</think>')
                if end_pos != -1:
                    buffer['text'] = buffer['text'][end_pos + 8:]
                    buffer['in_think'] = False
                else:
                    buffer['text'] = ''
                
                return before_think
            else:
                # No think tags, return as-is but keep last few chars in buffer
                # in case a tag spans chunks
                if len(text) > 10:
                    output = text[:-10]
                    buffer['text'] = text[-10:]
                    return output
                return ''
    
    def _flush_buffer(self, buffer: dict) -> str:
        """Flush any remaining content in the normalization buffer."""
        text = buffer.get('text', '')
        buffer['text'] = ''
        buffer['in_think'] = False
        return text
    
    def normalize_full_response(self, response: str) -> str:
        """
        Remove all <think>...</think> blocks from a complete response.
        
        Args:
            response: Full LLM response text
            
        Returns:
            Response with all think blocks removed
        """
        # Remove all <think>...</think> blocks (including multi-line)
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        return cleaned.strip()
    
    async def check_availability(self) -> bool:
        """Check if Ollama server is running and model is available."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.config.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    # Check if our model is available (with or without tag)
                    model_base = self.config.model.split(":")[0]
                    return any(model_base in m for m in models)
                return False
        except Exception as e:
            print(f"[LocalLLM] Availability check failed: {e}")
            return False
    
    async def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.config.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return [m.get("name", "") for m in data.get("models", [])]
                return []
        except Exception:
            return []
    
    def _build_messages(self, user_message: str) -> List[Dict[str, str]]:
        """Build the messages array for the API call."""
        messages = []
        
        # Add system prompt if set
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Add conversation history
        for msg in self.conversation_history:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    async def chat(
        self, 
        message: str, 
        stream: bool = True,
        normalize: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Send a message and stream the response.
        
        Args:
            message: User message to send
            stream: Whether to stream the response (default True)
            normalize: Whether to strip <think> blocks (default True, for DeepSeek-R1)
            
        Yields:
            Response text chunks (normalized if enabled)
        """
        url = f"{self.config.base_url}/api/chat"
        messages = self._build_messages(message)
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.config.temperature,
                "num_ctx": self.config.context_length
            }
        }
        
        session = await self._get_session()
        full_response = ""
        raw_response = ""  # Keep raw for history
        normalize_buffer = {}  # Buffer for streaming normalization
        
        last_client_error: Optional[Exception] = None

        for attempt in range(2):
            try:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error ({response.status}): {error_text}")

                    async for line in response.content:
                        if not line:
                            continue

                        try:
                            data = json.loads(line.decode('utf-8'))

                            if "error" in data:
                                raise Exception(f"Ollama error: {data['error']}")

                            if "message" in data:
                                chunk = data["message"].get("content", "")
                                if chunk:
                                    raw_response += chunk

                                    if normalize:
                                        normalized_chunk = self._normalize_chunk(chunk, normalize_buffer)
                                        if normalized_chunk:
                                            full_response += normalized_chunk
                                            yield normalized_chunk
                                    else:
                                        full_response += chunk
                                        yield chunk

                            if data.get("done", False):
                                break

                        except json.JSONDecodeError:
                            continue

                    if normalize:
                        remaining = self._flush_buffer(normalize_buffer)
                        if remaining:
                            full_response += remaining
                            yield remaining

                    self.add_message("user", message)
                    self.add_message("assistant", full_response)
                    return

            except asyncio.TimeoutError:
                raise Exception("LLM request timed out. Try a shorter prompt or increase timeout.")
            except (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError, aiohttp.ClientConnectionError) as e:
                last_client_error = e
                if attempt == 0:
                    await asyncio.sleep(0.5)
                    continue
                raise Exception(f"Connection error: {e}. Is Ollama running?")
            except aiohttp.ClientError as e:
                last_client_error = e
                raise Exception(f"Connection error: {e}. Is Ollama running?")

        if last_client_error:
            raise Exception(f"Connection error: {last_client_error}. Is Ollama running?")

    async def warmup(self) -> bool:
        """Warm up the Ollama model to reduce first-request latency.

        This sends a tiny /api/chat request WITHOUT mutating conversation_history,
        so normal chat behavior remains unchanged.

        Returns:
            True if warmup succeeded, False otherwise.
        """
        if self._warmed_up:
            return True

        async with self._warmup_lock:
            if self._warmed_up:
                return True

            try:
                session = await self._get_session()
                url = f"{self.config.base_url}/api/chat"
                payload = {
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": False,
                    "options": {
                        "temperature": 0,
                        "num_ctx": min(int(self.config.context_length or 2048), 2048),
                    },
                }
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        self._warmed_up = False
                        return False
                    # Consume json body to complete the request.
                    await response.json()

                self._warmed_up = True
                return True
            except Exception:
                self._warmed_up = False
                return False
    
    async def chat_complete(self, message: str) -> str:
        """
        Send a message and get the complete response (non-streaming).
        
        Args:
            message: User message to send
            
        Returns:
            Complete response text
        """
        full_response = ""
        async for chunk in self.chat(message, stream=True):
            full_response += chunk
        return full_response
    
    async def generate_code(
        self, 
        prompt: str, 
        language: str = "python",
        context: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate code based on a prompt.
        
        Args:
            prompt: Description of what code to generate
            language: Programming language (default: python)
            context: Optional existing code context
            
        Yields:
            Code chunks
        """
        code_prompt = f"""Generate {language} code for the following request.
Return ONLY the code without any explanation or markdown formatting.
Do not include ```python or ``` markers.

Request: {prompt}"""
        
        if context:
            code_prompt += f"\n\nExisting code context:\n{context}"
        
        async for chunk in self.chat(code_prompt):
            yield chunk
    
    async def generate_cad_code(
        self, 
        description: str,
        existing_code: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate build123d CAD code.
        
        Args:
            description: Description of the 3D model to create
            existing_code: Optional existing code to iterate on
            
        Yields:
            Code chunks
        """
        cad_system_prompt = """You are a Python CAD expert using the build123d library.
Generate clean, working Python code that creates 3D models.

Requirements:
1. Always start with: from build123d import *
2. Use BuildPart() context for solid models
3. Assign final result to: result_part = p.part
4. Export with: export_stl(result_part, 'output.stl')
5. Use conservative fillet/chamfer values (0.5-2mm)
6. Center models at origin (0,0,0)
7. Use millimeters for dimensions

Return ONLY the Python code, no explanations."""

        # Temporarily set CAD-specific system prompt
        original_system = self.system_prompt
        self.set_system_prompt(cad_system_prompt)
        
        try:
            if existing_code:
                prompt = f"""Modify this build123d code based on the request.

Current code:
{existing_code}

Modification request: {description}

Return the complete modified code."""
            else:
                prompt = f"Create a 3D model of: {description}"
            
            async for chunk in self.chat(prompt):
                yield chunk
        finally:
            # Restore original system prompt
            self.system_prompt = original_system
    
    async def extract_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from LLM response.
        Uses a simple JSON parsing approach for tool use.
        
        Args:
            response: LLM response text
            
        Returns:
            List of tool call dictionaries
        """
        tool_calls = []
        
        # Look for JSON tool calls in the response
        try:
            # Try to find JSON objects in the response
            import re
            json_pattern = r'\{[^{}]*"tool"[^{}]*\}'
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            for match in matches:
                try:
                    tool_call = json.loads(match)
                    if "tool" in tool_call or "name" in tool_call:
                        tool_calls.append(tool_call)
                except json.JSONDecodeError:
                    continue
                    
        except Exception:
            pass
        
        return tool_calls


class ConversationManager:
    """
    Manages multiple conversations with the local LLM.
    Useful for separating contexts (main chat, CAD agent, web agent).
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.conversations: Dict[str, LocalLLM] = {}
    
    def get_conversation(self, name: str) -> LocalLLM:
        """Get or create a named conversation."""
        if name not in self.conversations:
            self.conversations[name] = LocalLLM(self.config)
        return self.conversations[name]
    
    def reset_conversation(self, name: str):
        """Reset a specific conversation."""
        if name in self.conversations:
            self.conversations[name].reset_conversation()
    
    def reset_all(self):
        """Reset all conversations."""
        for conv in self.conversations.values():
            conv.reset_conversation()
    
    async def close_all(self):
        """Close all HTTP sessions."""
        for conv in self.conversations.values():
            await conv.close()


# Singleton instance for easy access
_default_llm: Optional[LocalLLM] = None


def get_llm(config: Optional[LLMConfig] = None) -> LocalLLM:
    """Get the default LocalLLM instance."""
    global _default_llm
    if _default_llm is None:
        _default_llm = LocalLLM(config)
    return _default_llm


async def test_connection():
    """Test the Ollama connection."""
    llm = LocalLLM()
    
    print("[Test] Checking Ollama availability...")
    available = await llm.check_availability()
    
    if not available:
        print("[Test] Ollama not available or model not found.")
        print("[Test] Available models:", await llm.list_models())
        return False
    
    print("[Test] Ollama is available!")
    print("[Test] Testing simple chat...")
    
    response = await llm.chat_complete("Say 'Hello, I am working!' in exactly those words.")
    print(f"[Test] Response: {response}")
    
    await llm.close()
    return True


if __name__ == "__main__":
    asyncio.run(test_connection())

