"""
Web Agent - Browser automation using browser-use + Ollama.

Uses local LLM (Ollama) to control web browsers via browser-use library.
Fully offline AI decision-making, only internet needed for actual browsing.

Features:
- DOM-based element detection (more reliable than vision-only)
- Natural language task execution
- Screenshot capture for frontend display
- Works 100% with local Ollama models
"""

import os
import asyncio
import base64
import json
import re
from typing import Optional, Callable, Dict, Any, List
from datetime import datetime


# Check if browser-use is available
BROWSER_USE_AVAILABLE = False
try:
    from browser_use import Agent, Browser
    from langchain_ollama import ChatOllama
    BROWSER_USE_AVAILABLE = True
    print("[WebAgent] browser-use library loaded successfully")
except ImportError as e:
    print(f"[WebAgent] browser-use not available: {e}")
    print("[WebAgent] Falling back to basic Playwright implementation")

# Fallback imports
from playwright.async_api import async_playwright, Page, Browser as PlaywrightBrowser


# Configuration
SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 900
DEFAULT_MODEL = "qwen2.5-coder:7b-instruct"  # Available local model


class WebAgent:
    """
    Web browsing agent using browser-use + local Ollama LLM.
    """
    
    def __init__(self, model: str = DEFAULT_MODEL):
        """
        Initialize the web agent.
        
        Args:
            model: Ollama model to use (default: llama3.1:8b)
        """
        self.model = model
        self.client = None
        self.browser = None
        self.context = None
        self.page = None
        self._screenshots: List[str] = []
        
        print(f"[WebAgent] Initialized with model: {model}")
        print(f"[WebAgent] browser-use available: {BROWSER_USE_AVAILABLE}")

    def denormalize_x(self, x: float, width: float) -> float:
        """Convert a normalized X coordinate in [0..1000] to pixel space."""
        try:
            return (float(x) / 1000.0) * float(width)
        except Exception:
            return 0.0

    def denormalize_y(self, y: float, height: float) -> float:
        """Convert a normalized Y coordinate in [0..1000] to pixel space."""
        try:
            return (float(y) / 1000.0) * float(height)
        except Exception:
            return 0.0
    
    async def run_task(
        self,
        prompt: str,
        update_callback: Optional[Callable] = None,
        max_steps: int = 15
    ) -> str:
        """
        Run a web browsing task.
        
        Args:
            prompt: Description of the task to accomplish
            update_callback: async callback(screenshot_b64, log_text)
            max_steps: Maximum number of actions to take
            
        Returns:
            Final result/summary from the agent
        """
        print(f"[WebAgent] Task: {prompt}")
        
        if BROWSER_USE_AVAILABLE:
            return await self._run_with_browser_use(prompt, update_callback, max_steps)
        else:
            return await self._run_with_playwright(prompt, update_callback, max_steps)
    
    async def _run_with_browser_use(
        self,
        prompt: str,
        update_callback: Optional[Callable] = None,
        max_steps: int = 15
    ) -> str:
        """Run task using browser-use library."""
        print("[WebAgent] Using browser-use engine")
        
        # browser-use v0.11+ changed API - ChatOllama not directly supported
        # Fall back to Playwright for now until we configure browser-use properly
        print("[WebAgent] Note: browser-use requires specific LLM setup, using Playwright fallback")
        return await self._run_with_playwright(prompt, update_callback, max_steps)
    
    async def _run_with_playwright(
        self,
        prompt: str,
        update_callback: Optional[Callable] = None,
        max_steps: int = 15
    ) -> str:
        """Fallback: Run task using basic Playwright + local LLM."""
        print("[WebAgent] Using Playwright fallback engine")
        
        from local_llm import LocalLLM, LLMConfig
        
        # System prompt for web navigation
        system_prompt = """You are a web browsing assistant. You control a browser to accomplish tasks.

Available actions (respond with JSON only):
- {"action": "navigate", "url": "https://..."}
- {"action": "click", "selector": "button.submit"} or {"action": "click", "x": 500, "y": 300}
- {"action": "type", "text": "search query", "selector": "input[name=q]"}
- {"action": "scroll", "direction": "down", "amount": 500}
- {"action": "wait", "seconds": 2}
- {"action": "back"}
- {"action": "done", "result": "summary of findings"}

Prefer CSS selectors over coordinates when possible.
Always respond with ONLY valid JSON, no other text."""

        config = LLMConfig(
            model=self.model,
            temperature=0.5,
            context_length=4096
        )
        llm = LocalLLM(config)
        llm.set_system_prompt(system_prompt)
        
        final_result = "Task completed"

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": SCREEN_WIDTH, "height": SCREEN_HEIGHT},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            page = await context.new_page()
            
            # Start at Google
            await page.goto("https://www.google.com")

            # Initial screenshot
            screenshot = await self._take_screenshot(page)
            if update_callback:
                await update_callback(screenshot, "Browser initialized")
            
            # Initial message
            initial_msg = f"""Task: {prompt}

Current page: {page.url}
Page title: {await page.title()}

What action should I take first? Respond with JSON only."""

            for turn in range(max_steps):
                print(f"[WebAgent] Turn {turn + 1}/{max_steps}")
                
                try:
                    if turn == 0:
                        message = initial_msg
                    else:
                        screenshot = await self._take_screenshot(page)
                        if update_callback:
                            await update_callback(screenshot, f"Turn {turn + 1}")
                        
                        # Get visible text for context
                        visible_text = await page.evaluate("""
                            () => {
                                const text = document.body.innerText;
                                return text.substring(0, 2000);
                            }
                        """)
                        
                        message = f"""Current page: {page.url}
Page title: {await page.title()}
Visible text preview: {visible_text[:500]}...

What action should I take next? Respond with JSON only."""

                    # Get LLM response
                    response = ""
                    async for chunk in llm.chat(message):
                        response += chunk
                    
                    if not response.strip():
                        continue
                    
                    # Parse action
                    action = self._parse_action(response)
                    if not action:
                        print(f"[WebAgent] Could not parse: {response[:100]}")
                        continue
                    
                    action_type = action.get("action", "")
                    print(f"[WebAgent] Action: {action_type}")
                    
                    # Execute action
                    if action_type == "done":
                        final_result = action.get("result", "Task completed")
                        break
                    
                    await self._execute_action(page, action)
                    await asyncio.sleep(1)
                    
                    if update_callback:
                        screenshot = await self._take_screenshot(page)
                        await update_callback(screenshot, f"Executed: {action_type}")
                    
                except Exception as e:
                    print(f"[WebAgent] Turn error: {e}")
            
            await browser.close()
        
        print(f"[WebAgent] Final result: {final_result}")
        return final_result
    
    async def _take_screenshot(self, page) -> str:
        """Take a screenshot and return as base64."""
        try:
            screenshot_bytes = await page.screenshot(type="png")
            return base64.b64encode(screenshot_bytes).decode('utf-8')
        except:
            return ""
    
    def _parse_action(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse action JSON from LLM response."""
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON
        patterns = [
            r'\{[^{}]*"action"[^{}]*\}',
            r'\{.*?\}',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    continue
        
        return None
    
    async def _execute_action(self, page, action: Dict[str, Any]):
        """Execute a browser action."""
        action_type = action.get("action", "")
        
        if action_type == "navigate":
            url = action.get("url", "")
            if url:
                if not url.startswith("http"):
                    url = "https://" + url
                await page.goto(url, wait_until="domcontentloaded")
        
        elif action_type == "click":
            selector = action.get("selector")
            if selector:
                try:
                    await page.click(selector, timeout=5000)
                except:
                    # Fallback to coordinates
                    x = action.get("x", 0)
                    y = action.get("y", 0)
                    if x and y:
                        await page.mouse.click(x, y)
            else:
                x = action.get("x", 0)
                y = action.get("y", 0)
                if x and y:
                    await page.mouse.click(x, y)
        
        elif action_type == "type":
            text = action.get("text", "")
            selector = action.get("selector")
            press_enter = action.get("press_enter", True)
            
            if selector:
                await page.fill(selector, text)
            else:
                x = action.get("x", 0)
                y = action.get("y", 0)
                if x and y:
                    await page.mouse.click(x, y)
                    await asyncio.sleep(0.2)
                await page.keyboard.type(text)

            if press_enter:
                await page.keyboard.press("Enter")
        
        elif action_type == "scroll":
            direction = action.get("direction", "down")
            amount = action.get("amount", 500)
            delta_y = amount if direction == "down" else -amount
            await page.mouse.wheel(0, delta_y)
        
        elif action_type == "wait":
            seconds = action.get("seconds", 2)
            await asyncio.sleep(seconds)
        
        elif action_type == "back":
            await page.go_back()


class WebResearchAgent(WebAgent):
    """
    Specialized agent for research tasks.
    Optimized for gathering information from multiple sources.
    """
    
    async def research(
        self,
        topic: str,
        sources: int = 3,
        update_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Research a topic across multiple sources.
        
        Args:
            topic: Topic to research
            sources: Number of sources to check
            update_callback: Progress callback
            
        Returns:
            Dictionary with findings
        """
        prompt = f"""Research the following topic: "{topic}"

Instructions:
1. Search for "{topic}" on Google
2. Visit {sources} relevant results
3. Extract key information from each
4. Summarize your findings

When done, respond with: {{"action": "done", "result": "Your comprehensive summary..."}}"""

        result = await self.run_task(prompt, update_callback, max_steps=20)
        
        return {
            "topic": topic,
            "summary": result,
            "timestamp": datetime.now().isoformat()
        }


# Standalone web agent instance for server
_standalone_web_agent: Optional[WebAgent] = None


def get_web_agent(model: str = DEFAULT_MODEL) -> WebAgent:
    """Get or create a standalone web agent instance."""
    global _standalone_web_agent
    if _standalone_web_agent is None:
        _standalone_web_agent = WebAgent(model=model)
    return _standalone_web_agent


# Test function
async def test_web_agent():
    """Test the web agent."""
    print("\n" + "="*60)
    print("[Test] Web Agent Test")
    print("="*60)
    
    agent = WebAgent()
    
    async def callback(screenshot_b64, log):
        print(f"[Update] {log} (screenshot: {len(screenshot_b64)} chars)")
    
    print("\n[Test] Running search task...")
    result = await agent.run_task(
        "Search Google for 'Python CadQuery tutorial' and tell me what you find",
        update_callback=callback,
        max_steps=10
    )
    
    print(f"\n[Test] Result: {result}")
    print("="*60)
    return True


if __name__ == "__main__":
    asyncio.run(test_web_agent())
