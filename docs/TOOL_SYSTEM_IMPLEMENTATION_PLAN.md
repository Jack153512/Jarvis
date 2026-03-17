# Tool System Implementation Plan: Web Search & System Info

**Status:** Planning (No Coding Yet)  
**Date:** March 2025

---

## 1. Current Tool Architecture (Summary)

| Component | Location | Behavior |
|-----------|----------|----------|
| **Tool definitions** | `backend/tools.py` — `tools_list` | JSON schema: name, description, parameters |
| **System prompt** | `backend/jarvis.py` — `SYSTEM_PROMPT` | Tools listed manually; `get_tools_prompt()` exists but is **not used** |
| **Tool extraction** | `backend/local_llm.py` — `extract_tool_calls()` | Regex: `\{[^{}]*"tool"[^{}]*\}` |
| **Tool execution** | `backend/jarvis.py` — `_execute_tool()` | if/elif dispatch by `tool_name` |
| **Tool result** | Emitted to frontend via `tool_activity` | **Not fed back to LLM** |
| **Permissions** | `backend/server.py` — `tool_permissions` | Per-tool allow/deny; Settings UI in `SettingsWindow.jsx` |

**Flow:** User message → LLM generates response (may include JSON tool call) → `_process_tool_calls` extracts → `_execute_tool` runs → result emitted to frontend only.

---

## 2. Tool Interface Definition

### 2.1 Tool Schema (Existing Pattern)

Each tool in `tools_list` has:

```python
{
    "name": "tool_name",
    "description": "What the tool does and when to use it.",
    "parameters": {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "..."},
            "param2": {"type": "string", "description": "..."}
        },
        "required": ["param1"]
    }
}
```

### 2.2 Tool Execution Contract

- **Input:** `tool_name: str`, `args: Dict[str, Any]`
- **Output:** `Any` — dict, list, str, or None
- **Format for LLM:** `format_tool_result(tool_name, result)` produces a string the LLM can interpret
- **Async:** Tools may be async (e.g., web search over network)

### 2.3 Tool Registry (Proposed)

To avoid hardcoding in `_execute_tool`, introduce a **tool registry**:

```python
# Conceptual
TOOL_REGISTRY: Dict[str, Callable] = {
    "generate_cad": _handle_cad_request,
    "web_search": _handle_web_search,
    "system_info": _handle_system_info,
    ...
}
```

For now, we can keep the if/elif in `_execute_tool` and add two new branches. A registry can be a later refactor.

---

## 3. Tool 1 — Web Search

### 3.1 Requirements

| Requirement | Specification |
|-------------|---------------|
| Input | `query: str` — search query from LLM |
| Output | Structured list: `[{title, url, snippet}, ...]` |
| API | Free, no API key — **DuckDuckGo** via `duckduckgo-search` |
| Rate limits | Throttle: max 5 searches per minute (configurable) |
| Timeout | 10 seconds per request |
| Error handling | Return `{"error": "message"}` on failure |
| Max results | 5–8 per query (configurable) |

### 3.2 Tool Definition

```python
{
    "name": "web_search",
    "description": "Search the web for up-to-date information. Use when the user asks about current events, recent news, documentation, or anything that may have changed since your training. Returns title, URL, and snippet for each result.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (e.g., 'Python 3.12 release notes', 'latest news about AI')"
            }
        },
        "required": ["query"]
    }
}
```

### 3.3 Implementation Approach

- **Library:** `duckduckgo-search` — `pip install duckduckgo-search`
- **Function:** `async def web_search(query: str) -> List[Dict]` or sync wrapped in `asyncio.to_thread`
- **Return format:**
  ```python
  [
      {"title": "...", "url": "https://...", "snippet": "..."},
      ...
  ]
  ```
- **Error:** `[{"error": "Search failed: <reason>"}]` or `{"error": "..."}` for `format_tool_result`

### 3.4 Considerations

| Consideration | Approach |
|---------------|----------|
| Rate limits | Simple in-memory counter + timestamp; skip if exceeded |
| Timeout | `asyncio.wait_for(..., timeout=10)` |
| Clean formatting | Truncate snippets to ~200 chars; sanitize URLs |
| No API key | DuckDuckGo HTML scraping — may break if site changes; fallback: return error |

---

## 4. Tool 2 — System Info

### 4.1 Requirements

| Requirement | Specification |
|-------------|---------------|
| Input | Optional `sections: List[str]` — e.g. `["os", "cpu", "memory", "gpu"]` or empty for all |
| Output | Structured dict with requested sections |
| Cross-platform | Windows, Linux, macOS |
| Safety | No sensitive env vars (e.g., API keys, tokens) |

### 4.2 Tool Definition

```python
{
    "name": "system_info",
    "description": "Retrieve system information: OS, CPU usage, RAM usage, GPU (if available). Use when the user asks about their computer, performance, hardware, or environment.",
    "parameters": {
        "type": "object",
        "properties": {
            "sections": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional. Sections to include: 'os', 'cpu', 'memory', 'gpu'. If omitted, returns all."
            }
        },
        "required": []
    }
}
```

### 4.3 Output Structure

```python
{
    "os": {
        "type": "Windows",
        "version": "10.0.26200",
        "machine": "AMD64"
    },
    "cpu": {
        "usage_percent": 12.5,
        "count": 8,
        "name": "Intel Core i7-..."
    },
    "memory": {
        "total_gb": 32.0,
        "available_gb": 18.5,
        "used_percent": 42.1
    },
    "gpu": {
        "name": "NVIDIA GeForce RTX 4060 Ti",
        "memory_total_mb": 16384,
        "memory_used_mb": 1024,
        "memory_free_mb": 15360
    }  # or null if no GPU / nvidia-smi unavailable
}
```

### 4.4 Implementation Approach

- **Libraries:** `platform`, `psutil` (CPU, RAM), `subprocess` for `nvidia-smi` (GPU on Windows/Linux)
- **psutil:** `pip install psutil` — cross-platform CPU/RAM
- **GPU:** Parse `nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader`
- **Env vars:** Exclude; or only include safe ones (e.g. `PATH`, `USER`) — optional, can omit for v1

---

## 5. How the LLM Selects a Tool

### 5.1 Current Mechanism

- System prompt lists tools and format: `{"tool": "name", "args": {...}}`
- LLM generates free-form text; we parse JSON via regex
- No native Ollama tools API; no structured function-calling schema

### 5.2 No Change Required

The existing flow works: the LLM decides when to use a tool based on the system prompt. We only need to:

1. Add `web_search` and `system_info` to the tool list in the system prompt (or use `get_tools_prompt()`)
2. Add their definitions to `tools_list` in `tools.py`
3. Implement handlers and wire them in `_execute_tool`

### 5.3 Prompt Update

Either:

- **Option A:** Add the two tools to the hardcoded list in `SYSTEM_PROMPT` (consistent with current approach)
- **Option B:** Use `get_tools_prompt()` from `tools.py` so the prompt is generated from `tools_list` (single source of truth)

**Recommendation:** Option B — refactor to use `get_tools_prompt()` so new tools are automatically included.

---

## 6. How Tool Outputs Are Returned to the Model

### 6.1 Current Behavior

- Tool result is **not** fed back to the LLM
- Only emitted to frontend via `tool_activity`
- LLM cannot say "I searched and found..." or "Your system has 16GB RAM"

### 6.2 Proposed Change (Tool Result Feedback)

After tool execution:

1. Format result: `formatted = format_tool_result(tool_name, result)`
2. Add to conversation: `llm.add_message("user", f"[Tool result: {tool_name}]\n{formatted}")`
3. **Optional:** Trigger a follow-up LLM call to generate a natural-language response incorporating the result

**Implementation choice:**

- **Minimal:** Add tool result as a user message, then call `process_text_input` with a synthetic message like `"Based on the tool result above, provide a concise summary for the user."` — but that would create a recursive loop.
- **Simpler:** Add tool result as a **user** message in history, then make **one** additional LLM call with an empty user message or a prompt like "Summarize the tool result above in a brief, helpful way for the user." The assistant's response would be streamed as usual.

**Recommended for v1:** Add the tool result to the conversation history as a user message (simulating "the system has provided this data"), then make a **single** follow-up LLM call with a system-like instruction: e.g. `"The user asked a question. You used a tool and received the result above. Now provide a clear, helpful response based on that result. Do not repeat the raw data; summarize and interpret it."` — and stream that response.

**Simpler alternative:** Just add the tool result to history and do **not** auto-generate a follow-up. The LLM would only see it in the next user turn. That's simpler but the user wouldn't get an immediate spoken/sent response about the tool result. Given the current flow (stream response → process tools → speak response), the tool runs *after* the initial response. So we need a follow-up call to get a response that incorporates the tool result.

**Flow with tool result feedback:**

1. User: "What's the latest Python version?"
2. LLM responds with `{"tool": "web_search", "args": {"query": "Python latest version 2025"}}` (and maybe some text)
3. We extract tool call, execute it, get results
4. We add to history: user message with tool result
5. We make a **second** LLM call: "Based on the search results above, answer the user's question concisely."
6. Stream that response to the user (and TTS)

This requires modifying the flow in `process_text_input` to support a "continuation" after tool execution.

---

## 7. Files to Modify or Create

### 7.1 New Files

| File | Purpose |
|------|---------|
| `backend/tools/web_search.py` | Web search implementation (optional: keep in jarvis or tools module) |
| `backend/tools/system_info.py` | System info implementation (optional) |

**Simpler:** Implement both as functions in a single new module `backend/tool_handlers.py` to avoid over-engineering.

### 7.2 Modified Files

| File | Changes |
|------|---------|
| `backend/tools.py` | Add `web_search` and `system_info` to `tools_list` |
| `backend/jarvis.py` | Add tool handlers in `_execute_tool`; add tool result feedback + follow-up LLM call; use `get_tools_prompt()` for system prompt (or add tools to prompt) |
| `backend/server.py` | Add `web_search` and `system_info` to `tool_permissions` in `DEFAULT_SETTINGS` |
| `requirements.txt` | Add `duckduckgo-search`, `psutil` |
| `src/components/SettingsWindow.jsx` | Add `web_search` and `system_info` to `TOOLS` array for permissions UI |

### 7.3 Optional / Later

| File | Change |
|------|--------|
| `backend/jarvis.py` | Refactor `_execute_tool` to use a registry pattern |
| `backend/jarvis.py` | Use `get_tools_prompt()` instead of hardcoded tool list |

---

## 8. Implementation Order

1. **Add dependencies** — `duckduckgo-search`, `psutil` to `requirements.txt`
2. **Implement `web_search`** — New function, sync or async, with timeout and error handling
3. **Implement `system_info`** — New function using `platform`, `psutil`, `nvidia-smi`
4. **Register tools** — Add to `tools_list`, system prompt, `_execute_tool`, `tool_permissions`, Settings UI
5. **Tool result feedback** — Add result to history and implement follow-up LLM call (or defer to a later phase)
6. **Test** — Manual tests: "Search for X", "What's my system info?"

---

## 9. Permission Defaults

| Tool | Default | Rationale |
|------|---------|-----------|
| `web_search` | `True` | Read-only, fetches public info |
| `system_info` | `True` | Read-only, no sensitive data exposed |

Both can be toggled in Settings.

---

## 10. Error Handling Summary

| Scenario | Behavior |
|---------|----------|
| Web search timeout | Return `{"error": "Search timed out"}` |
| Web search rate limit | Return `{"error": "Rate limit exceeded. Try again in a minute."}` |
| DuckDuckGo unavailable | Return `{"error": "Search service unavailable"}` |
| System info: psutil missing | Return `{"error": "System info unavailable"}` |
| System info: no NVIDIA GPU | `gpu` field is `null` |
| Unknown tool | Log and skip; no crash |

---

## 11. Summary

| Item | Decision |
|------|----------|
| **Tool interface** | Same as existing: `tools_list` schema, `_execute_tool` dispatch |
| **LLM tool selection** | Unchanged: prompt lists tools, LLM outputs JSON |
| **Tool output to model** | Add tool result to history + optional follow-up LLM call |
| **Web search** | `duckduckgo-search`, 10s timeout, 5–8 results, rate limit |
| **System info** | `platform` + `psutil` + `nvidia-smi`, structured dict |
| **New module** | `backend/tool_handlers.py` (or inline in jarvis) |
| **Files to touch** | `tools.py`, `jarvis.py`, `server.py`, `requirements.txt`, `SettingsWindow.jsx` |

---

*Ready for implementation once this plan is approved.*
