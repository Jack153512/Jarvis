# LLM Architecture Audit & Assistant Evolution Plan

**Document Version:** 1.0  
**Date:** March 2025  
**Status:** Research & Planning (No Implementation Yet)

---

## Phase 1 — Audit of Current LLM Architecture

### 1.1 Model Layer

| Aspect | Current State |
|--------|---------------|
| **Model** | `qwen2.5-coder:7b-instruct` (hardcoded in `jarvis.py`, overridable via `server.py` settings) |
| **Runtime** | **Ollama** — local HTTP API at `http://127.0.0.1:11434` |
| **Temperature** | `0.7` (jarvis) / `0.0` (warmup) |
| **Context length** | `8192` tokens (jarvis) / `2048` (warmup) |
| **Timeout** | Default `240s` in `LLMConfig`; warmup uses `30s` |
| **Base URL** | Configurable via `SETTINGS["llm"]["base_url"]` |

**Other LLM instances in the codebase:**
- **CAD Agent (cad_agent.py):** `deepseek-r1:14b`, temp=0.2, context=32768
- **CAD Agent v2 (cad_agent_v2.py):** `blenderllm`, temp=0.7, context=8192
- **Server warmup:** Uses `SETTINGS["llm"]` or defaults to `qwen2.5-coder:7b-instruct`

**Note:** `local_llm.py` defaults to `deepseek-r1:14b` in `LLMConfig`, but `jarvis.py` overrides with `qwen2.5-coder:7b-instruct`.

---

### 1.2 Invocation Layer

| Aspect | Current State |
|--------|---------------|
| **Entry point** | `audio_loop.process_text_input(text)` |
| **Trigger** | Socket event `user_input` → `server.py` line 861–873 |
| **Flow** | `user_input` → `audio_loop.process_text_input(text)` → `self.llm.chat(text)` |
| **Mode** | **Streaming** — `LocalLLM.chat()` yields chunks via `async for chunk in self.llm.chat(text)` |
| **Sync/Async** | Fully async (`aiohttp` to Ollama) |

**Prompt construction:** `LocalLLM._build_messages(user_message)` builds:
1. `[system]` — system prompt (if set)
2. `[user, assistant, user, assistant, ...]` — full `conversation_history`
3. `[user]` — current message

---

### 1.3 Prompt Architecture

| Aspect | Current State |
|--------|---------------|
| **System prompt location** | `backend/jarvis.py` lines 53–91 — `SYSTEM_PROMPT` constant |
| **Composition** | `_compose_system_prompt(lang)` = base + Identity + learning suffix + language rules |
| **Format** | Single block of natural language; no structured templates |
| **Role separation** | Yes — system / user / assistant via Ollama API |
| **Tool descriptions** | Hardcoded in system prompt; `tools.py` has `get_tools_prompt()` but it is **not used** by jarvis |
| **Conversation history** | Appended in full from `llm.conversation_history` |

**System prompt structure:**
- Personality (Jarvis, British, witty, concise)
- Pacing rules (~120 words, 1–3 paragraphs)
- Tool list (names only) + JSON format: `{"tool": "name", "args": {...}}`
- Language lock (en/vi) appended dynamically

---

### 1.4 Context / Memory

| Capability | Supported | Implementation |
|------------|-----------|----------------|
| **Short-term conversation** | ✅ | `LocalLLM.conversation_history` (in-memory list of `Message`) |
| **Persistent memory** | ✅ | `JarvisMemory` (SQLite) — `llm_history` per conversation |
| **Session history** | ✅ | Multi-conversation via `conversations` table; `export_llm_history` / `import_llm_history` on switch |
| **Long-term memory** | ⚠️ Partial | `upload_memory` injects text as system message; `personal_memory`, `project_memory` exist but are **not** injected into LLM prompts |
| **Context window optimization** | ❌ | No summarization, truncation, or sliding window; full history sent every turn |

**Storage:**
- `jarvis_memory.db` — conversations, `llm_history` JSON, chat_history, personal/project/short_term memory
- `jarvis_learning.db` — recommendations, tool events, feedback, strategy weights

---

### 1.5 Output Pipeline

| Aspect | Current State |
|--------|---------------|
| **Streaming** | ✅ Chunks yielded from `llm.chat()` |
| **Frontend delivery** | `on_transcription({"sender": assistant_name, "text": chunk})` → Socket `transcription` event |
| **Soft cap** | `soft_cap_chars` (450–1200) from LearningPolicy; streaming stops early unless `{"tool"` detected |
| **Normalization** | `_normalize_chunk()` strips `<think>...</think>` blocks (DeepSeek-R1 style) |
| **TTS** | `_clean_text_for_tts()` removes JSON, code blocks, markdown before speaking |
| **Tool detection** | After streaming: `extract_tool_calls(response_text)` — regex for `{"tool": ...}` |

---

### 1.6 Tool / Capability Layer

| Capability | Supported | Implementation |
|------------|-----------|----------------|
| **Function calling** | ⚠️ Pseudo | LLM outputs JSON in free text; parsed via regex `\{[^{}]*"tool"[^{}]*\}` |
| **Tool usage** | ✅ | 9 tools: generate_cad, iterate_cad, run_web_agent, write_file, read_file, read_directory, create_project, switch_project, list_projects |
| **Tool result feedback** | ❌ | Tool results are **not** fed back to the LLM; only emitted to frontend via `tool_activity` |
| **Retrieval / knowledge** | ❌ | No RAG, no vector search |
| **System commands** | ❌ | No shell/exec |
| **Multi-step reasoning** | ❌ | Single-turn; no agentic loop |

**Tool extraction:** `LocalLLM.extract_tool_calls()` — regex-based, fragile; no schema validation.

---

### 1.7 Architecture Diagram (Current)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ FRONTEND (Electron + React)                                                  │
│  ├─ Chat input → socket.emit('user_input', { text })                        │
│  └─ socket.on('transcription') → append to AI bubble, stream to UI          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ SERVER (FastAPI + Socket.IO)                                                  │
│  ├─ user_input → audio_loop.process_text_input(text)                         │
│  ├─ create/load/delete conversation → save/load llm_history                  │
│  └─ upload_memory → audio_loop.llm.add_message("system", context)             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ JARVIS (AudioLoop)                                                            │
│  ├─ process_text_input → LLM.chat(text)                                      │
│  ├─ _compose_system_prompt(lang) → system + identity + learning + language    │
│  ├─ Stream chunks → on_transcription(assistant, chunk)                      │
│  ├─ extract_tool_calls(response) → execute tools                              │
│  └─ _speak(response_text) → TTS → on_audio_data                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ LocalLLM (Ollama)                                                            │
│  ├─ _build_messages(user_msg) → [system, ...history, user]                   │
│  ├─ POST /api/chat → stream chunks                                          │
│  ├─ add_message(user, msg) + add_message(assistant, full_response)          │
│  └─ extract_tool_calls(response) → regex JSON parse                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 2 — System Limitations (Why It Behaves Like a Basic Chatbot)

### 2.1 Architectural Weaknesses

| Issue | Root Cause |

|------|-------------|
| **Weak conversational structure** | System prompt is a single block; no structured sections (identity, behavior, tools, constraints). No few-shot examples. |
| **Poor reasoning capability** | Model is Qwen2.5-Coder (code-focused); no explicit reasoning instructions. No chain-of-thought or step-by-step prompts. |
| **Limited analytical responses** | No prompt design for "break down the problem" or "analyze before answering". Soft cap (450–1200 chars) truncates long responses. |
| **Inefficient prompt construction** | Full history sent every turn; no summarization. `get_tools_prompt()` exists but is unused; tool descriptions are minimal. |
| **No structured outputs** | Free-form text only; no JSON schema, no markdown sections for structured answers. |
| **No tool integration** | Tool results never fed back to LLM. No agentic loop. LLM cannot say "I've created the file" or "I've generated the CAD." |
| **Weak context management** | No context window optimization; no retrieval of relevant past context; personal/project memory not injected. |
| **Static personality** | Personality is fixed in prompt; no dynamic adaptation based on task or user preference. |
| **Fragile tool extraction** | Regex-based JSON parsing; fails on nested JSON or malformed output. |
| **No native function calling** | Ollama supports `tools` in JSON schema; we use ad-hoc JSON-in-text. |

### 2.2 Model-Level vs Architecture-Level

| Limitation | Model | Architecture |
|------------|-------|---------------|
| Reasoning depth | Qwen2.5-Coder is code-optimized, not reasoning-optimized | No prompts for "think step by step" or "analyze first" |
| Response quality | 7B model has limited capacity | Soft cap cuts responses early; no structured output format |
| Tool use | Model can output JSON | We don't use Ollama tools API; no result feedback loop |
| Context | 8K context | Full history sent; no summarization or retrieval |

---

## Phase 3 — Improved Assistant Architecture Design

### 3.1 Stronger Prompt Architecture

**Proposed structure:**
```
[IDENTITY]
- Name, role, creator
- Behavioral rules (e.g., analytical, concise, helpful)

[CAPABILITIES]
- Tool descriptions (from tools.py get_tools_prompt or schema)
- When to use tools vs when to answer directly

[CONSTRAINTS]
- Pacing, language lock, output format

[CONTEXT] (optional)
- Current project, recent tool results, retrieved memory
```

**Implementation:**
- Dedicated `prompt_manager` module
- Templates: `system_identity`, `system_tools`, `system_constraints`
- Role separation enforced in message array

---

### 3.2 Better Context Management

- **Sliding window:** Keep last N turns + optional summary of older turns
- **Summarization:** When approaching context limit, summarize oldest messages
- **Memory injection:** Inject `personal_memory` and `project_memory` when relevant (e.g., user name, project context)
- **Tool result injection:** After tool execution, add `[Tool result: ...]` to conversation and optionally trigger a follow-up LLM call

---

### 3.3 Analytical Capability

**Prompt additions:**
- "When the user asks for analysis or a complex answer, break the problem into steps."
- "For technical questions, consider: what is known? what is unknown? what are the options?"
- "Provide actionable answers when possible; if uncertain, ask one clarifying question."

**Optional:** Use a reasoning-focused model (e.g., Qwen3.5, DeepSeek-R1) for analytical turns; keep Qwen2.5-Coder for code/tool-heavy turns.

---

### 3.4 Ollama Model Research

| Model | Size | Strengths | Use Case |
|------|------|-----------|----------|
| **qwen2.5-coder:7b-instruct** | 7B | Code generation, 128K context | Current; good for tools/code |
| **qwen2.5:7b-instruct** | 7B | General chat, instruction-following | Better for conversational assistant |
| **qwen2.5:14b-instruct** | 14B | Stronger reasoning, still efficient | Balanced upgrade |
| **qwen3.5:7b** | 7B | Multimodal, thinking variants | If vision/reasoning needed |
| **deepseek-r1:7b** | 7B | Explicit reasoning (think blocks) | Analytical tasks |
| **deepseek-r1:14b** | 14B | Best reasoning in class | High-quality analysis; ~9GB VRAM |
| **mistral:7b-instruct** | 7B | General purpose, good instruction | Alternative to Qwen |
| **llama3.2:3b** | 3B | Lightweight | Low-resource fallback |

**Recommendation for assistant evolution:**
1. **Primary:** `qwen2.5:7b-instruct` or `qwen2.5:14b-instruct` — better general assistant behavior than coder variant
2. **Reasoning:** `deepseek-r1:7b` or `deepseek-r1:14b` — for analytical tasks (optional model routing)
3. **Stay with coder:** If tool/code use is primary, keep `qwen2.5-coder:7b` but improve prompts

---

### 3.5 Tool Integration (Recommended)

1. **Use Ollama tools API** — Pass tool definitions in `/api/chat`; model returns structured tool calls
2. **Agentic loop** — After tool execution, add result to messages and call LLM again for natural-language summary
3. **Tool result format** — `[Tool: generate_cad] Result: output.stl created successfully.`
4. **New tools (optional):** system_info, search, run_command (with safety)

---

### 3.6 Assistant Identity & Behavior

**Target persona:**
- Professional but natural
- Analytical — breaks down problems when appropriate
- Helpful for technical workflows (CAD, code, files, web)
- Explains reasoning when useful
- Concise by default; expands when asked

**Behavior rules (to add to prompt):**
- Prefer one clarifying question over wrong assumptions
- For "how" or "why" questions, give a short explanation before diving in
- When using tools, briefly confirm what was done in natural language

---

## Phase 4 — Implementation Plan (No Coding Yet)

### 4.1 Backend Changes

| File | Changes |
|------|---------|
| **`backend/jarvis.py`** | Refactor `_compose_system_prompt`; integrate `prompt_manager`; add tool-result feedback loop; optionally route to different models |
| **`backend/local_llm.py`** | Add Ollama tools support; improve `extract_tool_calls` or replace with native parsing; add `chat_with_tool_result()` for agentic loop |
| **`backend/tools.py`** | Ensure `get_tools_prompt()` is used; add Ollama-compatible tool schema export |
| **`backend/memory.py`** | Add `get_relevant_context()` for injecting personal/project memory into prompts |
| **`backend/server.py`** | Pass model config from settings to jarvis; ensure LLM warmup uses correct model |

### 4.2 New Modules

| Module | Purpose |
|--------|---------|
| **`prompt_manager.py`** | System prompt templates, composition, role sections |
| **`context_manager.py`** | Sliding window, summarization, context assembly |
| **`tool_router.py`** | Execute tools, format results, optionally trigger follow-up LLM call |
| **`memory_handler.py`** | Retrieve and inject relevant memory into prompts (optional; can extend `memory.py`) |

### 4.3 Frontend Impact

| Area | Changes |
|------|---------|
| **Response formatting** | No change if streaming stays the same; tool results may appear as additional messages |
| **Streaming** | May need to handle multi-turn tool loops (e.g., show "Using tool..." then streamed response) |
| **Settings** | Add model selector if we support multiple models |

### 4.4 Implementation Order

1. **Prompt architecture** — `prompt_manager`, refactor `_compose_system_prompt`
2. **Tool result feedback** — Add tool result to conversation; optional follow-up LLM call
3. **Context management** — Sliding window, summarization
4. **Model evaluation** — Test `qwen2.5:7b` vs `qwen2.5-coder:7b` for assistant behavior
5. **Ollama tools API** — Migrate from regex to native tools (if Ollama version supports)
6. **Memory injection** — Optional; inject personal/project context when relevant

---

## Summary

The current system is a **streaming chatbot** with a single-turn flow, basic tool execution, and no agentic loop. To evolve into a **capable AI assistant**, we need:

1. **Stronger prompts** — Structured, analytical, with clear tool instructions
2. **Tool result feedback** — So the LLM can acknowledge and act on tool results
3. **Better context** — Sliding window, summarization, optional memory injection
4. **Model choice** — Consider `qwen2.5:7b` or `qwen2.5:14b` for general assistant; `deepseek-r1` for reasoning
5. **Optional agentic loop** — Multi-turn tool use with natural-language summaries

---

*Next step: Review this plan with stakeholders; approve architecture; then begin implementation.*
