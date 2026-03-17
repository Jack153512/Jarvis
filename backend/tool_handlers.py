"""
Tool handlers for Web Search and System Info.
Used by jarvis.py _execute_tool.
"""
import asyncio
import logging

logger = logging.getLogger("jarvis.tools")
import platform
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

# Rate limit for web search (only applies to network requests, not cache hits)
_WEB_SEARCH_LAST_CALL: float = 0.0
_WEB_SEARCH_MIN_INTERVAL: float = 2.0  # seconds between real API calls

# In-memory cache: (query_lower, max_results) -> (result_dict, expiry_time)
_WEB_SEARCH_CACHE: Dict[Tuple[str, int], Tuple[Dict[str, Any], float]] = {}
_CACHE_TTL: float = 120.0  # seconds


def _normalize_results(raw: List[Dict], max_results: int) -> List[Dict[str, str]]:
    """Normalize search results to {title, url, snippet}."""
    formatted = []
    for r in (raw or [])[:max_results]:
        title = (r.get("title") or "")[:200]
        url = (r.get("href") or r.get("url") or r.get("link") or "")[:500]
        snippet = (r.get("body") or r.get("snippet") or r.get("description") or "")[:300]
        formatted.append({"title": title, "url": url, "snippet": snippet})
    return formatted


def _do_search_ddgs(query: str, max_results: int) -> List[Dict]:
    """Use ddgs package (multi-backend: bing, brave, duckduckgo, etc.)."""
    from ddgs import DDGS
    return list(DDGS().text(query, max_results=max_results, backend="auto"))


def _do_search_duckduckgo(query: str, max_results: int) -> List[Dict]:
    """Fallback: duckduckgo-search package."""
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


async def web_search(query: str, max_results: int = 6, timeout: float = 10.0) -> Dict[str, Any]:
    """
    Search the web. Uses ddgs (multi-backend) or duckduckgo-search. No API key required.
    Results are cached to reduce rate-limit issues.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        timeout: Request timeout in seconds
        
    Returns:
        {"results": [{"title", "url", "snippet"}, ...]} or {"error": "message"}
    """
    global _WEB_SEARCH_LAST_CALL, _WEB_SEARCH_CACHE
    
    query = str(query or "").strip()
    if not query:
        return {"error": "Search query cannot be empty."}
    
    cache_key = (query.lower(), max_results)
    now = time.time()
    
    # Check cache first (bypasses rate limit)
    if cache_key in _WEB_SEARCH_CACHE:
        cached_result, expiry = _WEB_SEARCH_CACHE[cache_key]
        if now < expiry:
            logger.debug("web_search cache hit: query=%r, %d results", query, len(cached_result.get("results", [])))
            return cached_result
        del _WEB_SEARCH_CACHE[cache_key]
    
    # Rate limit only for real network requests
    wait_time = _WEB_SEARCH_MIN_INTERVAL - (now - _WEB_SEARCH_LAST_CALL)
    if wait_time > 0:
        # Wait briefly instead of failing (handles assistant retry loop)
        await asyncio.sleep(min(wait_time, 3.0))
    
    # Try ddgs first (multi-backend), then duckduckgo-search
    def _do_search():
        try:
            return _do_search_ddgs(query, max_results)
        except ImportError:
            return _do_search_duckduckgo(query, max_results)
    
    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(_do_search),
            timeout=timeout
        )
        _WEB_SEARCH_LAST_CALL = time.time()
        formatted = _normalize_results(results, max_results)
        out = {"results": formatted, "query": query}
        _WEB_SEARCH_CACHE[cache_key] = (out, now + _CACHE_TTL)
        # Limit cache size
        if len(_WEB_SEARCH_CACHE) > 50:
            oldest = min(_WEB_SEARCH_CACHE.items(), key=lambda x: x[1][1])
            del _WEB_SEARCH_CACHE[oldest[0]]
        logger.info("web_search: query=%r -> %d results", query, len(formatted))
        return out
    except asyncio.TimeoutError:
        logger.warning("web_search timeout: query=%r", query)
        return {"error": "Search timed out. Try again."}
    except Exception as e:
        logger.warning("web_search error: query=%r, error=%s", query, e)
        return {"error": f"Search failed: {str(e)}"}


def system_info(sections: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Retrieve system information: OS, CPU, memory, GPU.
    
    Args:
        sections: Optional list of sections to include: "os", "cpu", "memory", "gpu".
                  If None or empty, returns all.
        
    Returns:
        Structured dict with requested sections
    """
    out: Dict[str, Any] = {}
    requested = set(sections or []) or {"os", "cpu", "memory", "gpu"}
    
    if "os" in requested:
        out["os"] = {
            "type": platform.system(),
            "version": platform.version(),
            "release": platform.release(),
            "machine": platform.machine(),
        }
    
    if "cpu" in requested or "memory" in requested:
        try:
            import psutil
        except ImportError:
            out["error"] = "System info unavailable. Install psutil: pip install psutil"
            return out
        
        if "cpu" in requested:
            try:
                out["cpu"] = {
                    "count": psutil.cpu_count(logical=True) or 0,
                    "usage_percent": round(psutil.cpu_percent(interval=0.5), 1),
                }
            except Exception:
                out["cpu"] = {"error": "Could not read CPU info"}
        
        if "memory" in requested:
            try:
                mem = psutil.virtual_memory()
                out["memory"] = {
                    "total_gb": round(mem.total / (1024**3), 2),
                    "available_gb": round(mem.available / (1024**3), 2),
                    "used_percent": round(mem.percent, 1),
                }
            except Exception:
                out["memory"] = {"error": "Could not read memory info"}
    
    if "gpu" in requested:
        gpu_info = _get_gpu_info()
        out["gpu"] = gpu_info
    
    return out


def _get_gpu_info() -> Optional[Dict[str, Any]]:
    """Get NVIDIA GPU info via nvidia-smi. Returns None if unavailable."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            name = parts[0]
            try:
                total_mb = int(float(parts[1]))
                used_mb = int(float(parts[2]))
                free_mb = int(float(parts[3]))
            except (ValueError, IndexError):
                return {"name": name, "memory_total_mb": None, "memory_used_mb": None, "memory_free_mb": None}
            return {
                "name": name,
                "memory_total_mb": total_mb,
                "memory_used_mb": used_mb,
                "memory_free_mb": free_mb,
            }
        return {"name": parts[0] if parts else "NVIDIA GPU"}
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return None
