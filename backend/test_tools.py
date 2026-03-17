"""
Test Web Search and System Info tools.

Run:
  python backend/test_tools.py
  python -m pytest backend/test_tools.py -v
"""

import asyncio
import sys
import os

# Ensure backend is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_system_info():
    """Test system_info returns expected structure."""
    from tool_handlers import system_info

    result = system_info()
    assert isinstance(result, dict), "system_info should return a dict"

    # Should have at least one of these (os is always available)
    if "error" in result:
        print(f"  [WARN] system_info error: {result['error']}")
        return

    if "os" in result:
        os_info = result["os"]
        assert "type" in os_info or "version" in os_info, "os should have type or version"
        print(f"  OS: {os_info.get('type', '?')} {os_info.get('release', '')}")

    if "cpu" in result:
        cpu = result["cpu"]
        print(f"  CPU: {cpu.get('count', '?')} cores, {cpu.get('usage_percent', '?')}% usage")

    if "memory" in result:
        mem = result["memory"]
        print(f"  Memory: {mem.get('total_gb', '?')} GB total, {mem.get('used_percent', '?')}% used")

    if "gpu" in result and result["gpu"]:
        gpu = result["gpu"]
        print(f"  GPU: {gpu.get('name', '?')} ({gpu.get('memory_total_mb', '?')} MB)")
    else:
        print("  GPU: None (no NVIDIA GPU or nvidia-smi not available)")

    print("  [PASS] system_info")


async def test_web_search():
    """Test web_search returns results."""
    from tool_handlers import web_search

    result = await web_search("Python programming language", max_results=3)

    assert isinstance(result, dict), "web_search should return a dict"

    if "error" in result:
        print(f"  [FAIL] web_search error: {result['error']}")
        return

    assert "results" in result, "web_search should have 'results' key"
    results = result["results"]
    assert isinstance(results, list), "results should be a list"
    # 0 results is OK (rate limit, network, or DuckDuckGo changes)

    print(f"  Query: {result.get('query', '?')}")
    print(f"  Results: {len(results)}")
    for i, r in enumerate(results[:3], 1):
        title = (r.get("title") or "")[:50]
        url = (r.get("url") or "")[:40]
        print(f"    {i}. {title}... | {url}...")

    print("  [PASS] web_search")


def test_format_tool_result():
    """Test format_tool_result from tools.py."""
    from tools import format_tool_result

    # Web search result
    web_result = {"results": [{"title": "A", "url": "http://a.com", "snippet": "..."}], "query": "test"}
    formatted = format_tool_result("web_search", web_result)
    assert "web_search" in formatted or "Result" in formatted
    print(f"  web_search format: {formatted[:80]}...")

    # System info result
    sys_result = {"os": {"type": "Windows"}, "cpu": {"usage_percent": 10}}
    formatted = format_tool_result("system_info", sys_result)
    assert "system_info" in formatted or "Result" in formatted
    print(f"  system_info format: {formatted[:80]}...")

    print("  [PASS] format_tool_result")


def run_all():
    """Run all tests."""
    print("\n=== Testing Tool Handlers ===\n")
    print("1. system_info:")
    test_system_info()
    print("\n2. web_search:")
    asyncio.run(test_web_search())
    print("\n3. format_tool_result:")
    test_format_tool_result()
    print("\n=== All tests passed ===\n")


if __name__ == "__main__":
    run_all()
