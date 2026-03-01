import importlib
import os
import sys
import copy

import pytest


@pytest.mark.asyncio
async def test_sqlite_identity_survives_restart_and_loads_into_server_settings(tmp_path, monkeypatch):
    db_path = tmp_path / "mem.db"

    # Write identity into SQLite memory store
    if "memory" in sys.modules:
        del sys.modules["memory"]
    memory = importlib.import_module("memory")
    mem = memory.JarvisMemory(str(db_path))
    mem.set_user_name("Alex")

    # Import server with DB path override and without loading real settings.json
    if "server" in sys.modules:
        del sys.modules["server"]

    monkeypatch.setenv("JARVIS_MEMORY_DB_PATH", str(db_path))

    orig_exists = os.path.exists
    os.path.exists = lambda _p: False
    try:
        server = importlib.import_module("server")
    finally:
        os.path.exists = orig_exists

    # Ensure the server loaded name from SQLite into SETTINGS at import/init
    assert isinstance(server.SETTINGS, dict)
    assert "identity" in server.SETTINGS
    assert server.SETTINGS["identity"]["user_name"] == "Alex"

    # Simulate a restart by re-importing server from scratch and verifying it still loads
    if "server" in sys.modules:
        del sys.modules["server"]

    os.path.exists = lambda _p: False
    try:
        server2 = importlib.import_module("server")
    finally:
        os.path.exists = orig_exists

    assert server2.SETTINGS["identity"]["user_name"] == "Alex"
