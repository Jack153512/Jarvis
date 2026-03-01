import sys
import types
import importlib
import copy
import os

import pytest


def _install_stub(fullname: str, **attrs):
    mod = types.ModuleType(fullname)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[fullname] = mod
    return mod


def _import_jarvis_with_stubs():
    # jarvis.py imports some optional/heavy deps. Stub them so tests can run
    # even if those libs are not installed in the test environment.
    if "cv2" not in sys.modules:
        _install_stub("cv2")

    if "mss" not in sys.modules:
        _install_stub("mss")

    if "pyaudio" not in sys.modules:
        class _PyAudio:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("PyAudio stub: not available in tests")

        _install_stub("pyaudio", paInt16=8, PyAudio=_PyAudio)

    if "PIL" not in sys.modules and "PIL.Image" not in sys.modules:
        pil = _install_stub("PIL")
        pil_image = _install_stub("PIL.Image")
        pil.Image = pil_image

    if "jarvis" in sys.modules:
        return importlib.reload(sys.modules["jarvis"])
    return importlib.import_module("jarvis")


def _make_audio_loop_shell(jarvis_module, assistant_name="JARVIS"):
    loop = jarvis_module.AudioLoop.__new__(jarvis_module.AudioLoop)
    loop.assistant_name = assistant_name
    return loop


def test_extract_user_name_english():
    jarvis = _import_jarvis_with_stubs()
    loop = _make_audio_loop_shell(jarvis)

    assert loop._extract_user_name("My name is Alex.") == "Alex"
    assert loop._extract_user_name("Call me John Smith") == "John Smith"
    assert loop._extract_user_name("I'm Mary") == "Mary"


def test_extract_user_name_vietnamese():
    jarvis = _import_jarvis_with_stubs()
    loop = _make_audio_loop_shell(jarvis)

    assert loop._extract_user_name("Tôi là An") == "An"
    assert loop._extract_user_name("Tên tôi là Huy") == "Huy"


def test_extract_user_name_rejects_jarvis_or_assistant_name():
    jarvis = _import_jarvis_with_stubs()
    loop = _make_audio_loop_shell(jarvis, assistant_name="JARVIS")

    assert loop._extract_user_name("My name is Jarvis") is None
    assert loop._extract_user_name("I'm J.A.R.V.I.S") is None
    assert loop._extract_user_name("Call me JARVIS") is None


@pytest.mark.asyncio
async def test_persist_identity_update_sanitizes_and_snapshots(monkeypatch):
    # Import server in a way that avoids reading/writing the real backend/settings.json
    if "server" in sys.modules:
        del sys.modules["server"]

    orig_exists = os.path.exists
    os.path.exists = lambda _p: False
    try:
        server = importlib.import_module("server")
    finally:
        os.path.exists = orig_exists

    server.SETTINGS = copy.deepcopy(server.DEFAULT_SETTINGS)

    saved = []

    def _capture_snapshot(snapshot):
        saved.append(copy.deepcopy(snapshot))

    monkeypatch.setattr(server, "save_settings_snapshot", _capture_snapshot)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)

    await server._persist_identity_update({"user_name": "Alex"})
    assert server.SETTINGS["identity"]["user_name"] == "Alex"
    assert saved and saved[-1]["identity"]["user_name"] == "Alex"

    await server._persist_identity_update({"user_name": "Jarvis"})
    assert server.SETTINGS["identity"]["user_name"] == server.DEFAULT_SETTINGS["identity"]["user_name"]
    assert saved[-1]["identity"]["user_name"] == server.DEFAULT_SETTINGS["identity"]["user_name"]
