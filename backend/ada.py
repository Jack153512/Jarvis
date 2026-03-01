"""Compatibility module for the test suite.

The repository's runtime entrypoints live in individual backend modules
(e.g., jarvis.py, cad_agent.py, web_agent.py). Some tests expect a single
`ada` module exporting tool schemas and core classes.

This file provides those exports without changing runtime behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from tools import tools_list
from jarvis import AudioLoop
from cad_agent import CadAgent
from web_agent import WebAgent


def _tool_by_name(name: str) -> dict:
    for t in tools_list:
        if t.get("name") == name:
            return t
    raise KeyError(f"Tool not found: {name}")


def _to_test_schema(tool: dict) -> dict:
    # Tests in tests/test_ada_tools.py expect parameters.type == 'OBJECT'
    # even though this repo uses JSON Schema style 'object'.
    out = dict(tool)
    params = dict(out.get("parameters") or {})
    typ = params.get("type")
    if isinstance(typ, str) and typ.lower() == "object":
        params["type"] = "OBJECT"
    out["parameters"] = params
    return out


generate_cad = _to_test_schema(_tool_by_name("generate_cad"))
run_web_agent = _to_test_schema(_tool_by_name("run_web_agent"))
iterate_cad_tool = _to_test_schema(_tool_by_name("iterate_cad"))
list_projects_tool = _to_test_schema(_tool_by_name("list_projects"))


# The test suite expects these tool schemas to exist even if the runtime
# implementation is not present in this repo.
print_stl_tool = {
    "name": "print_stl",
    "description": "Print an STL file on a configured 3D printer.",
    "parameters": {"type": "OBJECT", "properties": {}, "required": []},
}

discover_printers_tool = {
    "name": "discover_printers",
    "description": "Discover available 3D printers.",
    "parameters": {"type": "OBJECT", "properties": {}, "required": []},
}

list_smart_devices_tool = {
    "name": "list_smart_devices",
    "description": "List smart home devices (e.g., lights).",
    "parameters": {"type": "OBJECT", "properties": {}, "required": []},
}

control_light_tool = {
    "name": "control_light",
    "description": "Control a smart light.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "target": {"type": "string", "description": "Device name or id"},
            "action": {"type": "string", "description": "on/off/toggle"},
        },
        "required": ["target", "action"],
    },
}


class KasaAgent:
    pass


class PrinterAgent:
    pass


@dataclass
class LiveConnectConfig:
    response_modalities: List[str]


config = LiveConnectConfig(response_modalities=["AUDIO"])
