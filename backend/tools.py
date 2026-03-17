"""
A.D.A Tool Definitions for Local LLM.

These tool definitions are used to help the LLM understand
what capabilities are available for it to use.
"""

tools_list = [
    {
        "name": "generate_cad",
        "description": "Generate a 3D CAD model from a text description using build123d. Creates parametric 3D models that can be exported as STL files.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Detailed description of the 3D model to create (e.g., 'a 30mm cube with 2mm rounded edges')"
                }
            },
            "required": ["prompt"]
        }
    },
    {
        "name": "iterate_cad",
        "description": "Modify an existing CAD design based on a description. Use this to make changes to the current 3D model.",
    "parameters": {
            "type": "object",
        "properties": {
            "prompt": {
                    "type": "string",
                    "description": "Description of the changes to make (e.g., 'add a 10mm hole in the center')"
            }
        },
        "required": ["prompt"]
    }
    },
    {
        "name": "run_web_agent",
        "description": "Control a web browser to perform tasks like searching, navigating websites, and extracting information.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Task to perform in the browser (e.g., 'search for Python build123d documentation')"
                }
            },
            "required": ["prompt"]
        }
    },
    {
    "name": "write_file",
        "description": "Write content to a file in the current project directory.",
    "parameters": {
            "type": "object",
        "properties": {
            "path": {
                    "type": "string",
                    "description": "Relative path for the file (e.g., 'notes/design.txt')"
            },
            "content": {
                    "type": "string",
                    "description": "Content to write to the file"
            }
        },
        "required": ["path", "content"]
    }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file.",
    "parameters": {
            "type": "object",
        "properties": {
            "path": {
                    "type": "string",
                    "description": "Path to the file to read"
            }
        },
        "required": ["path"]
    }
    },
    {
        "name": "read_directory",
        "description": "List the contents of a directory.",
    "parameters": {
            "type": "object",
        "properties": {
            "path": {
                    "type": "string",
                    "description": "Path to the directory to list"
            }
        },
        "required": ["path"]
    }
    },
    {
        "name": "create_project",
        "description": "Create a new project directory for organizing work.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name for the new project"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "switch_project",
        "description": "Switch to a different project directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the project to switch to"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "list_projects",
        "description": "List all available projects.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
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
    },
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
]


def get_tools_prompt() -> str:
    """
    Generate a prompt describing available tools for the LLM.
    
    Returns:
        String describing all available tools
    """
    lines = ["You have access to the following tools:", ""]
    
    for tool in tools_list:
        lines.append(f"**{tool['name']}**: {tool['description']}")
        
        if tool.get('parameters', {}).get('properties'):
            params = tool['parameters']['properties']
            required = tool['parameters'].get('required', [])
            
            param_lines = []
            for name, info in params.items():
                req_marker = "*" if name in required else ""
                param_lines.append(f"  - {name}{req_marker}: {info.get('description', '')}")
            
            if param_lines:
                lines.extend(param_lines)
        
        lines.append("")
    
    lines.append("To use a tool, respond with JSON: {\"tool\": \"tool_name\", \"args\": {...}}")
    
    return "\n".join(lines)


# Additional tools defined elsewhere (ada.py, server) — include in permissions sync
EXTRA_TOOL_NAMES = ("print_stl", "discover_printers", "control_light", "list_smart_devices", "create_directory")


def get_all_tool_names() -> list:
    """Return all tool names (from tools_list + extras) for permissions sync."""
    names = [t["name"] for t in tools_list]
    for name in EXTRA_TOOL_NAMES:
        if name not in names:
            names.append(name)
    return names


def get_default_tool_permissions() -> dict:
    """Return default permissions for all tools. New tools default to False for safety."""
    defaults = {
        "generate_cad": True,
        "iterate_cad": False,
        "run_web_agent": True,
        "write_file": False,
        "read_file": True,
        "read_directory": True,
        "create_project": True,
        "switch_project": True,
        "list_projects": True,
        "web_search": True,
        "system_info": True,
        "print_stl": False,
        "discover_printers": False,
        "control_light": False,
        "list_smart_devices": False,
        "create_directory": False,
    }
    # Ensure tools_list tools have entries
    for t in tools_list:
        name = t.get("name")
        if name and name not in defaults:
            defaults[name] = False
    return defaults


def sync_tool_permissions(perms: dict) -> dict:
    """Merge perms with defaults so all known tools have an entry. Returns new dict."""
    defaults = get_default_tool_permissions()
    out = dict(defaults)
    out.update(perms)
    # Only keep keys that are known tools
    all_names = set(get_all_tool_names()) | set(defaults)
    return {k: v for k, v in out.items() if k in all_names}


def tools_to_ollama_format(tools: list = None) -> list:
    """
    Convert tools_list to Ollama API format for native tool calling.
    Models like Qwen3 support this; Qwen2.5 may not.
    """
    source = tools or tools_list
    out = []
    for t in source:
        name = t.get("name")
        if not name:
            continue
        params = t.get("parameters") or {}
        if isinstance(params.get("type"), str) and params["type"].lower() != "object":
            params = {"type": "object", "properties": params.get("properties", {}), "required": params.get("required", [])}
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": t.get("description", ""),
                "parameters": params,
            },
        })
    return out


def format_tool_result(tool_name: str, result: any) -> str:
    """
    Format a tool result for the LLM.
    
    Args:
        tool_name: Name of the tool that was executed
        result: Result from the tool execution
        
    Returns:
        Formatted string for the LLM
    """
    if result is None:
        return f"[{tool_name}] Completed successfully."
    elif isinstance(result, dict):
        import json
        return f"[{tool_name}] Result:\n{json.dumps(result, indent=2)}"
    elif isinstance(result, (list, tuple)):
        items = "\n".join(f"  - {item}" for item in result)
        return f"[{tool_name}] Results:\n{items}"
    else:
        return f"[{tool_name}] {result}"
