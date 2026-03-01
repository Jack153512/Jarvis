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
