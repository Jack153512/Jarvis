"""
CAD Agent - Generates 3D CAD models using build123d.

Uses local LLM (Ollama with Qwen2.5-Coder-14B) to generate Python code
that creates 3D models using the build123d library.

Fully offline, no paid APIs required.
"""

import os
import re
import ast
import asyncio
import subprocess
import sys
import base64
import tempfile
from datetime import datetime
from typing import Optional, Callable, Dict, Any, Tuple, List

from local_llm import LocalLLM, LLMConfig


# System prompt for CAD code generation - uses strict markers
CAD_SYSTEM_PROMPT = """You are a Python CAD expert using the build123d library.
Generate clean, working Python code that creates 3D models.

CRITICAL OUTPUT FORMAT:
You MUST wrap your code between these EXACT markers:
# CAD_CODE_START
<your python code here>
# CAD_CODE_END

CRITICAL REQUIREMENTS:
1. Always start with: from build123d import *
2. Use BuildPart() context manager for solid models
3. Assign final result to: result_part = p.part
4. Export with: export_stl(result_part, 'output.stl')
5. Use conservative fillet/chamfer values (0.5-2mm max)
6. Center models at origin (0,0,0)
7. Use millimeters for all dimensions

CODING GUIDELINES:
- Use lowercase builder methods: extrude(), fillet(), chamfer(), revolve(), loft(), sweep()
- Do NOT use PascalCase for operations like Extrude(), Fillet() - use lowercase
- Keep fillet radii small to avoid geometry errors
- For complex shapes, break into simple operations

EXAMPLE OUTPUT:
# CAD_CODE_START
from build123d import *

with BuildPart() as p:
    Box(20, 20, 10)
    fillet(p.edges().filter_by(Axis.Z), radius=1)

result_part = p.part
export_stl(result_part, 'output.stl')
# CAD_CODE_END

Return ONLY the code between markers. No other text."""

# Required elements that must be present in valid CAD code
REQUIRED_ELEMENTS = [
    "from build123d import",
    "BuildPart()",
    "export_stl"
]


class CadAgent:
    """CAD generation agent using local LLM and build123d.

    Features robust code extraction, validation, and smart retry logic.
    """
    
    def __init__(
        self,
        on_thought: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[Dict], None]] = None
    ):
        """
        Initialize the CAD agent.
        
        Args:
            on_thought: Callback for streaming thought/progress updates
            on_status: Callback for status updates (generating, retrying, etc.)
        """
        self.on_thought = on_thought
        self.on_status = on_status
        
        # Initialize LLM with reasoning-focused settings
        # Using deepseek-r1:14b - 97.3% MATH, superior geometric reasoning for CAD
        # Requires ~9GB VRAM
        config = LLMConfig(
            model="deepseek-r1:14b",
            temperature=0.2,  # Lower temperature for deterministic geometric output
            context_length=32768  # DeepSeek-R1 supports 32K context
        )
        self.llm = LocalLLM(config)
        self.llm.set_system_prompt(CAD_SYSTEM_PROMPT)

        self.client = self.llm
        self.system_prompt = CAD_SYSTEM_PROMPT

    def _normalize_path(self, path: str) -> str:
        return os.path.normpath(os.path.abspath(path))

    def _fs_path(self, path: str) -> str:
        p = self._normalize_path(path)
        if os.name != 'nt':
            return p
        if p.startswith('\\\\?\\') or p.startswith('\\\\.\\'):
            return p
        if len(p) < 240:
            return p
        if p.startswith('\\\\'):
            return '\\\\?\\UNC\\' + p[2:]
        return '\\\\?\\' + p

    def _resolve_work_dir(self, output_dir: Optional[str]) -> str:
        if not output_dir:
            return tempfile.gettempdir()

        try:
            out = self._normalize_path(output_dir)
            os.makedirs(self._fs_path(out), exist_ok=True)
            test_path = os.path.join(out, '._jarvis_path_test')
            with open(self._fs_path(test_path), 'wb') as f:
                f.write(b'ok')
            os.remove(self._fs_path(test_path))
            return out
        except Exception:
            return tempfile.gettempdir()
    
    def _validate_syntax(self, code: str) -> Tuple[bool, str]:
        """
        Validate Python syntax using AST parser.
        
        Args:
            code: Python code to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            error_msg = f"Line {e.lineno}: {e.msg}"
            if e.text:
                error_msg += f" near '{e.text.strip()[:50]}'"
            return False, error_msg
        except Exception as e:
            return False, str(e)
    
    def _check_required_elements(self, code: str) -> Tuple[bool, List[str]]:
        """
        Check if code contains all required elements.
        
        Args:
            code: Python code to check
            
        Returns:
            Tuple of (all_present, missing_elements)
        """
        missing = []
        for element in REQUIRED_ELEMENTS:
            if element not in code:
                missing.append(element)
        return len(missing) == 0, missing
    
    def _categorize_error(self, error_msg: str) -> str:
        """
        Categorize an error for targeted retry prompting.
        
        Args:
            error_msg: Error message from execution
            
        Returns:
            Error category string
        """
        error_lower = error_msg.lower()
        
        if "syntaxerror" in error_lower or "invalid syntax" in error_lower:
            return "syntax"
        elif "nameerror" in error_lower:
            return "undefined_name"
        elif "importerror" in error_lower or "modulenotfounderror" in error_lower:
            return "import"
        elif "typeerror" in error_lower:
            return "type"
        elif "attributeerror" in error_lower:
            return "attribute"
        elif "valueerror" in error_lower:
            return "value"
        elif "no stl" in error_lower or "export" in error_lower:
            return "no_output"
        else:
            return "runtime"
    
    def _get_retry_prompt(self, error_category: str, error_msg: str, original_prompt: str) -> str:
        """
        Generate a targeted retry prompt based on error category.
        
        Args:
            error_category: Category of the error
            error_msg: Full error message
            original_prompt: Original user request
            
        Returns:
            Retry prompt string
        """
        base = f"Original request: {original_prompt}\n\n"
        
        if error_category == "syntax":
            return base + f"""The code has a SYNTAX ERROR:
{error_msg}

Fix the syntax error. Common issues:
- Missing colons after if/for/with/def
- Unmatched parentheses or brackets
- Missing commas in function calls
- Incorrect indentation

Return the COMPLETE corrected code between # CAD_CODE_START and # CAD_CODE_END markers."""

        elif error_category == "undefined_name":
            return base + f"""The code has an UNDEFINED NAME error:
{error_msg}

A variable or function is used before it's defined. Check:
- All variables are defined before use
- All imported functions exist in build123d
- Spelling of variable names

Return the COMPLETE corrected code between # CAD_CODE_START and # CAD_CODE_END markers."""

        elif error_category == "import":
            return base + f"""The code has an IMPORT ERROR:
{error_msg}

Only use: from build123d import *
Do not import other CAD libraries.

Return the COMPLETE corrected code between # CAD_CODE_START and # CAD_CODE_END markers."""

        elif error_category == "attribute":
            return base + f"""The code has an ATTRIBUTE ERROR:
{error_msg}

A method or property doesn't exist. In build123d:
- Use lowercase methods: fillet(), chamfer(), extrude()
- NOT PascalCase: Fillet(), Chamfer(), Extrude()

Return the COMPLETE corrected code between # CAD_CODE_START and # CAD_CODE_END markers."""

        elif error_category == "no_output":
            return base + f"""The code ran but NO STL FILE was created.

Make sure to include at the end:
result_part = p.part
export_stl(result_part, 'output.stl')

Return the COMPLETE corrected code between # CAD_CODE_START and # CAD_CODE_END markers."""

        else:
            return base + f"""The code failed with this error:
{error_msg}

Fix the error and return the COMPLETE corrected code between # CAD_CODE_START and # CAD_CODE_END markers."""
    
    async def generate_prototype(
        self,
        prompt: str,
        output_dir: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a 3D CAD model from a text description.
        Features pre-execution validation and smart retry logic.
        
        Args:
            prompt: Description of the 3D model to create
            output_dir: Directory to save the output files
            
        Returns:
            Dict with 'format', 'data' (base64 STL), 'file_path', or None on failure
        """
        print(f"[CadAgent] Generating: {prompt}")

        work_dir = self._resolve_work_dir(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_stl = self._normalize_path(os.path.join(work_dir, f"output_{timestamp}.stl"))
        script_path = self._normalize_path(os.path.join(work_dir, "current_design.py"))

        max_retries = 3
        current_prompt = f"""Create a 3D model of: {prompt}

Remember to wrap your code between # CAD_CODE_START and # CAD_CODE_END markers."""
        
        last_error = None
        
        for attempt in range(max_retries):
            print(f"[CadAgent] Attempt {attempt + 1}/{max_retries}")
            
            # Emit status
            if self.on_status:
                self.on_status({
                    "status": "generating" if attempt == 0 else "retrying",
                    "attempt": attempt + 1,
                    "max_attempts": max_retries,
                    "error": last_error
                })
            
            # Generate code using local LLM
            raw_response = ""
            try:
                async for chunk in self.llm.chat(current_prompt):
                    raw_response += chunk
                    if self.on_thought:
                        self.on_thought(chunk)
            except Exception as e:
                print(f"[CadAgent] LLM error: {e}")
                last_error = str(e)
                continue
            
            if not raw_response.strip():
                print("[CadAgent] Empty response from LLM")
                last_error = "Empty response from LLM"
                continue
            
            # Clean up the code
            code = self._clean_code(raw_response)
            
            if not code:
                print("[CadAgent] Failed to extract valid code from response")
                last_error = "Failed to extract code from LLM response"
                current_prompt = self._get_retry_prompt("syntax", 
                    "Could not extract valid Python code from your response. Make sure to use # CAD_CODE_START and # CAD_CODE_END markers.",
                    prompt)
                self.llm.reset_conversation()
                continue
            
            # PRE-EXECUTION VALIDATION 1: Check required elements
            has_elements, missing = self._check_required_elements(code)
            if not has_elements:
                print(f"[CadAgent] Missing required elements: {missing}")
                last_error = f"Missing: {', '.join(missing)}"
                current_prompt = self._get_retry_prompt("no_output",
                    f"Missing required elements: {', '.join(missing)}",
                    prompt)
                self.llm.reset_conversation()
                continue
            
            # PRE-EXECUTION VALIDATION 2: Syntax check
            is_valid, syntax_error = self._validate_syntax(code)
            if not is_valid:
                print(f"[CadAgent] Syntax error: {syntax_error}")
                last_error = syntax_error
                if self.on_status:
                    self.on_status({
                        "status": "retrying",
                        "attempt": attempt + 1,
                        "max_attempts": max_retries,
                        "error": f"SyntaxError: {syntax_error}"
                    })
                current_prompt = self._get_retry_prompt("syntax", syntax_error, prompt)
                self.llm.reset_conversation()
                continue
            
            print("[CadAgent] Code passed validation checks")
            
            # Inject correct output path (use forward slashes to avoid unicode escape issues)
            safe_output_path = output_stl.replace("\\", "/")
            # Handle various quote styles
            code = re.sub(r"['\"]output\.stl['\"]", f"'{safe_output_path}'", code)
            code = code.replace("output.stl", f"'{safe_output_path}'")
            
            # Save script
            with open(self._fs_path(script_path), "w", encoding="utf-8") as f:
                f.write(code)
            
            print(f"[CadAgent] Executing: {script_path}")

            # Execute the script
            try:
                proc = await asyncio.to_thread(
                    subprocess.run,
                    [sys.executable, self._fs_path(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if proc.returncode != 0:
                    error_msg = proc.stderr or "Unknown error"
                    
                    # Get last few lines for display
                    error_lines = error_msg.strip().split('\n')
                    short_error = error_lines[-1][:100] if error_lines else "Script error"
                    
                    print(f"[CadAgent] Script failed: {short_error}")
                    last_error = short_error
                    
                    if self.on_status:
                        self.on_status({
                            "status": "retrying",
                            "attempt": attempt + 1,
                            "max_attempts": max_retries,
                            "error": short_error
                        })
                    
                    # Categorize error for smart retry
                    error_category = self._categorize_error(error_msg)
                    current_prompt = self._get_retry_prompt(error_category, error_msg, prompt)
                    
                    # Reset conversation for fresh attempt
                    self.llm.reset_conversation()
                    continue
                
                print("[CadAgent] Script executed successfully")
                
            except subprocess.TimeoutExpired:
                print("[CadAgent] Script timeout")
                last_error = "Script execution timed out"
                current_prompt = self._get_retry_prompt("runtime", 
                    "Script timed out. Simplify the geometry.",
                    prompt)
                self.llm.reset_conversation()
                continue
            except Exception as e:
                print(f"[CadAgent] Execution error: {e}")
                last_error = str(e)
                continue
            
            # Check output file
            if os.path.exists(self._fs_path(output_stl)):
                file_size = os.path.getsize(self._fs_path(output_stl))
                print(f"[CadAgent] STL generated: {output_stl} ({file_size} bytes)")
                
                if file_size < 100:
                    print("[CadAgent] STL file too small, likely invalid")
                    last_error = "Generated STL file is too small (invalid)"
                    current_prompt = self._get_retry_prompt("no_output",
                        "The STL file was created but is empty or invalid. Make sure the geometry is valid.",
                        prompt)
                    self.llm.reset_conversation()
                    continue
                
                with open(self._fs_path(output_stl), "rb") as f:
                    stl_data = f.read()
                        
                b64_stl = base64.b64encode(stl_data).decode('utf-8')
                    
                return {
                    "format": "stl",
                    "data": b64_stl,
                    "file_path": output_stl
                }
            else:
                print(f"[CadAgent] STL not found: {output_stl}")
                last_error = "No STL file was created"
                current_prompt = self._get_retry_prompt("no_output", 
                    "No STL file was created.",
                    prompt)
                self.llm.reset_conversation()
                continue

        # All attempts failed
        print(f"[CadAgent] All {max_retries} attempts failed. Last error: {last_error}")
        if self.on_status:
            self.on_status({
                "status": "failed",
                "attempt": max_retries,
                "max_attempts": max_retries,
                "error": last_error or "All generation attempts failed"
            })

        return None

    async def iterate_prototype(
        self,
        modification: str,
        output_dir: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Iterate on an existing CAD design.
        Features pre-execution validation and smart retry logic.
        
        Args:
            modification: Description of changes to make
            output_dir: Directory containing the existing design
            
        Returns:
            Dict with updated STL data, or None on failure
        """
        print(f"[CadAgent] Iterating: {modification}")

        work_dir = self._resolve_work_dir(output_dir)
        script_path = self._normalize_path(os.path.join(work_dir, "current_design.py"))
        
        # Load existing code
        existing_code = ""
        if os.path.exists(self._fs_path(script_path)):
            with open(self._fs_path(script_path), "r", encoding="utf-8") as f:
                existing_code = f.read()
            
            # Clean up paths in existing code (handle both forward and backslashes)
            existing_code = re.sub(
                r"['\"][A-Za-z]:[/\\][^'\"]+output[^'\"]*\.stl['\"]",
                "'output.stl'",
                existing_code
            )
        
        if not existing_code:
            print("[CadAgent] No existing design, creating new")
            return await self.generate_prototype(modification, output_dir)
        
        # Generate new output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_stl = self._normalize_path(os.path.join(work_dir, f"output_{timestamp}.stl"))
        
        # Create iteration prompt with markers
        iteration_prompt = f"""Modify this build123d script based on the request.

Current code:
# CAD_CODE_START
{existing_code}
# CAD_CODE_END

Modification request: {modification}

Return the COMPLETE modified code between # CAD_CODE_START and # CAD_CODE_END markers.
Keep export_stl at the end."""
        
        max_retries = 3
        current_prompt = iteration_prompt
        last_error = None

        for attempt in range(max_retries):
            print(f"[CadAgent] Iteration attempt {attempt + 1}/{max_retries}")

            if self.on_status:
                self.on_status({
                    "status": "generating" if attempt == 0 else "retrying",
                    "attempt": attempt + 1,
                    "max_attempts": max_retries,
                    "error": last_error
                })

            raw_response = ""
            try:
                async for chunk in self.llm.chat(current_prompt):
                    raw_response += chunk
                    if self.on_thought:
                        self.on_thought(chunk)
            except Exception as e:
                print(f"[CadAgent] LLM error: {e}")
                last_error = str(e)
                continue

            if not raw_response.strip():
                last_error = "Empty response"
                continue

            code = self._clean_code(raw_response)
            if not code:
                last_error = "Failed to extract code"
                current_prompt = self._get_retry_prompt(
                    "syntax",
                    "Could not extract valid code. Use # CAD_CODE_START and # CAD_CODE_END markers.",
                    modification
                )
                self.llm.reset_conversation()
                continue

            is_valid, syntax_error = self._validate_syntax(code)
            if not is_valid:
                print(f"[CadAgent] Syntax error: {syntax_error}")
                last_error = syntax_error
                current_prompt = self._get_retry_prompt("syntax", syntax_error, modification)
                self.llm.reset_conversation()
                continue

            has_elements, missing = self._check_required_elements(code)
            if not has_elements:
                print(f"[CadAgent] Missing elements: {missing}")
                last_error = f"Missing: {', '.join(missing)}"
                current_prompt = self._get_retry_prompt(
                    "no_output",
                    f"Missing: {', '.join(missing)}",
                    modification
                )
                self.llm.reset_conversation()
                continue

            safe_output_path = output_stl.replace("\\", "/")
            code = re.sub(r"['\"]output\.stl['\"]", f"'{safe_output_path}'", code)
            code = code.replace("output.stl", f"'{safe_output_path}'")

            with open(self._fs_path(script_path), "w", encoding="utf-8") as f:
                f.write(code)

            try:
                proc = await asyncio.to_thread(
                    subprocess.run,
                    [sys.executable, self._fs_path(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if proc.returncode != 0:
                    error_msg = proc.stderr or "Unknown error"
                    error_lines = error_msg.strip().split('\n')
                    short_error = error_lines[-1][:100] if error_lines else "Script error"
                    print(f"[CadAgent] Script failed: {short_error}")
                    last_error = short_error
                    error_category = self._categorize_error(error_msg)
                    current_prompt = self._get_retry_prompt(error_category, error_msg, modification)
                    self.llm.reset_conversation()
                    continue

            except subprocess.TimeoutExpired:
                last_error = "Script timeout"
                continue
            except Exception as e:
                print(f"[CadAgent] Execution error: {e}")
                last_error = str(e)
                continue

            if os.path.exists(self._fs_path(output_stl)):
                file_size = os.path.getsize(self._fs_path(output_stl))
                print(f"[CadAgent] Iterated STL: {output_stl} ({file_size} bytes)")

                if file_size < 100:
                    last_error = "Invalid STL (too small)"
                    current_prompt = self._get_retry_prompt(
                        "no_output",
                        "STL file is empty or invalid.",
                        modification
                    )
                    self.llm.reset_conversation()
                    continue

                with open(self._fs_path(output_stl), "rb") as f:
                    stl_data = f.read()

                return {
                    "format": "stl",
                    "data": base64.b64encode(stl_data).decode('utf-8'),
                    "file_path": output_stl
                }

            last_error = "No STL created"
            current_prompt = self._get_retry_prompt(
                "no_output",
                "No STL file was created.",
                modification
            )
            self.llm.reset_conversation()

        print(f"[CadAgent] All {max_retries} iteration attempts failed. Last error: {last_error}")
        if self.on_status:
            self.on_status({
                "status": "failed",
                "attempt": max_retries,
                "max_attempts": max_retries,
                "error": last_error or "All iteration attempts failed"
            })

        return None

    def _clean_code(self, raw_code: str) -> str:
        """
        Extract and clean Python code from LLM output.
        Uses multiple strategies for robust extraction.
        
        Args:
            raw_code: Raw LLM output
            
        Returns:
            Clean Python code, or empty string if no valid code found
        """
        # Step 0: Remove <think>...</think> blocks (DeepSeek-R1 reasoning)
        # This ensures we only process the actual code output
        cleaned_input = re.sub(r'<think>.*?</think>', '', raw_code, flags=re.DOTALL | re.IGNORECASE)
        cleaned_input = cleaned_input.strip()
        
        if not cleaned_input:
            print("[CadAgent] Warning: Response was entirely <think> block, no code found")
            return ""
        
        print(f"[CadAgent] Processing {len(cleaned_input)} chars after removing think blocks")
        
        # Strategy 1: Extract from CAD_CODE markers (preferred)
        marker_match = re.search(
            r'#\s*CAD_CODE_START\s*\n(.*?)#\s*CAD_CODE_END',
            cleaned_input,
            re.DOTALL | re.IGNORECASE
        )
        if marker_match:
            code = marker_match.group(1).strip()
            print("[CadAgent] Extracted code using CAD_CODE markers")
            return code
        
        # Strategy 2: Extract from ```python blocks
        python_match = re.search(r'```python\s*\n?(.*?)```', cleaned_input, re.DOTALL)
        if python_match:
            code = python_match.group(1).strip()
            print("[CadAgent] Extracted code from ```python block")
            return code
        
        # Strategy 3: Extract from generic ``` blocks
        generic_match = re.search(r'```\s*\n?(.*?)```', cleaned_input, re.DOTALL)
        if generic_match:
            code = generic_match.group(1).strip()
            if 'build123d' in code or 'BuildPart' in code:
                print("[CadAgent] Extracted code from ``` block")
                return code
        
        # Strategy 4: Find code starting from 'from build123d'
        if "from build123d import" in cleaned_input:
            # Find the start of the import
            import_pos = cleaned_input.find("from build123d import")
            
            # Take everything from import onwards
            code_section = cleaned_input[import_pos:]
            
            # Try to find where the code ends (look for prose indicators)
            lines = code_section.split('\n')
            code_lines = []
            
            for line in lines:
                stripped = line.strip()
                
                # Skip empty lines but include them in output
                if not stripped:
                    code_lines.append(line)
                    continue
                
                # Check if this looks like code
                is_code = (
                    stripped.startswith('#') or  # Comment
                    stripped.startswith('from ') or
                    stripped.startswith('import ') or
                    stripped.startswith('def ') or
                    stripped.startswith('class ') or
                    stripped.startswith('with ') or
                    stripped.startswith('if ') or
                    stripped.startswith('for ') or
                    stripped.startswith('while ') or
                    stripped.startswith('try:') or
                    stripped.startswith('except') or
                    stripped.startswith('else:') or
                    stripped.startswith('elif ') or
                    stripped.startswith('return ') or
                    stripped.startswith('result_') or
                    stripped.startswith('export_') or
                    '=' in stripped or
                    '(' in stripped or
                    stripped.startswith(' ') or  # Indented
                    stripped.startswith('\t') or  # Tab indented
                    stripped in ['pass', 'break', 'continue']
                )
                
                if is_code:
                    code_lines.append(line)
                else:
                    # Check if it's a short identifier-like line (variable name)
                    if len(stripped) < 30 and stripped.replace('_', '').isalnum():
                        code_lines.append(line)
                    else:
                        # Looks like prose, stop here
                        break
            
            if code_lines:
                code = '\n'.join(code_lines).strip()
                print("[CadAgent] Extracted code from 'from build123d' onwards")
                return code
        
        # Strategy 5: Return the cleaned input if it contains key elements
        if all(elem in cleaned_input for elem in ['build123d', 'BuildPart', 'export_stl']):
            print("[CadAgent] Using cleaned output as code (contains all key elements)")
            return cleaned_input.strip()
        
        print("[CadAgent] Failed to extract valid code")
        return ""


# Test function
async def test_cad_agent():
    """Test the CAD agent."""
    print("[Test] Initializing CAD Agent...")
    
    def on_thought(text):
        print(f"[Thought] {text[:50]}..." if len(text) > 50 else f"[Thought] {text}")
    
    def on_status(status):
        print(f"[Status] {status}")
    
    agent = CadAgent(on_thought=on_thought, on_status=on_status)
    
    print("[Test] Generating a simple cube...")
    result = await agent.generate_prototype("a simple 20mm cube with rounded edges")
    
    if result:
        print(f"[Test] Success! STL size: {len(result['data'])} bytes (base64)")
        print(f"[Test] File: {result['file_path']}")
    else:
        print("[Test] Generation failed")
    
    return result is not None


if __name__ == "__main__":
    asyncio.run(test_cad_agent())
