"""
Shap-E Based CAD Agent - Neural 3D Generation

This agent uses OpenAI's Shap-E model to generate 3D meshes directly
from text prompts, bypassing unreliable LLM code generation.

Key advantages:
- No LLM hallucination issues
- Direct mesh output (no code execution)
- Fast generation (seconds, not minutes)
- Consistent, reliable results
"""

import os
import sys
import torch
import base64
import asyncio
import tempfile
import shutil
from datetime import datetime
from typing import Optional, Callable, Dict
import re
import gc
from contextlib import nullcontext

# Check for CUDA availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[ShapE] Using device: {DEVICE}")

# Shared model cache (class-level to avoid reloading)
_SHARED_MODELS = {
    'loaded': False,
    'xm': None,
    'model': None,
    'diffusion': None,
    'sample_latents': None,
    'decode_latent_mesh': None,
    'device': None,
}


class ShapECadAgent:
    """
    Neural 3D generation using OpenAI's Shap-E model.
    
    Pipeline:
    1. Text prompt → Shap-E neural network
    2. Neural network → 3D implicit function
    3. Implicit function → Mesh (marching cubes)
    4. Mesh → STL export
    
    No code generation, no execution failures, no hallucination.
    """
    
    def __init__(
        self,
        on_thought: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[Dict], None]] = None,
        on_spec: Optional[Callable[[Dict], None]] = None,
    ):
        self.on_thought = on_thought
        self.on_status = on_status
        self.on_spec = on_spec
        
        self.models_loaded = False
        self.xm = None
        self.model = None
        self.diffusion = None

        self.device = DEVICE

        self.last_error: Optional[str] = None
        
        self.current_attempt = 0
        self.max_attempts = 1  # Shap-E is deterministic, no retry needed
        
    def _emit_status(self, status: str, **kwargs):
        if self.on_status:
            self.on_status({
                "status": status,
                "attempt": self.current_attempt,
                "max_attempts": self.max_attempts,
                **kwargs
            })
    
    def _emit_thought(self, text: str):
        if self.on_thought:
            # Ensure text is safe for console
            safe_text = text.encode('ascii', 'replace').decode('ascii')
            self.on_thought(safe_text)

    def _resolve_work_dir(self, output_dir: Optional[str]) -> str:
        """Resolve a usable directory for writing temp OBJ/STL files.

        On Windows, some paths can raise OSError(22) depending on characters,
        length, or other environmental issues. If output_dir is unusable, we
        fall back to the OS temp directory.
        """
        if not output_dir:
            return tempfile.gettempdir()

        try:
            out = self._normalize_path(output_dir)
            os.makedirs(self._fs_path(out), exist_ok=True)
            test_path = os.path.join(out, "._jarvis_path_test")
            with open(self._fs_path(test_path), "wb") as f:
                f.write(b"ok")
            os.remove(self._fs_path(test_path))
            return out
        except Exception as e:
            fallback = tempfile.gettempdir()
            self.last_error = f"Invalid output_dir '{output_dir}': {type(e).__name__}: {e}. Falling back to '{fallback}'"
            self._emit_thought(f"[WARN] {self.last_error}\n")
            return fallback

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

    def _safe_filename(self, stem: str, ext: str, max_len: int = 48) -> str:
        base = str(stem or '').strip()
        base = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', base)
        base = base.rstrip(' .')
        if not base:
            base = 'cad'
        upper = base.upper()
        reserved = {
            'CON', 'PRN', 'AUX', 'NUL',
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        }
        if upper in reserved:
            base = '_' + base
        base = base[:max_len].rstrip(' .')
        ext = (ext or '').strip()
        if ext and not ext.startswith('.'):
            ext = '.' + ext
        return base + ext

    def _atomic_write_bytes(self, path: str, data: bytes):
        path = self._normalize_path(path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(self._fs_path(parent), exist_ok=True)

        fd = None
        tmp_path = None
        try:
            tmp_dir = self._fs_path(parent) if parent else None
            fd, tmp_path = tempfile.mkstemp(prefix='jarvis_tmp_', dir=tmp_dir)
            with os.fdopen(fd, 'wb') as f:
                fd = None
                f.write(data)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
        except Exception as e:
            try:
                if fd is not None:
                    os.close(fd)
            except Exception:
                pass
            raise RuntimeError(
                f"atomic_write: write_tmp failed: {type(e).__name__}: {e} (tmp_path={tmp_path!r}, path={path!r})"
            )

        try:
            os.replace(self._fs_path(tmp_path), self._fs_path(path))
            return
        except Exception as e:
            try:
                with open(self._fs_path(path), 'wb') as f:
                    f.write(data)
                    f.flush()
            except Exception as e2:
                raise RuntimeError(
                    f"atomic_write: replace failed: {type(e).__name__}: {e} (tmp_path={tmp_path!r}, path={path!r}); "
                    f"direct_overwrite failed: {type(e2).__name__}: {e2}"
                )
            finally:
                try:
                    if tmp_path and os.path.exists(self._fs_path(tmp_path)):
                        os.remove(self._fs_path(tmp_path))
                except Exception:
                    pass
    
    async def _load_models(self):
        """Load Shap-E models (done once, shared across all instances)."""
        global _SHARED_MODELS
        
        # Check if already loaded (by this or another instance)
        shared_device = _SHARED_MODELS.get('device')
        if _SHARED_MODELS['loaded'] and shared_device == str(self.device):
            self.xm = _SHARED_MODELS['xm']
            self.model = _SHARED_MODELS['model']
            self.diffusion = _SHARED_MODELS['diffusion']
            self._sample_latents = _SHARED_MODELS['sample_latents']
            self._decode_latent_mesh = _SHARED_MODELS['decode_latent_mesh']
            self.models_loaded = True
            
            # Verify models are actually valid
            if self.model is None or self.xm is None:
                print("[ShapE] WARNING: Shared models were None, reloading...")
                _SHARED_MODELS['loaded'] = False
            else:
                self._emit_thought("[SHAP-E] Using pre-loaded models\n")
                return True
        
        if self.device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        self._emit_thought(f"[SHAP-E] Loading neural models on {self.device}...\n")
        
        def _is_cuda_oom(err: Exception) -> bool:
            if isinstance(err, torch.cuda.OutOfMemoryError):
                return True
            msg = str(err).lower()
            return 'cuda' in msg and 'out of memory' in msg

        async def _try_load_on_device() -> bool:
            try:
                # Import Shap-E components
                from shap_e.diffusion.sample import sample_latents
                from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
                from shap_e.models.download import load_model, load_config
                from shap_e.util.notebooks import decode_latent_mesh

                # Load models
                self._emit_thought("[SHAP-E] Loading transmitter model...\n")
                xm = load_model('transmitter', device=self.device)

                self._emit_thought("[SHAP-E] Loading text300M model...\n")
                model = load_model('text300M', device=self.device)

                if self.device.type == 'cuda':
                    try:
                        xm = xm.half()
                    except Exception:
                        pass
                    try:
                        model = model.half()
                    except Exception:
                        pass

                self._emit_thought("[SHAP-E] Loading diffusion config...\n")
                diffusion = diffusion_from_config(load_config('diffusion'))

                _SHARED_MODELS['xm'] = xm
                _SHARED_MODELS['model'] = model
                _SHARED_MODELS['diffusion'] = diffusion
                _SHARED_MODELS['sample_latents'] = sample_latents
                _SHARED_MODELS['decode_latent_mesh'] = decode_latent_mesh
                _SHARED_MODELS['loaded'] = True
                _SHARED_MODELS['device'] = str(self.device)

                self.xm = xm
                self.model = model
                self.diffusion = diffusion
                self._sample_latents = sample_latents
                self._decode_latent_mesh = decode_latent_mesh
                self.models_loaded = True

                self._emit_thought("[SHAP-E] Models loaded successfully!\n")
                self.last_error = None
                return True

            except Exception as e:
                # Always try to aggressively release memory on failures.
                try:
                    self.xm = None
                    self.model = None
                    self.diffusion = None
                except Exception:
                    pass
                try:
                    _SHARED_MODELS['loaded'] = False
                    _SHARED_MODELS['xm'] = None
                    _SHARED_MODELS['model'] = None
                    _SHARED_MODELS['diffusion'] = None
                    _SHARED_MODELS['sample_latents'] = None
                    _SHARED_MODELS['decode_latent_mesh'] = None
                    _SHARED_MODELS['device'] = None
                except Exception:
                    pass
                try:
                    gc.collect()
                except Exception:
                    pass
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

                if _is_cuda_oom(e):
                    raise

                self.last_error = f"Failed to load Shap-E models: {type(e).__name__}: {e}"
                self._emit_thought(f"[ERROR] {self.last_error}\n")
                return False

        try:
            return await _try_load_on_device()
        except Exception as e:
            if self.device.type == 'cuda' and _is_cuda_oom(e):
                self.last_error = "CUDA out of memory while loading Shap-E models. Falling back to CPU."
                self._emit_thought(f"[WARN] {self.last_error}\n")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                self.device = torch.device('cpu')
                return await _try_load_on_device()

            self.last_error = f"Failed to load Shap-E models: {type(e).__name__}: {e}"
            self._emit_thought(f"[ERROR] {self.last_error}\n")
            return False
    
    async def generate_prototype(
        self,
        prompt: str,
        output_dir: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Generate 3D model from text using Shap-E neural network.
        
        Args:
            prompt: Text description of the 3D object
            output_dir: Directory to save output STL
            
        Returns:
            Dict with format, data (base64), file_path, or None on failure
        """
        self._emit_thought(f"[SHAP-E] Generating: {prompt}\n")
        self._emit_status("generating")
        self.current_attempt = 1
        self.last_error = None
        
        # Load models if needed
        if not await self._load_models():
            if not self.last_error:
                self.last_error = "Failed to load Shap-E models"
            self._emit_status("failed", error=self.last_error)
            return None
        
        work_dir = tempfile.mkdtemp(prefix="jarvis_cad_")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stl_name = self._safe_filename(f"s_{timestamp}", "stl")
        obj_name = self._safe_filename(f"s_{timestamp}", "obj")
        output_stl = self._normalize_path(os.path.join(work_dir, stl_name))
        output_obj = self._normalize_path(os.path.join(work_dir, obj_name))

        requested_output_dir = self._resolve_work_dir(output_dir) if output_dir else None
        requested_output_dir = self._normalize_path(requested_output_dir) if requested_output_dir else None
        requested_stl = self._normalize_path(os.path.join(requested_output_dir, stl_name)) if requested_output_dir else None
        requested_obj = self._normalize_path(os.path.join(requested_output_dir, obj_name)) if requested_output_dir else None
        
        try:
            # Generate latents from text
            self._emit_thought("[SHAP-E] Sampling latent space...\n")
            
            # Run generation in thread to avoid blocking
            latents = await asyncio.to_thread(
                self._generate_latents,
                prompt
            )

            if latents is None and self.device.type == 'cuda':
                last = (self.last_error or '').lower()
                if 'out of memory' in last or 'cuda out of memory' in last:
                    self._emit_thought("[SHAP-E] CUDA OOM detected during sampling; retrying on CPU...\n")
                    try:
                        self.device = torch.device('cpu')
                        await self._load_models()
                    except Exception as e:
                        self.last_error = f"CPU fallback model load failed: {type(e).__name__}: {e}"
                        self._emit_thought(f"[ERROR] {self.last_error}\n")
                    else:
                        latents = await asyncio.to_thread(self._generate_latents, prompt)
            
            if latents is None:
                error_msg = "Latent generation failed - check console for details"
                self.last_error = error_msg
                self._emit_thought(f"[ERROR] {error_msg}\n")
                self._emit_status("failed", error=error_msg)
                return None
            
            self._emit_thought("[SHAP-E] Decoding to mesh...\n")
            
            # Decode latent to mesh
            mesh = await asyncio.to_thread(
                self._decode_mesh,
                latents
            )

            if mesh is None and self.device.type == 'cuda':
                last = (self.last_error or '').lower()
                if 'out of memory' in last or 'cuda out of memory' in last:
                    self._emit_thought("[SHAP-E] CUDA OOM detected during decoding; retrying on CPU...\n")
                    try:
                        self.device = torch.device('cpu')
                        await self._load_models()
                    except Exception as e:
                        self.last_error = f"CPU fallback model load failed: {type(e).__name__}: {e}"
                        self._emit_thought(f"[ERROR] {self.last_error}\n")
                    else:
                        latents = await asyncio.to_thread(self._generate_latents, prompt)
                        if latents is not None:
                            mesh = await asyncio.to_thread(self._decode_mesh, latents)
            
            if mesh is None:
                self.last_error = "Mesh decoding failed"
                self._emit_status("failed", error=self.last_error)
                return None
            
            # Export to OBJ first (Shap-E native format)
            self._emit_thought("[SHAP-E] Exporting mesh...\n")
            
            await asyncio.to_thread(
                self._export_mesh,
                mesh,
                output_obj,
                output_stl
            )
            
            # Release GPU-heavy intermediates early
            try:
                del latents
            except Exception:
                pass
            try:
                gc.collect()
            except Exception:
                pass
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            # Check if STL was created
            if os.path.exists(self._fs_path(output_stl)):
                size = os.path.getsize(self._fs_path(output_stl))
                self._emit_thought(f"[SHAP-E] Success! STL created: {size} bytes\n")

                final_stl_path = output_stl
                final_obj_path = output_obj

                if requested_output_dir and requested_stl:
                    try:
                        os.makedirs(self._fs_path(requested_output_dir), exist_ok=True)
                        try:
                            shutil.copy2(self._fs_path(output_stl), self._fs_path(requested_stl))
                        except OSError:
                            with open(self._fs_path(output_stl), 'rb') as f:
                                self._atomic_write_bytes(requested_stl, f.read())
                        final_stl_path = requested_stl
                        if os.path.exists(self._fs_path(output_obj)) and requested_obj:
                            try:
                                shutil.copy2(self._fs_path(output_obj), self._fs_path(requested_obj))
                            except OSError:
                                with open(self._fs_path(output_obj), 'rb') as f:
                                    self._atomic_write_bytes(requested_obj, f.read())
                            final_obj_path = requested_obj
                    except Exception as e:
                        self._emit_thought(f"[WARN] Could not copy CAD artifacts to output_dir: {type(e).__name__}: {e}\n")

                with open(self._fs_path(final_stl_path), "rb") as f:
                    stl_data = f.read()

                self._emit_status("completed")
                self.last_error = None

                return {
                    "format": "stl",
                    "data": base64.b64encode(stl_data).decode('utf-8'),
                    "file_path": final_stl_path,
                    "engine": "shap-e"
                }
            else:
                self.last_error = "STL export failed"
                self._emit_status("failed", error=self.last_error)
                return None
                
        except Exception as e:
            self.last_error = f"Generation failed: {type(e).__name__}: {e} (work_dir={work_dir!r}, output_obj={output_obj!r}, output_stl={output_stl!r}, requested_output_dir={requested_output_dir!r})"
            self._emit_thought(f"[ERROR] {self.last_error}\n")
            self._emit_status("failed", error=self.last_error)
            return None
    
    def _generate_latents(self, prompt: str):
        """Generate latent representation from text (runs in thread)."""
        try:
            # Clear CUDA cache before generation to avoid OOM
            if self.device.type == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            batch_size = 1
            guidance_scale = float(os.environ.get('SHAPE_GUIDANCE_SCALE', '12.0'))
            karras_steps = int(os.environ.get('SHAPE_KARRAS_STEPS', '32'))
            
            print(f"[ShapE] Generating latents for: {prompt}")
            print(f"[ShapE] Device: {self.device}, Models loaded: {self.model is not None}")
            
            # Check if models are loaded
            if self.model is None or self._sample_latents is None:
                msg = "Models not loaded"
                self.last_error = msg
                print(f"[ShapE] ERROR: {msg}!")
                return None
            
            amp_ctx = nullcontext()
            if self.device.type == 'cuda':
                try:
                    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
                except Exception:
                    amp_ctx = nullcontext()

            with torch.inference_mode():
                with amp_ctx:
                    latents = self._sample_latents(
                        batch_size=batch_size,
                        model=self.model,
                        diffusion=self.diffusion,
                        guidance_scale=guidance_scale,
                        model_kwargs=dict(texts=[prompt] * batch_size),
                        progress=True,
                        clip_denoised=True,
                        use_fp16=(self.device.type == 'cuda'),
                        use_karras=True,
                        karras_steps=karras_steps,
                        sigma_min=1e-3,
                        sigma_max=160,
                        s_churn=0,
                    )
            
            print(f"[ShapE] Latents generated successfully, shape: {latents.shape if hasattr(latents, 'shape') else 'N/A'}")
            return latents
            
        except torch.cuda.OutOfMemoryError as e:
            self.last_error = "CUDA out of memory during latent generation"
            print(f"[ShapE] CUDA Out of Memory! Falling back to CPU if possible.")
            print(f"[ShapE] Error: {e}")
            # Try to recover
            torch.cuda.empty_cache()
            return None
        except Exception as e:
            msg = str(e).lower()
            if self.device.type == 'cuda' and ('cuda' in msg and 'out of memory' in msg):
                self.last_error = "CUDA out of memory during latent generation"
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                return None
            self.last_error = f"Latent generation error: {type(e).__name__}: {e}"
            print(f"[ShapE] Latent generation error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _decode_mesh(self, latents):
        """Decode latent to mesh (runs in thread)."""
        try:
            amp_ctx = nullcontext()
            if self.device.type == 'cuda':
                try:
                    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
                except Exception:
                    amp_ctx = nullcontext()

            with torch.inference_mode():
                with amp_ctx:
                    mesh = self._decode_latent_mesh(self.xm, latents[0]).tri_mesh()
            return mesh
        except torch.cuda.OutOfMemoryError as e:
            self.last_error = "CUDA out of memory during mesh decoding"
            print("[ShapE] CUDA OOM during mesh decoding")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return None
        except Exception as e:
            msg = str(e).lower()
            if self.device.type == 'cuda' and ('cuda' in msg and 'out of memory' in msg):
                self.last_error = "CUDA out of memory during mesh decoding"
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                return None
            self.last_error = f"Mesh decode error: {e}"
            print(f"[ShapE] Mesh decode error: {e}")
            return None
    
    def _export_mesh(self, mesh, obj_path: str, stl_path: str):
        """Export mesh to OBJ and STL using trimesh for proper format."""
        import trimesh
        import io
        
        try:
            obj_path = self._normalize_path(obj_path)
            stl_path = self._normalize_path(stl_path)

            obj_fs_path = self._fs_path(obj_path)
            stl_fs_path = self._fs_path(stl_path)

            tri_mesh = None
            verts = getattr(mesh, 'verts', None)
            faces = getattr(mesh, 'faces', None)
            if verts is not None and faces is not None:
                try:
                    tri_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                except Exception:
                    tri_mesh = None

            obj_bytes = None
            if tri_mesh is None:
                try:
                    raw = io.BytesIO()
                    txt = io.TextIOWrapper(raw, encoding='utf-8', newline='\n')
                    mesh.write_obj(txt)
                    txt.flush()
                    obj_bytes = raw.getvalue()
                    self._atomic_write_bytes(obj_path, obj_bytes)
                except Exception:
                    obj_bytes = None

                if obj_bytes is None:
                    try:
                        mesh.write_obj(obj_fs_path)
                    except Exception:
                        with open(obj_fs_path, 'w', encoding='utf-8', newline='\n') as f:
                            mesh.write_obj(f)

                    tri_mesh = trimesh.load(obj_fs_path, force='mesh')
                else:
                    tri_mesh = trimesh.load(file_obj=io.BytesIO(obj_bytes), file_type='obj', force='mesh')

            if not hasattr(tri_mesh, 'export'):
                raise RuntimeError("Could not construct trimesh mesh for export")

            try:
                if obj_bytes is None:
                    obj_bytes = tri_mesh.export(file_type='obj')
                    if isinstance(obj_bytes, str):
                        obj_bytes = obj_bytes.encode('utf-8')
                    self._atomic_write_bytes(obj_path, obj_bytes)
            except Exception as e:
                print(f"[ShapE] [WARN] OBJ export failed: {type(e).__name__}: {e}")

            stl_bytes = tri_mesh.export(file_type='stl')
            if isinstance(stl_bytes, str):
                stl_bytes = stl_bytes.encode('utf-8')
            self._atomic_write_bytes(stl_path, stl_bytes)

            if hasattr(tri_mesh, 'faces'):
                print(f"[ShapE] Exported STL with {len(tri_mesh.faces)} triangles")
            else:
                print(f"[ShapE] Exported STL")
                
        except Exception as e:
            self.last_error = f"Export error: {type(e).__name__}: {e} (obj_path={obj_path!r}, stl_path={stl_path!r})"
            print(f"[ShapE] Export error: {e}")
            # Fallback: try direct PLY export if available
            try:
                ply_path = obj_path.replace('.obj', '.ply')
                ply_path = self._normalize_path(ply_path)
                ply_fs_path = self._fs_path(ply_path)
                with open(ply_fs_path, 'wb') as f:
                    mesh.write_ply(f)
                tri_mesh = trimesh.load(ply_fs_path, force='mesh')
                stl_bytes = tri_mesh.export(file_type='stl')
                if isinstance(stl_bytes, str):
                    stl_bytes = stl_bytes.encode('utf-8')
                self._atomic_write_bytes(stl_path, stl_bytes)
                print(f"[ShapE] Exported STL via PLY fallback")
            except Exception as e2:
                self.last_error = f"Export error: {type(e).__name__}: {e} (obj_path={obj_path!r}, stl_path={stl_path!r}); PLY fallback failed: {type(e2).__name__}: {e2}"
                print(f"[ShapE] PLY fallback also failed: {e2}")
    
    async def iterate_prototype(
        self,
        modification: str,
        output_dir: Optional[str] = None,
        existing_spec: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Modify existing design - regenerates from new prompt."""
        self._emit_thought(f"[SHAP-E] Iterating: {modification}\n")
        return await self.generate_prototype(modification, output_dir)


# Alias for compatibility with existing code
TwoStageCadAgent = ShapECadAgent
ParametricCadAgent = ShapECadAgent


# =============================================================================
# TEST
# =============================================================================

async def test_shape():
    """Test Shap-E generation."""
    print("=" * 60)
    print("Testing Shap-E Neural 3D Generation")
    print("=" * 60)
    
    def on_thought(text):
        print(text, end='', flush=True)
    
    def on_status(status):
        print(f"\n[STATUS] {status}")
    
    agent = ShapECadAgent(
        on_thought=on_thought,
        on_status=on_status
    )
    
    test_prompts = [
        "a simple robot",
        "a coffee mug with handle",
        "a wooden chair"
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"TEST: {prompt}")
        print("=" * 60)
        
        result = await agent.generate_prototype(prompt)
        
        if result:
            size_kb = len(result['data']) / 1024
            print(f"\n[SUCCESS] {size_kb:.1f} KB STL generated")
            print(f"File: {result.get('file_path', 'N/A')}")
        else:
            print(f"\n[FAILED] Could not generate {prompt}")
    
    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    asyncio.run(test_shape())

