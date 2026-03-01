"""
Local image generation using Hugging Face diffusers.
Model downloads automatically on first use (~1.7 GB, cached to ~/.cache/huggingface/).
No separate server needed — runs directly in the backend process.
"""

# ── Environment flags — MUST be set before any library import ────────────────
# These are read by libraries at import time, not at call time, so they must
# come before every other import in this file.
import os

# tqdm reads TQDM_DISABLE at import time and sets its global default to
# disabled.  This prevents tqdm from ever calling sys.stderr.flush(), which
# raises "OSError: [Errno 22] Invalid argument" when stderr is a Windows named
# pipe (as it is when Python runs as an Electron child process).
os.environ['TQDM_DISABLE'] = '1'

# Suppress TensorFlow's hardware-scanning startup noise before numpy is touched.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import base64
import gc
import io
import sys
import threading
import time
from typing import Optional

# Lazy globals — model loaded once on first generation
_pipe = None
_device = None
# Prevents multiple asyncio.to_thread workers from downloading the model
# simultaneously when several generate requests arrive before the first one
# finishes loading.  All threads after the first will block here and then
# immediately return the already-loaded pipeline.
_load_lock = threading.Lock()


def _get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load_pipeline(model_id: str):
    """Load and optimise the diffusion pipeline (blocking — call via asyncio.to_thread)."""
    global _pipe, _device
    # Fast path — model already loaded, no lock needed.
    if _pipe is not None:
        return _pipe
    # Slow path — acquire the lock so only the first thread downloads the model.
    # Every subsequent thread that arrives here while the download is running
    # will block until the first thread finishes, then return the cached pipe.
    with _load_lock:
        if _pipe is not None:   # re-check after acquiring (double-checked locking)
            return _pipe

        import torch

        _device = _get_device()

        # Cap PyTorch's intra-op CPU thread pool to 4 cores so the OS scheduler
        # has headroom for Electron's rendering thread.  set_num_threads is safe
        # to call from a worker thread and on repeated calls.
        # set_num_interop_threads is intentionally omitted — it may only be called
        # once, strictly before any parallel work begins, and the server's import
        # chain will have already started PyTorch's interop pool by the time the
        # first generate request arrives.  Calling it here risks a deadlock or an
        # unhandled exception variant that bypasses our RuntimeError catch.
        try:
            torch.set_num_threads(4)
        except Exception:
            pass

        # float16 on CUDA cuts VRAM in half and is ~2× faster than float32
        dtype = torch.float16 if _device == "cuda" else torch.float32

        print(f"[ImageGen] Loading model '{model_id}' on {_device} (dtype={dtype})…", flush=True)
        t0 = time.perf_counter()

        try:
            try:
                from diffusers import AutoPipelineForText2Image
                PipelineClass = AutoPipelineForText2Image
            except ImportError:
                from diffusers import DiffusionPipeline
                PipelineClass = DiffusionPipeline

            def _load_from_pretrained(**extra):
                # newer diffusers (≥0.27) renamed torch_dtype → dtype;
                # try the new name first, fall back to the old one so the
                # code works across the version range without warnings.
                try:
                    return PipelineClass.from_pretrained(
                        model_id,
                        dtype=dtype,
                        safety_checker=None,
                        requires_safety_checker=False,
                        **extra,
                    )
                except TypeError:
                    return PipelineClass.from_pretrained(
                        model_id,
                        torch_dtype=dtype,
                        safety_checker=None,
                        requires_safety_checker=False,
                        **extra,
                    )

            # ── Loading strategy (two-tier fallback) ───────────────────────────
            # Primary path: default accelerate loading.  torch_dtype=float16 is
            #   already set above so the weights are cast to float16 after loading,
            #   working correctly with whatever files are already cached locally.
            #   We deliberately do NOT use variant="fp16" here because that would
            #   trigger a fresh ~1.7 GB download of the fp16-specific safetensors
            #   files the first time this code runs, which stalls or crashes the
            #   backend on machines where those files are not pre-cached.
            # Fallback: low_cpu_mem_usage=False — last resort for environments where
            #   accelerate < 0.20 places weights on a "meta" device and .to(device)
            #   raises "Cannot copy out of meta tensor; no data!".
            # ────────────────────────────────────────────────────────────────────
            try:
                _pipe = _load_from_pretrained()
            except RuntimeError as meta_err:
                if "meta" in str(meta_err).lower() or "no data" in str(meta_err).lower():
                    print("[ImageGen] Meta tensor detected — retrying with low_cpu_mem_usage=False", flush=True)
                    _pipe = _load_from_pretrained(low_cpu_mem_usage=False)
                else:
                    raise

            _pipe = _pipe.to(_device)
        except Exception:
            _pipe = None
            raise

        if _device == "cuda":
            import torch.backends.cudnn as cudnn
            # benchmark=True profiles many convolution algorithms on first run and
            # caches the fastest one.  The profiling allocates extra CUDA memory;
            # with RAM already tight we leave it off so there's no extra spike.
            cudnn.benchmark = False

            # Prefer xformers if available; PyTorch 2.0 SDPA is the fallback.
            try:
                import xformers  # noqa: F401
                _pipe.enable_xformers_memory_efficient_attention()
                print("[ImageGen] xformers memory-efficient attention enabled", flush=True)
            except Exception:
                print("[ImageGen] xformers not available — PyTorch SDPA active (torch >= 2.0)", flush=True)

            # DPMSolverMultistep: same quality as DDIM in 10–15 steps vs 20.
            try:
                from diffusers import DPMSolverMultistepScheduler
                _pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    _pipe.scheduler.config
                )
                print("[ImageGen] Scheduler → DPMSolverMultistep (faster convergence)", flush=True)
            except Exception:
                pass

            try:
                _pipe.enable_vae_slicing()
            except Exception:
                pass

        # When the backend runs as an Electron child process, sys.stderr is a
        # Windows named pipe that does not support tqdm's status_printer flush.
        # Disabling the progress bar prevents "OSError: [Errno 22] Invalid argument"
        # from crashing every generation.
        try:
            _pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        # Attention slicing trades speed for lower peak memory.  On CUDA the
        # RTX 4060 Ti has 16 GB VRAM and SD v1.5 only uses ~2 GB, so slicing
        # would just add overhead with no benefit.  On CPU every bit of RAM
        # matters, so enable it unconditionally for the CPU path.
        if _device != "cuda":
            try:
                _pipe.enable_attention_slicing()
            except Exception:
                pass

        elapsed = time.perf_counter() - t0

        try:
            import psutil
            ram = psutil.virtual_memory()
            ram_used = round(ram.used / 1024 ** 3, 1)
            ram_total = round(ram.total / 1024 ** 3, 1)
            ram_info = f"  RAM: {ram_used}/{ram_total} GB"
        except Exception:
            ram_info = ""

        vram_info = ""
        if _device == "cuda":
            try:
                import torch
                alloc = round(torch.cuda.memory_allocated() / 1024 ** 3, 1)
                reserved = round(torch.cuda.memory_reserved() / 1024 ** 3, 1)
                vram_info = f"  VRAM: {alloc} GB alloc / {reserved} GB reserved"
            except Exception:
                pass

        print(f"[ImageGen] Model ready in {elapsed:.1f}s{ram_info}{vram_info}", flush=True)
        return _pipe


_MAX_SAFE_PIXELS = 786_432   # 1024×768 — beyond this SD v1.5 OOMs on CUDA


def _clamp_size(width: int, height: int) -> tuple[int, int]:
    """
    SD v1.5 was trained on 512×512.  At 1024×1024 the UNet attention maps
    require ~4× the VRAM and the CUDA allocator can trigger a hard
    STATUS_ACCESS_VIOLATION (exit 3221225477) that Python cannot catch.

    We keep the requested aspect ratio but scale down so that
    width × height ≤ _MAX_SAFE_PIXELS, then round both dims to the
    nearest multiple of 8 (required by the VAE encoder).
    """
    pixels = width * height
    if pixels <= _MAX_SAFE_PIXELS:
        return width, height

    scale = (_MAX_SAFE_PIXELS / pixels) ** 0.5
    new_w = max(64, (int(width * scale) // 8) * 8)
    new_h = max(64, (int(height * scale) // 8) * 8)
    print(
        f"[ImageGen] Size {width}×{height} exceeds safe limit — "
        f"scaled down to {new_w}×{new_h} to prevent CUDA OOM crash",
        flush=True,
    )
    return new_w, new_h


def _generate_sync(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: Optional[int],
    model_id: str,
) -> dict:
    """Synchronous generation — runs in a thread pool."""
    import torch

    # Clamp to a safe pixel budget before anything else — an OOM inside a
    # CUDA kernel raises a hard C++ exception (exit 3221225477) that bypasses
    # all Python try/except blocks and kills the entire backend process.
    width, height = _clamp_size(width, height)

    # Cap steps to a reasonable upper bound; very high step counts saturate
    # quality and waste GPU time / increase OOM risk.
    if steps > 50:
        steps = 50
        print(f"[ImageGen] steps clamped to 50", flush=True)

    pipe = _load_pipeline(model_id)

    # Enable VAE tiling for images larger than 512×512 so the VAE decoder
    # processes the image in tiles instead of all at once — prevents a second
    # wave of OOM that would occur during decode even if the UNet survived.
    if width > 512 or height > 512:
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass

    generator = None
    if seed is not None:
        generator = torch.Generator(device=_device).manual_seed(int(seed))

    # tqdm (used by diffusers for the progress bar) calls sys.stderr.flush()
    # during __init__.  When Python runs as an Electron child process, stderr
    # is a Windows named pipe that raises "OSError: [Errno 22] Invalid argument"
    # on flush().  Swapping stderr for an in-memory buffer for the duration of
    # the pipeline call is the most reliable fix — it works regardless of tqdm
    # version, diffusers version, or whether a _SafeStream wrapper was applied.
    import io as _io
    _old_stderr = sys.stderr
    sys.stderr = _io.StringIO()
    t0 = time.perf_counter()
    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
    finally:
        sys.stderr = _old_stderr
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Convert image to base64 before releasing CUDA cache so PIL doesn't hold
    # GPU memory while we encode.
    image = result.images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Free PyTorch's CUDA allocator cache and unreferenced Python objects after
    # each generation.  PyTorch holds on to freed blocks in a pool by default;
    # explicit cleanup keeps VRAM use flat between requests and prevents system
    # RAM from filling up with tensors awaiting GC.
    try:
        import torch
        if torch.cuda.is_available():
            # Release cached VRAM blocks back to the CUDA allocator pool so
            # subsequent generations start with a clean slate.
            # ipc_collect() is intentionally omitted — it manages shared IPC
            # handles between processes and is a no-op (or worse, a driver-level
            # fault) in our single-process inference server.
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

    return {
        "format": "png",
        "image_b64": b64,
        "model": model_id,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "size": f"{width}x{height}",
        "seed": seed,
        "timings": {"inference_ms": elapsed_ms},
    }


async def generate_image(
    prompt: str,
    negative_prompt: str = "",
    size: str = "512x512",
    steps: int = 12,
    guidance_scale: float = 7.0,
    seed: Optional[int] = None,
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
) -> dict:
    """Async wrapper — non-blocking, safe to await from FastAPI/socketio handlers."""
    width, height = 512, 512
    for sep in ("x", "*"):
        if sep in size:
            try:
                w, h = map(int, size.split(sep))
                width, height = w, h
                break
            except ValueError:
                pass

    return await asyncio.to_thread(
        _generate_sync,
        prompt,
        negative_prompt,
        width,
        height,
        steps,
        guidance_scale,
        seed,
        model_id,
    )
