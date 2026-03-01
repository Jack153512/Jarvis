"""Regression tests for image_gen.py.

Each test class is named after the bug it protects against so failures are
immediately self-documenting.

Run:
    python -m pytest backend/test_image_gen.py -v
    python backend/test_image_gen.py           (no pytest required)
"""

import asyncio
import base64
import io
import os
import struct
import sys
import unittest
import zlib
from unittest.mock import MagicMock, call, patch

# Allow `import image_gen` regardless of where pytest is launched from.
sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reload():
    """Return a fresh image_gen module (resets _pipe / _device globals)."""
    import importlib
    import image_gen as ig
    importlib.reload(ig)
    return ig


def _make_tiny_png() -> bytes:
    """Build a valid 1×1 white PNG without Pillow."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    sig  = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    idat = chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))   # filter + RGB white
    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


class _FakePilImage:
    """Minimal stand-in for a Pillow Image — just writes a real PNG to the buffer."""
    def save(self, buf, format="PNG"):   # noqa: A002
        buf.write(_make_tiny_png())


def _fake_pipe_call(prompt, **kwargs):
    result = MagicMock()
    result.images = [_FakePilImage()]
    return result


# ---------------------------------------------------------------------------
# 1. Import-level regressions
#    Catches: "cannot import name 'StableDiffusionPipeline' from 'diffusers'"
# ---------------------------------------------------------------------------

class TestDiffusersImports(unittest.TestCase):
    """Verifies that the exact names image_gen.py uses can be imported."""

    def test_auto_pipeline_importable(self):
        """AutoPipelineForText2Image must exist in the installed diffusers."""
        try:
            from diffusers import AutoPipelineForText2Image  # noqa: F401
        except ImportError as exc:
            self.fail(
                f"diffusers import failed: {exc}\n"
                "Fix: pip install --upgrade diffusers"
            )

    def test_stable_diffusion_pipeline_not_imported_directly(self):
        """Regression: image_gen.py must NOT import StableDiffusionPipeline directly.

        'StableDiffusionPipeline' is not exported by all diffusers versions and
        caused a hard crash at runtime. AutoPipelineForText2Image (with
        DiffusionPipeline as fallback) is the stable replacement.
        """
        import image_gen as ig
        with open(ig.__file__, encoding="utf-8") as fh:
            source = fh.read()

        self.assertNotIn(
            "from diffusers import StableDiffusionPipeline",
            source,
            "StableDiffusionPipeline is not reliably exported. "
            "Use AutoPipelineForText2Image / DiffusionPipeline instead.",
        )

    def test_diffusion_pipeline_fallback_present_in_source(self):
        """DiffusionPipeline fallback must exist for environments where
        AutoPipelineForText2Image is unavailable (older install, PATH mismatch)."""
        import image_gen as ig
        with open(ig.__file__, encoding="utf-8") as fh:
            source = fh.read()
        self.assertIn(
            "DiffusionPipeline",
            source,
            "DiffusionPipeline fallback not found in image_gen.py",
        )

    def test_torch_importable(self):
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            self.fail(f"torch not installed: {exc}\nFix: pip install torch")


# ---------------------------------------------------------------------------
# 2. Meta-tensor regression
#    Catches: "Cannot copy out of meta tensor; no data!"
# ---------------------------------------------------------------------------

class TestTensorflowDelayGuard(unittest.TestCase):
    """Protects against the 8-18 s TensorFlow startup delay."""

    def test_tf_env_vars_set_before_imports(self):
        """TF_CPP_MIN_LOG_LEVEL and TF_ENABLE_ONEDNN_OPTS must be set at
        module level in image_gen.py, before any import that could trigger TF.

        Without these, TensorFlow initialises oneDNN and scans CUDA devices on
        the first numpy import, adding up to 18 s to every generation request.
        """
        import image_gen as ig
        with open(ig.__file__, encoding="utf-8") as fh:
            source = fh.read()

        for var in ("TF_CPP_MIN_LOG_LEVEL", "TF_ENABLE_ONEDNN_OPTS"):
            self.assertIn(var, source, f"{var} env variable not set in image_gen.py")

        # Both must appear before any `import torch` / `import diffusers`
        tf_pos   = min(source.find("TF_CPP_MIN_LOG_LEVEL"),
                       source.find("TF_ENABLE_ONEDNN_OPTS"))
        torch_pos = source.find("import torch")
        diff_pos  = source.find("import diffusers")
        heavy_pos = min(p for p in (torch_pos, diff_pos) if p != -1)
        self.assertLess(
            tf_pos, heavy_pos,
            "TF env vars must be set BEFORE torch/diffusers imports",
        )

    def test_tf_env_vars_set_in_server(self):
        """server.py must also set TF env vars at startup, since it is the
        process entry point and imports happen before image_gen is loaded."""
        server_path = os.path.join(os.path.dirname(__file__), "server.py")
        with open(server_path, encoding="utf-8") as fh:
            source = fh.read()
        for var in ("TF_CPP_MIN_LOG_LEVEL", "TF_ENABLE_ONEDNN_OPTS"):
            self.assertIn(var, source, f"{var} not set in server.py")


class TestMetaTensorGuard(unittest.TestCase):
    """Protects against the accelerate / meta-device crash."""

    def test_source_contains_low_cpu_mem_usage_false(self):
        """low_cpu_mem_usage=False must be in the from_pretrained call.

        When accelerate is installed, from_pretrained defaults to
        low_cpu_mem_usage=True, placing weights on a data-less 'meta' device.
        A subsequent .to(device) call then raises:
            Cannot copy out of meta tensor; no data!
        Setting low_cpu_mem_usage=False avoids this entirely.
        """
        import image_gen as ig
        with open(ig.__file__, encoding="utf-8") as fh:
            source = fh.read()
        self.assertIn(
            "low_cpu_mem_usage=False",
            source,
            "low_cpu_mem_usage=False must be passed to from_pretrained",
        )

    def test_from_pretrained_not_called_with_low_cpu_mem_usage_false_by_default(self):
        """Primary load path must NOT pass low_cpu_mem_usage=False.

        Passing low_cpu_mem_usage=False forces the entire model into RAM as one
        contiguous allocation before moving to CUDA.  On Windows this triggers a
        native ACCESS_VIOLATION (exit code 0xC0000005) when the contiguous block
        cannot be allocated.  The default accelerate loading is efficient and
        handles meta tensors correctly in accelerate >= 0.20.
        low_cpu_mem_usage=False is only used as a fallback when the meta-tensor
        RuntimeError is actually raised.
        """
        ig = _reload()

        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_cls = MagicMock()
        mock_cls.from_pretrained.return_value = mock_pipe

        with patch("diffusers.AutoPipelineForText2Image", mock_cls, create=True), \
             patch("image_gen._get_device", return_value="cpu"):
            ig._load_pipeline("any/model")

        _, kw = mock_cls.from_pretrained.call_args
        self.assertNotIn(
            "low_cpu_mem_usage", kw,
            "low_cpu_mem_usage must NOT be passed on the primary load path — "
            "it causes a native crash (0xC0000005) on Windows",
        )

    def test_falls_back_to_low_cpu_mem_usage_false_on_meta_tensor_error(self):
        """If a meta-tensor RuntimeError is raised, retries with low_cpu_mem_usage=False."""
        ig = _reload()

        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_cls = MagicMock()
        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("Cannot copy out of meta tensor; no data!")
            return mock_pipe

        mock_cls.from_pretrained.side_effect = side_effect

        with patch("diffusers.AutoPipelineForText2Image", mock_cls, create=True), \
             patch("image_gen._get_device", return_value="cpu"):
            ig._load_pipeline("any/model")

        self.assertEqual(mock_cls.from_pretrained.call_count, 2,
                         "from_pretrained should be called twice (primary + fallback)")
        _, fallback_kw = mock_cls.from_pretrained.call_args
        self.assertFalse(
            fallback_kw.get("low_cpu_mem_usage", True),
            "Fallback call must pass low_cpu_mem_usage=False",
        )


# ---------------------------------------------------------------------------
# 3. Pipeline singleton & error recovery
# ---------------------------------------------------------------------------

class TestPipelineFallback(unittest.TestCase):
    """Regression: backend must work even when AutoPipelineForText2Image is
    missing (older diffusers install or Electron PATH mismatch)."""

    def test_fallback_try_except_structure_present(self):
        """image_gen.py must have a try/except ImportError that falls back
        from AutoPipelineForText2Image to DiffusionPipeline.

        Note: testing this via live attribute manipulation is unreliable because
        diffusers >= 0.25 uses lazy __getattr__ loading that re-creates removed
        attributes on demand. Source inspection is the authoritative check.
        """
        import image_gen as ig
        with open(ig.__file__, encoding="utf-8") as fh:
            source = fh.read()

        self.assertIn("AutoPipelineForText2Image", source)
        self.assertIn("DiffusionPipeline", source)
        # Both must be inside a try/except ImportError block
        self.assertIn("except ImportError", source,
                      "No 'except ImportError' fallback found — "
                      "the code will hard-crash if AutoPipelineForText2Image is absent")

    def test_fallback_uses_diffusion_pipeline_when_import_raises(self):
        """If importing AutoPipelineForText2Image raises ImportError at runtime,
        DiffusionPipeline.from_pretrained must be called instead."""
        ig = _reload()

        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_auto_cls  = MagicMock()
        mock_diffusion_cls = MagicMock()
        mock_diffusion_cls.from_pretrained.return_value = mock_pipe

        # Intercept the `from diffusers import X` calls inside _load_pipeline
        # by patching builtins.__import__ to raise for AutoPipelineForText2Image
        # while passing everything else through normally.
        real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def selective_import(name, *args, **kwargs):
            mod = real_import(name, *args, **kwargs)
            if name == "diffusers":
                # Raise on the specific attribute access to simulate absence
                class _FakeDiffusers:
                    @property
                    def AutoPipelineForText2Image(self):
                        raise ImportError("simulated: AutoPipelineForText2Image absent")
                    DiffusionPipeline = mock_diffusion_cls
                return _FakeDiffusers()
            return mod

        with patch("image_gen._get_device", return_value="cpu"), \
             patch("builtins.__import__", side_effect=selective_import):
            try:
                ig._load_pipeline("local/model")
            except Exception:
                pass  # load may fail for other reasons; we only care about which class was tried

        self.assertTrue(
            mock_diffusion_cls.from_pretrained.called,
            "DiffusionPipeline.from_pretrained was not called as fallback",
        )


class TestPipelineSingleton(unittest.TestCase):

    def test_from_pretrained_called_only_once(self):
        """_load_pipeline must not re-download the model on repeated calls."""
        ig = _reload()
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_cls = MagicMock()
        mock_cls.from_pretrained.return_value = mock_pipe

        with patch("diffusers.AutoPipelineForText2Image", mock_cls, create=True), \
             patch("image_gen._get_device", return_value="cpu"):
            r1 = ig._load_pipeline("m/model")
            r2 = ig._load_pipeline("m/model")
            r3 = ig._load_pipeline("m/model")

        self.assertIs(r1, r2)
        self.assertIs(r2, r3)
        self.assertEqual(mock_cls.from_pretrained.call_count, 1,
                         "from_pretrained should only be called once (singleton guard)")

    def test_pipe_reset_to_none_after_failed_load(self):
        """Regression: if loading fails, _pipe must stay None so retry works.

        Without this, a failed first load would leave _pipe pointing at a broken
        object and every subsequent call would silently return it.
        """
        ig = _reload()
        mock_cls = MagicMock()
        mock_cls.from_pretrained.side_effect = RuntimeError("simulated load failure")

        with patch("diffusers.AutoPipelineForText2Image", mock_cls, create=True), \
             patch("image_gen._get_device", return_value="cpu"):
            with self.assertRaises(RuntimeError):
                ig._load_pipeline("m/model")

        self.assertIsNone(ig._pipe,
                          "_pipe must be None after a failed load so the next call retries")


# ---------------------------------------------------------------------------
# 4. End-to-end generate_image (mocked pipeline — no model download)
# ---------------------------------------------------------------------------

class TestGenerateImage(unittest.TestCase):

    def setUp(self):
        self.ig = _reload()

    def test_result_contains_required_keys(self):
        fake = MagicMock(side_effect=_fake_pipe_call)
        with patch("image_gen._load_pipeline", return_value=fake), \
             patch("image_gen._device", "cpu"):
            result = asyncio.run(self.ig.generate_image("a red apple"))

        for key in ("image_b64", "model", "prompt", "negative_prompt", "size", "timings"):
            self.assertIn(key, result, f"Missing key '{key}' in result")

    def test_image_b64_decodes_to_valid_png(self):
        fake = MagicMock(side_effect=_fake_pipe_call)
        with patch("image_gen._load_pipeline", return_value=fake), \
             patch("image_gen._device", "cpu"):
            result = asyncio.run(self.ig.generate_image("a blue sky"))

        raw = base64.b64decode(result["image_b64"])
        self.assertEqual(raw[:8], b"\x89PNG\r\n\x1a\n",
                         "image_b64 did not decode to a valid PNG")

    def test_prompt_echoed_correctly(self):
        fake = MagicMock(side_effect=_fake_pipe_call)
        with patch("image_gen._load_pipeline", return_value=fake), \
             patch("image_gen._device", "cpu"):
            result = asyncio.run(self.ig.generate_image("purple dragon"))

        self.assertEqual(result["prompt"], "purple dragon")

    def test_size_string_parsed_to_width_height(self):
        fake = MagicMock(side_effect=_fake_pipe_call)
        with patch("image_gen._load_pipeline", return_value=fake), \
             patch("image_gen._device", "cpu"):
            asyncio.run(self.ig.generate_image("test", size="768x512"))

        _, kw = fake.call_args
        self.assertEqual(kw.get("width"),  768, "width not parsed from size string")
        self.assertEqual(kw.get("height"), 512, "height not parsed from size string")

    def test_default_size_is_512x512(self):
        fake = MagicMock(side_effect=_fake_pipe_call)
        with patch("image_gen._load_pipeline", return_value=fake), \
             patch("image_gen._device", "cpu"):
            asyncio.run(self.ig.generate_image("test"))

        _, kw = fake.call_args
        self.assertEqual(kw.get("width"),  512)
        self.assertEqual(kw.get("height"), 512)

    def test_timings_inference_ms_is_non_negative(self):
        fake = MagicMock(side_effect=_fake_pipe_call)
        with patch("image_gen._load_pipeline", return_value=fake), \
             patch("image_gen._device", "cpu"):
            result = asyncio.run(self.ig.generate_image("castle"))

        self.assertIn("inference_ms", result["timings"])
        self.assertGreaterEqual(result["timings"]["inference_ms"], 0)

    def test_negative_prompt_forwarded_to_pipeline(self):
        fake = MagicMock(side_effect=_fake_pipe_call)
        with patch("image_gen._load_pipeline", return_value=fake), \
             patch("image_gen._device", "cpu"):
            asyncio.run(self.ig.generate_image("dog", negative_prompt="blurry"))

        _, kw = fake.call_args
        self.assertEqual(kw.get("negative_prompt"), "blurry")


# ---------------------------------------------------------------------------
# 5. Memory-pressure regression
#    Catches: "almost crashes computer" — RAM exhausted during model load/inference
# ---------------------------------------------------------------------------

class TestFp16VariantLoading(unittest.TestCase):
    """Ensures variant='fp16' is intentionally NOT used on the primary load path.

    variant='fp16' causes diffusers to look for model.fp16.safetensors files.
    If those files are not already in the local HuggingFace cache, diffusers
    silently triggers a fresh ~1.7 GB download.  On a machine where only the
    default safetensors are cached, this download stalls the backend process
    and makes the app appear to crash.  We rely on torch_dtype=float16 instead,
    which converts the already-cached full-precision weights on load without any
    extra network activity.
    """

    def test_variant_not_used_on_primary_load(self):
        """Primary from_pretrained call must NOT include variant='fp16'."""
        ig = _reload()

        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_cls = MagicMock()
        mock_cls.from_pretrained.return_value = mock_pipe

        with patch("diffusers.AutoPipelineForText2Image", mock_cls, create=True), \
             patch("image_gen._get_device", return_value="cuda"):
            ig._load_pipeline("any/model")

        call_kwargs = mock_cls.from_pretrained.call_args[1]
        self.assertNotIn(
            "variant", call_kwargs,
            "variant='fp16' must NOT be passed — it triggers a fresh ~1.7 GB download "
            "when the fp16 safetensors are not already cached locally, crashing the backend.",
        )

    def test_torch_dtype_float16_used_on_cuda(self):
        """On CUDA, torch_dtype=float16 must be passed so the already-cached full-precision
        weights are cast to float16 during loading — no extra download required."""
        ig = _reload()

        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_cls = MagicMock()
        mock_cls.from_pretrained.return_value = mock_pipe

        with patch("diffusers.AutoPipelineForText2Image", mock_cls, create=True), \
             patch("image_gen._get_device", return_value="cuda"):
            ig._load_pipeline("any/model")

        call_kwargs = mock_cls.from_pretrained.call_args[1]
        import torch
        self.assertEqual(
            call_kwargs.get("torch_dtype"), torch.float16,
            "torch_dtype=float16 must be passed on CUDA for memory-efficient loading",
        )


class TestMemoryCleanup(unittest.TestCase):
    """Protects against VRAM/RAM accumulation across multiple generations.

    Without explicit cleanup, PyTorch holds freed allocations in its CUDA
    memory pool indefinitely, causing VRAM to grow with each request and
    eventually starving other processes (browser, Electron, etc.).
    """

    def test_gc_collect_called_after_generation(self):
        """gc.collect() must be called after every generation to free Python objects."""
        import image_gen as ig
        with open(ig.__file__, encoding="utf-8") as fh:
            source = fh.read()
        self.assertIn(
            "gc.collect()",
            source,
            "gc.collect() must be called after generation to free unreferenced tensors",
        )

    def test_cuda_empty_cache_called_after_generation(self):
        """torch.cuda.empty_cache() must be called after every generation."""
        import image_gen as ig
        with open(ig.__file__, encoding="utf-8") as fh:
            source = fh.read()
        self.assertIn(
            "torch.cuda.empty_cache()",
            source,
            "torch.cuda.empty_cache() must be called after generation to release VRAM pool",
        )

    def test_attention_slicing_present_for_cpu_path(self):
        """enable_attention_slicing() must exist in the source for the CPU path.

        On CPU every byte of RAM matters, so attention slicing is enabled.
        On CUDA (RTX 4060 Ti, 16 GB VRAM) SD v1.5 only uses ~2 GB, so slicing
        would add overhead with no benefit — it must NOT be called unconditionally.
        """
        import image_gen as ig
        with open(ig.__file__, encoding="utf-8") as fh:
            source = fh.read()
        self.assertIn(
            "enable_attention_slicing()",
            source,
            "enable_attention_slicing() must be present for the CPU inference path",
        )
        # Verify it is guarded by a device check, not called unconditionally.
        # The line 'if _device != "cuda":' must precede the enable call.
        cpu_guard_idx = source.find('if _device != "cuda":')
        attn_slice_idx = source.find("enable_attention_slicing()")
        self.assertGreater(
            attn_slice_idx,
            cpu_guard_idx,
            "enable_attention_slicing() must be inside the CPU-only guard block",
        )

    def test_ipc_collect_not_present(self):
        """torch.cuda.ipc_collect() must NOT be called.

        ipc_collect() manages shared IPC handles between processes.  In a
        single-process inference server there are no IPC handles, and calling
        it can trigger driver-level faults on some CUDA versions.
        """
        import image_gen as ig
        with open(ig.__file__, encoding="utf-8") as fh:
            source = fh.read()
        self.assertNotIn(
            "ipc_collect()",
            source,
            "torch.cuda.ipc_collect() must not be called in single-process inference",
        )

    def test_set_num_interop_threads_not_present(self):
        """torch.set_num_interop_threads() must NOT be called.

        This function may only be called once, before any parallel work begins.
        By the time the first generate request arrives the server's import chain
        will already have started PyTorch's interop pool, making a late call
        either a no-op or a source of race conditions.
        """
        import image_gen as ig
        with open(ig.__file__, encoding="utf-8") as fh:
            source = fh.read()
        self.assertNotIn(
            "set_num_interop_threads",
            source,
            "set_num_interop_threads must not be called from a thread-pool worker",
        )

    def test_cudnn_benchmark_is_false(self):
        """cudnn.benchmark must be False to avoid extra VRAM from algorithm profiling.

        benchmark=True causes cuDNN to profile many convolution algorithms on the
        first inference and cache results, consuming extra VRAM.  With only 6 GB
        free system RAM this can be the difference between stable and crashing.
        """
        import image_gen as ig
        with open(ig.__file__, encoding="utf-8") as fh:
            source = fh.read()
        self.assertIn(
            "cudnn.benchmark = False",
            source,
            "cudnn.benchmark must be False to avoid extra VRAM allocation",
        )
        self.assertNotIn(
            "cudnn.benchmark = True",
            source,
            "cudnn.benchmark = True found — this causes VRAM spikes and must be removed",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
