import asyncio
import base64
import json
import os
import random
import re
from typing import Any, Dict, Optional, Tuple

import aiohttp


class DashScopeError(RuntimeError):
    def __init__(
        self,
        status: int,
        message: str,
        request_id: Optional[str] = None,
        code: Optional[str] = None,
        raw: Optional[str] = None,
    ):
        super().__init__(message)
        self.status = int(status)
        self.request_id = request_id
        self.code = code
        self.raw = raw


def _clean_api_key(api_key: Optional[str]) -> Optional[str]:
    if api_key is None:
        return None
    k = str(api_key).strip()
    if not k:
        return None
    if (k.startswith('"') and k.endswith('"')) or (k.startswith("'") and k.endswith("'")):
        k = k[1:-1].strip()
    return k or None


def _load_from_dotenv(var_name: str) -> Optional[str]:
    candidates = []
    try:
        candidates.append(os.path.join(os.getcwd(), ".env"))
        candidates.append(os.path.join(os.getcwd(), ".env.local"))
    except Exception:
        pass
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, ".."))
        candidates.append(os.path.join(root, ".env"))
        candidates.append(os.path.join(root, ".env.local"))
        candidates.append(os.path.join(here, ".env"))
        candidates.append(os.path.join(here, ".env.local"))
    except Exception:
        pass

    for path in candidates:
        try:
            if not path or not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    ln = line.strip()
                    if not ln or ln.startswith("#"):
                        continue
                    if ln.startswith("export "):
                        ln = ln[len("export "):].strip()
                    if not ln.startswith(f"{var_name}="):
                        continue
                    _, v = ln.split("=", 1)
                    return _clean_api_key(v)
        except Exception:
            continue
    return None


def _dashscope_base_url(provider_cfg: Dict[str, Any]) -> str:
    v = provider_cfg.get("base_url") if isinstance(provider_cfg, dict) else None
    v = str(v).strip() if v else ""
    if v:
        return v.rstrip("/")
    region = str((provider_cfg or {}).get("region") or "cn").strip().lower()
    if region in ("intl", "sg", "singapore"):
        return "https://dashscope-intl.aliyuncs.com"
    return "https://dashscope.aliyuncs.com"


def _infer_format_from_url(url: str) -> str:
    u = str(url or "")
    m = re.search(r"\.(png|jpg|jpeg|webp)(?:\?|$)", u, flags=re.IGNORECASE)
    if not m:
        return "png"
    ext = m.group(1).lower()
    if ext == "jpeg":
        ext = "jpg"
    return ext


def _infer_format_from_content_type(ct: Optional[str]) -> Optional[str]:
    if not ct:
        return None
    t = ct.lower()
    if "image/png" in t:
        return "png"
    if "image/jpeg" in t or "image/jpg" in t:
        return "jpg"
    if "image/webp" in t:
        return "webp"
    return None


async def generate_image(
    prompt: str,
    negative_prompt: Optional[str] = None,
    size: str = "1664*928",
    prompt_extend: bool = True,
    watermark: bool = False,
    seed: Optional[int] = None,
    model: str = "qwen-image-max",
    api_key: Optional[str] = None,
    provider_cfg: Optional[Dict[str, Any]] = None,
    timeout_s: float = 60.0,
) -> Dict[str, Any]:
    provider_cfg = provider_cfg or {}
    api_key = _clean_api_key(api_key) or _clean_api_key(os.environ.get("DASHSCOPE_API_KEY")) or _load_from_dotenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY")

    base_url = _dashscope_base_url(provider_cfg)
    url = f"{base_url}/api/v1/services/aigc/multimodal-generation/generation"
    has_explicit_endpoint = False
    try:
        if isinstance(provider_cfg, dict):
            if str(provider_cfg.get("base_url") or "").strip():
                has_explicit_endpoint = True
            elif str(provider_cfg.get("region") or "").strip():
                has_explicit_endpoint = True
    except Exception:
        has_explicit_endpoint = False
    tried_alternate_endpoint = False

    params: Dict[str, Any] = {
        "negative_prompt": str(negative_prompt or "").strip() or None,
        "prompt_extend": bool(prompt_extend),
        "watermark": bool(watermark),
        "size": str(size or "1664*928"),
    }
    if seed is not None:
        try:
            params["seed"] = int(seed)
        except Exception:
            pass

    parameters = {k: v for k, v in params.items() if v is not None}

    payload = {
        "model": str(model or "qwen-image-max"),
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": str(prompt or "")}
                    ],
                }
            ]
        },
        "parameters": parameters,
    }

    timeout = aiohttp.ClientTimeout(total=float(timeout_s))
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    async with aiohttp.ClientSession(timeout=timeout) as session:
        data = None
        last_err: Optional[DashScopeError] = None
        for attempt in range(3):
            async with session.post(url, headers=headers, json=payload) as resp:
                raw = await resp.text()
                if resp.status >= 400:
                    err_obj = None
                    try:
                        err_obj = json.loads(raw)
                    except Exception:
                        err_obj = None

                    req_id = None
                    code = None
                    msg = raw
                    if isinstance(err_obj, dict):
                        req_id = err_obj.get("request_id") or err_obj.get("requestId")
                        code = err_obj.get("code")
                        msg = err_obj.get("message") or err_obj.get("msg") or msg

                    if resp.status == 401:
                        if (not has_explicit_endpoint) and (not tried_alternate_endpoint):
                            tried_alternate_endpoint = True
                            alt_base = "https://dashscope-intl.aliyuncs.com" if "dashscope-intl" not in base_url else "https://dashscope.aliyuncs.com"
                            base_url = alt_base
                            url = f"{base_url}/api/v1/services/aigc/multimodal-generation/generation"
                            await asyncio.sleep(0)
                            continue
                        msg = (
                            "Unauthorized (401). Verify DASHSCOPE_API_KEY is correct for your region "
                            "(CN vs Intl keys differ), and that your key has access to Qwen-Image. "
                            f"{msg}"
                        )

                    last_err = DashScopeError(
                        status=resp.status,
                        message=f"DashScope error {resp.status}{f' ({code})' if code else ''}: {str(msg)[:800]}",
                        request_id=req_id,
                        code=code,
                        raw=str(raw)[:1200],
                    )

                    if resp.status in (429, 500, 502, 503, 504) and attempt < 2:
                        await asyncio.sleep(0.6 * (2 ** attempt) + random.random() * 0.25)
                        continue
                    raise last_err

                data = await resp.json(content_type=None)
                break

        if data is None:
            raise last_err or DashScopeError(status=500, message="DashScope request failed", raw=None)

        image_url = None
        try:
            image_url = (
                data.get("output", {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", [{}])[0]
                .get("image")
            )
        except Exception:
            image_url = None

        if not image_url:
            raise RuntimeError(f"DashScope response missing image url: {str(data)[:800]}")

        inferred_ext = _infer_format_from_url(image_url)

        async with session.get(image_url) as img_resp:
            if img_resp.status >= 400:
                body = await img_resp.text()
                raise RuntimeError(f"Image fetch error {img_resp.status}: {body[:300]}")
            img_bytes = await img_resp.read()
            ct = img_resp.headers.get("content-type")

    fmt = _infer_format_from_content_type(ct) or inferred_ext or "png"
    b64 = base64.b64encode(img_bytes).decode("utf-8")

    usage = data.get("usage") if isinstance(data, dict) else {}
    width = usage.get("width") if isinstance(usage, dict) else None
    height = usage.get("height") if isinstance(usage, dict) else None

    return {
        "format": fmt,
        "image_b64": b64,
        "image_url": image_url,
        "model": payload.get("model"),
        "prompt": str(prompt or ""),
        "negative_prompt": str(negative_prompt or "").strip() or "",
        "size": str(size or ""),
        "prompt_extend": bool(prompt_extend),
        "watermark": bool(watermark),
        "seed": seed,
        "width": width,
        "height": height,
        "request_id": data.get("request_id") if isinstance(data, dict) else None,
    }


async def warmup(provider_cfg: Optional[Dict[str, Any]] = None) -> bool:
    try:
        _ = _dashscope_base_url(provider_cfg or {})
        await asyncio.sleep(0)
        return True
    except Exception:
        return False
