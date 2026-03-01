import json
import os
import random
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class LearningStore:
    def __init__(self, db_path: str = "jarvis_learning.db"):
        p = str(db_path) if db_path is not None else "jarvis_learning.db"
        if not os.path.isabs(p):
            p = str((Path(__file__).resolve().parent / p).resolve())
        self.db_path = p
        self._lock = threading.Lock()
        self.init_db()

    def init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS recommendations (id TEXT PRIMARY KEY, ts REAL, user_text TEXT, intent TEXT, context_json TEXT, strategy_json TEXT, first_token_ms REAL, total_ms REAL, response_chars INTEGER, error TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS tool_events (id INTEGER PRIMARY KEY AUTOINCREMENT, rec_id TEXT, ts REAL, tool TEXT, event TEXT, request_id TEXT, ok INTEGER, message TEXT, meta_json TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, rec_id TEXT, ts REAL, outcome TEXT, note TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS strategy_weights (key TEXT PRIMARY KEY, weight REAL, n INTEGER, last_update REAL)"
            )
            conn.commit()

    def start_recommendation(
        self,
        rec_id: str,
        user_text: str,
        intent: str,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
    ) -> None:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO recommendations (id, ts, user_text, intent, context_json, strategy_json, first_token_ms, total_ms, response_chars, error) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL)",
                    (
                        rec_id,
                        float(time.time()),
                        str(user_text or ""),
                        str(intent or ""),
                        json.dumps(context or {}, ensure_ascii=False),
                        json.dumps(strategy or {}, ensure_ascii=False),
                    ),
                )
                conn.commit()

    def update_recommendation_metrics(
        self,
        rec_id: str,
        first_token_ms: Optional[float] = None,
        total_ms: Optional[float] = None,
        response_chars: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        sets: List[str] = []
        params: List[Any] = []
        if first_token_ms is not None:
            sets.append("first_token_ms = ?")
            params.append(float(first_token_ms))
        if total_ms is not None:
            sets.append("total_ms = ?")
            params.append(float(total_ms))
        if response_chars is not None:
            sets.append("response_chars = ?")
            params.append(int(response_chars))
        if error is not None:
            sets.append("error = ?")
            params.append(str(error))
        if not sets:
            return
        params.append(rec_id)
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    f"UPDATE recommendations SET {', '.join(sets)} WHERE id = ?",
                    tuple(params),
                )
                conn.commit()

    def log_tool_event(
        self,
        rec_id: Optional[str],
        tool: str,
        event: str,
        request_id: Optional[str] = None,
        ok: Optional[bool] = None,
        message: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not rec_id:
            return
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO tool_events (rec_id, ts, tool, event, request_id, ok, message, meta_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        rec_id,
                        float(time.time()),
                        str(tool or ""),
                        str(event or ""),
                        str(request_id) if request_id else None,
                        (1 if ok else 0) if ok is not None else None,
                        str(message) if message else None,
                        json.dumps(meta or {}, ensure_ascii=False),
                    ),
                )
                conn.commit()

    def record_feedback(self, rec_id: str, outcome: str, note: Optional[str] = None) -> Dict[str, Any]:
        oc = str(outcome or "").strip().lower()
        if oc not in {"accepted", "modified", "rejected", "ignored"}:
            return {"ok": False, "error": "invalid_outcome"}

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO feedback (rec_id, ts, outcome, note) VALUES (?, ?, ?, ?)",
                    (rec_id, float(time.time()), oc, str(note) if note else None),
                )
                conn.commit()

        update = self._apply_reward(rec_id, oc)
        return {"ok": True, **update}

    def has_feedback(self, rec_id: str) -> bool:
        if not rec_id:
            return False
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT 1 FROM feedback WHERE rec_id = ? LIMIT 1",
                (rec_id,),
            )
            return cur.fetchone() is not None

    def auto_mark_ignored(self, rec_id: str) -> bool:
        if not rec_id:
            return False
        if self.has_feedback(rec_id):
            return False
        out = self.record_feedback(rec_id, "ignored")
        return bool(out.get("ok"))

    def _get_weight_row(self, key: str) -> Tuple[float, int]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT weight, n FROM strategy_weights WHERE key = ?",
                (key,),
            )
            row = cur.fetchone()
            if not row:
                return 0.0, 0
            return float(row[0] or 0.0), int(row[1] or 0)

    def _set_weight_row(self, key: str, weight: float, n: int) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO strategy_weights (key, weight, n, last_update) VALUES (?, ?, ?, ?)",
                (key, float(weight), int(n), float(time.time())),
            )
            conn.commit()

    def _reward_value(self, outcome: str) -> float:
        if outcome == "accepted":
            return 1.0
        if outcome == "modified":
            return 0.35
        if outcome == "ignored":
            return 0.0
        if outcome == "rejected":
            return -1.0
        return 0.0

    def _count_tool_errors(self, rec_id: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT COUNT(1) FROM tool_events WHERE rec_id = ? AND event = 'error'",
                (rec_id,),
            )
            row = cur.fetchone()
            return int(row[0] or 0) if row else 0

    def _get_strategy_for_rec(self, rec_id: str) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT strategy_json FROM recommendations WHERE id = ?",
                (rec_id,),
            )
            row = cur.fetchone()
            if not row or not row[0]:
                return {}
            try:
                return json.loads(row[0]) or {}
            except Exception:
                return {}

    def _apply_reward(self, rec_id: str, outcome: str) -> Dict[str, Any]:
        strategy = self._get_strategy_for_rec(rec_id)
        if not strategy:
            return {"updated": False}

        reward = self._reward_value(outcome)
        tool_errs = self._count_tool_errors(rec_id)
        reward = float(reward) - 0.2 * float(tool_errs)

        alpha = 0.15
        updated_keys: List[str] = []

        def upd(dim: str, val: Optional[str], factor: float = 1.0) -> None:
            if not val:
                return
            key = f"{dim}:{val}"
            w, n = self._get_weight_row(key)
            new_w = float(w) + float(alpha) * float(factor) * float(reward)
            self._set_weight_row(key, new_w, n + 1)
            updated_keys.append(key)

        upd("prompt_variant", strategy.get("prompt_variant"))
        upd("reasoning_depth", strategy.get("reasoning_depth"))
        upd("response_timing", strategy.get("response_timing"))

        tool_bias = strategy.get("tool_bias")
        if isinstance(tool_bias, list):
            for i, t in enumerate(tool_bias[:6]):
                upd("tool", str(t), factor=max(0.1, 1.0 - 0.12 * i))

        return {"updated": True, "reward": reward, "keys": updated_keys}

    def get_weights(self, prefix: Optional[str] = None, limit: int = 50) -> Dict[str, float]:
        q = "SELECT key, weight FROM strategy_weights"
        args: List[Any] = []
        if prefix:
            q += " WHERE key LIKE ?"
            args.append(f"{prefix}:%")
        q += " ORDER BY weight DESC LIMIT ?"
        args.append(int(limit))
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(q, tuple(args))
            out: Dict[str, float] = {}
            for k, w in cur.fetchall():
                out[str(k)] = float(w or 0.0)
            return out

    def get_failure_patterns(self, lookback: int = 50) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT tool, COUNT(1) AS c FROM tool_events WHERE event = 'error' GROUP BY tool ORDER BY c DESC LIMIT 6"
            )
            tool_errors = [{"tool": str(r[0]), "count": int(r[1] or 0)} for r in cur.fetchall()]

            cur = conn.execute(
                "SELECT outcome, COUNT(1) AS c FROM feedback GROUP BY outcome ORDER BY c DESC"
            )
            outcomes = [{"outcome": str(r[0]), "count": int(r[1] or 0)} for r in cur.fetchall()]

            cur = conn.execute(
                "SELECT error, COUNT(1) AS c FROM recommendations WHERE error IS NOT NULL AND error != '' GROUP BY error ORDER BY c DESC LIMIT 6"
            )
            rec_errors = [{"error": str(r[0]), "count": int(r[1] or 0)} for r in cur.fetchall()]

            cur = conn.execute(
                "SELECT rec_id, outcome FROM feedback ORDER BY id DESC LIMIT ?",
                (int(lookback),),
            )
            tail = cur.fetchall() or []
            recent_rejects = sum(1 for _, o in tail if str(o) == "rejected")
            recent_modified = sum(1 for _, o in tail if str(o) == "modified")

        return {
            "tool_errors": tool_errors,
            "outcomes": outcomes,
            "recommendation_errors": rec_errors,
            "recent": {"rejected": int(recent_rejects), "modified": int(recent_modified), "n": int(len(tail))},
        }


class LearningPolicy:
    def __init__(self, store: LearningStore):
        self.store = store

    def select_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        prompt_variants = ["default", "tool_strict", "concise"]
        reasoning_depths = ["shallow", "normal", "deep"]
        response_timings = ["fast", "balanced"]

        pv = self._choose("prompt_variant", prompt_variants, epsilon=0.18)
        rd = self._choose("reasoning_depth", reasoning_depths, epsilon=0.18)
        rt = self._choose("response_timing", response_timings, epsilon=0.12)

        tools = [
            "generate_cad",
            "iterate_cad",
            "run_web_agent",
            "read_file",
            "read_directory",
            "write_file",
            "create_project",
            "switch_project",
            "list_projects",
        ]
        tool_bias = self._rank_tools(tools)

        base_caps = {"shallow": 450, "normal": 700, "deep": 1200}
        timing_mult = {"fast": 0.8, "balanced": 1.0}
        soft_cap = int(float(base_caps.get(rd, 700)) * float(timing_mult.get(rt, 1.0)))

        return {
            "prompt_variant": pv,
            "reasoning_depth": rd,
            "response_timing": rt,
            "soft_cap_chars": soft_cap,
            "tool_bias": tool_bias,
            "intent": context.get("intent"),
        }

    def prompt_suffix(self, strategy: Dict[str, Any]) -> str:
        pv = str(strategy.get("prompt_variant") or "default")
        tool_bias = strategy.get("tool_bias")
        pref = ""
        if isinstance(tool_bias, list) and tool_bias:
            top = [str(t) for t in tool_bias[:5]]
            pref = "\n\nTool preference (learned): " + " > ".join(top)

        if pv == "tool_strict":
            return (
                "\n\nTool-use policy (strict): If and only if a tool is required, emit exactly one valid JSON tool call object and nothing else. If no tool is required, answer normally."
                + pref
            )
        if pv == "concise":
            return "\n\nPacing override: Prefer the shortest correct answer; ask one clarifying question when uncertain." + pref
        return pref

    def _choose(self, dim: str, options: List[str], epsilon: float = 0.15) -> str:
        if not options:
            return ""
        if random.random() < float(epsilon):
            return random.choice(options)

        best = options[0]
        best_w = None
        for o in options:
            w, _n = self._get(dim, o)
            if best_w is None or w > best_w:
                best_w = w
                best = o
        return best

    def _get(self, dim: str, val: str) -> Tuple[float, int]:
        key = f"{dim}:{val}"
        return self.store._get_weight_row(key)

    def _rank_tools(self, tools: List[str]) -> List[str]:
        scored: List[Tuple[float, str]] = []
        for t in tools:
            w, _n = self._get("tool", str(t))
            scored.append((float(w), str(t)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _w, t in scored]
