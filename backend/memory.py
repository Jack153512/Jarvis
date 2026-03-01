import sqlite3
import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any


class JarvisMemory:
    def __init__(self, db_path: str = "jarvis_memory.db"):
        p = str(db_path) if db_path is not None else "jarvis_memory.db"
        if not os.path.isabs(p):
            p = str((Path(__file__).resolve().parent / p).resolve())
        self.db_path = p
        self.init_db()

    def init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS personal_memory (key TEXT PRIMARY KEY, value TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS project_memory (key TEXT PRIMARY KEY, value TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS short_term_memory (key TEXT PRIMARY KEY, value TEXT)"
            )
            conn.commit()

    def _set_value(self, table: str, key: str, value: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO {table} (key, value) VALUES (?, ?)",
                (key, value),
            )
            conn.commit()

    def _get_value(self, table: str, key: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(f"SELECT value FROM {table} WHERE key = ?", (key,))
            row = cur.fetchone()
            return row[0] if row else None

    def _update_value(self, table: str, key: str, value: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                f"UPDATE {table} SET value = ? WHERE key = ?",
                (value, key),
            )
            conn.commit()
            return cur.rowcount > 0

    def _delete_key(self, table: str, key: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(f"DELETE FROM {table} WHERE key = ?", (key,))
            conn.commit()
            return cur.rowcount > 0

    def save_personal(self, key: str, value: str) -> None:
        self._set_value("personal_memory", key, value)

    def get_personal(self, key: str) -> Optional[str]:
        return self._get_value("personal_memory", key)

    def update_personal(self, key: str, value: str) -> bool:
        return self._update_value("personal_memory", key, value)

    def delete_personal(self, key: str) -> bool:
        return self._delete_key("personal_memory", key)

    def save_project(self, key: str, value: str) -> None:
        self._set_value("project_memory", key, value)

    def get_project(self, key: str) -> Optional[str]:
        return self._get_value("project_memory", key)

    def update_project(self, key: str, value: str) -> bool:
        return self._update_value("project_memory", key, value)

    def delete_project(self, key: str) -> bool:
        return self._delete_key("project_memory", key)

    def save_short_term(self, key: str, value: str) -> None:
        self._set_value("short_term_memory", key, value)

    def get_short_term(self, key: str) -> Optional[str]:
        return self._get_value("short_term_memory", key)

    def clear_short_term(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM short_term_memory")
            conn.commit()

    def set_user_name(self, user_name: str) -> None:
        self.save_personal("user_name", user_name)

    def get_user_name(self) -> Optional[str]:
        return self.get_personal("user_name")

    def set_assistant_name(self, assistant_name: str) -> None:
        self.save_personal("assistant_name", assistant_name)

    def get_assistant_name(self) -> Optional[str]:
        return self.get_personal("assistant_name")

    # ── Chat history ──────────────────────────────────────────────────────────

    def _ensure_chat_history_table(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS chat_history (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    sender    TEXT    NOT NULL,
                    text      TEXT    NOT NULL,
                    timestamp TEXT    NOT NULL,
                    session   TEXT    NOT NULL DEFAULT ''
                )"""
            )
            conn.commit()

    def add_chat_message(self, sender: str, text: str, timestamp: str, session: str = "") -> None:
        self._ensure_chat_history_table()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO chat_history (sender, text, timestamp, session) VALUES (?, ?, ?, ?)",
                (sender, text, timestamp, session),
            )
            conn.commit()

    def get_chat_history(self, limit: int = 200) -> list:
        self._ensure_chat_history_table()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT sender, text, timestamp, session FROM chat_history ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
        # Return oldest-first so the chat renders in chronological order
        return [
            {"sender": r[0], "text": r[1], "time": r[2], "session": r[3], "fromHistory": True}
            for r in reversed(rows)
        ]

    def clear_chat_history(self) -> None:
        self._ensure_chat_history_table()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM chat_history")
            conn.commit()

    # ── Multi-conversation management ─────────────────────────────────────────

    def _ensure_conversation_tables(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id          TEXT PRIMARY KEY,
                    title       TEXT NOT NULL DEFAULT 'New Chat',
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL,
                    llm_history TEXT NOT NULL DEFAULT '[]'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    sender          TEXT NOT NULL,
                    text            TEXT NOT NULL,
                    timestamp       TEXT NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conv_msgs_conv_id "
                "ON conversation_messages(conversation_id)"
            )
            conn.commit()

    def create_conversation(self, conv_id: str, title: str, created_at: str = "") -> None:
        from datetime import datetime, timezone
        self._ensure_conversation_tables()
        ts = created_at or datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conv_id, title, ts, ts),
            )
            conn.commit()

    def get_conversations(self) -> List[Dict[str, Any]]:
        self._ensure_conversation_tables()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
            )
            return [
                {"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]}
                for r in cur.fetchall()
            ]

    def get_conversation(self, conv_id: str) -> Optional[Dict[str, Any]]:
        self._ensure_conversation_tables()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conv_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {"id": row[0], "title": row[1], "created_at": row[2], "updated_at": row[3]}

    def update_conversation_title(self, conv_id: str, title: str) -> None:
        self._ensure_conversation_tables()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (title, conv_id),
            )
            conn.commit()

    def update_conversation_timestamp(self, conv_id: str, updated_at: str) -> None:
        self._ensure_conversation_tables()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (updated_at, conv_id),
            )
            conn.commit()

    def save_llm_history(self, conv_id: str, history: List[Dict[str, str]]) -> None:
        self._ensure_conversation_tables()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE conversations SET llm_history = ? WHERE id = ?",
                (json.dumps(history), conv_id),
            )
            conn.commit()

    def get_llm_history(self, conv_id: str) -> List[Dict[str, str]]:
        self._ensure_conversation_tables()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT llm_history FROM conversations WHERE id = ?",
                (conv_id,),
            )
            row = cur.fetchone()
        if not row or not row[0]:
            return []
        try:
            return json.loads(row[0])
        except Exception:
            return []

    def delete_conversation(self, conv_id: str) -> None:
        self._ensure_conversation_tables()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM conversation_messages WHERE conversation_id = ?", (conv_id,))
            conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
            conn.commit()

    def add_conversation_message(
        self, conv_id: str, sender: str, text: str, timestamp: str
    ) -> None:
        from datetime import datetime, timezone
        self._ensure_conversation_tables()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO conversation_messages (conversation_id, sender, text, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (conv_id, sender, text, timestamp),
            )
            conn.commit()
        # Always use a server-side ISO timestamp for updated_at so the
        # sidebar's relative-time display works correctly.
        self.update_conversation_timestamp(conv_id, datetime.now(timezone.utc).isoformat())

    def get_conversation_messages(self, conv_id: str) -> List[Dict[str, Any]]:
        self._ensure_conversation_tables()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT sender, text, timestamp FROM conversation_messages "
                "WHERE conversation_id = ? ORDER BY id ASC",
                (conv_id,),
            )
            return [{"sender": r[0], "text": r[1], "time": r[2]} for r in cur.fetchall()]
