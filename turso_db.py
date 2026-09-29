"""
turso_db.py - tiny Turso (libSQL) client over the HTTP API. Standard library only.

Set these in .env to turn it on:
    TURSO_DATABASE_URL=libsql://your-db-name-your-org.turso.io
    TURSO_AUTH_TOKEN=eyJ...

If they're not set, `db.enabled` is False and the rest of the bot falls back to
the old local JSON files, so nothing breaks in local dev.

Reads are synchronous (used once at startup). Writes go through a single
background thread with a FIFO queue, so bot commands never block on the
network and writes reach Turso in the order they happened. Network failures
are retried with backoff; the queue is flushed on normal shutdown.
"""

import atexit
import json
import logging
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from typing import Any, List, Optional, Sequence, Tuple

# main.py imports guild_store (-> this module) before it calls load_dotenv(),
# so load .env here too or the TURSO_* vars wouldn't be visible yet.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

log = logging.getLogger("turso")

Stmt = Tuple[str, Sequence[Any]]

SCHEMA: List[str] = [
    """CREATE TABLE IF NOT EXISTS collection (
        guild_id TEXT NOT NULL, user_id TEXT NOT NULL,
        species_key TEXT NOT NULL, species TEXT NOT NULL,
        PRIMARY KEY (guild_id, user_id, species_key))""",
    """CREATE TABLE IF NOT EXISTS reserve (
        guild_id TEXT NOT NULL, user_id TEXT NOT NULL,
        species_key TEXT NOT NULL, species TEXT NOT NULL,
        PRIMARY KEY (guild_id, user_id, species_key))""",
    """CREATE TABLE IF NOT EXISTS shiny (
        guild_id TEXT NOT NULL, user_id TEXT NOT NULL, species TEXT NOT NULL,
        PRIMARY KEY (guild_id, user_id))""",
    """CREATE TABLE IF NOT EXISTS guild_settings (
        guild_id TEXT PRIMARY KEY, res_role TEXT)""",
    """CREATE TABLE IF NOT EXISTS disabled_guilds (guild_id TEXT PRIMARY KEY)""",
    """CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)""",
]


class TursoError(Exception):
    """Permanent failure (bad SQL, bad token...) - retrying won't help."""


class TursoNetworkError(Exception):
    """Transient failure (timeout, 5xx, DNS...) - safe to retry."""


def _http_url(url: str) -> str:
    url = url.strip().rstrip("/")
    for prefix in ("libsql://", "wss://", "ws://"):
        if url.startswith(prefix):
            return "https://" + url[len(prefix):]
    return url


def _arg(v: Any) -> dict:
    if v is None:
        return {"type": "null"}
    if isinstance(v, bool):
        v = int(v)
    if isinstance(v, int):
        return {"type": "integer", "value": str(v)}
    if isinstance(v, float):
        return {"type": "float", "value": v}
    return {"type": "text", "value": str(v)}


def _cell(c: dict) -> Any:
    t = c.get("type")
    if t == "null":
        return None
    if t == "integer":
        return int(c["value"])
    if t == "float":
        return float(c["value"])
    return c.get("value")


class TursoDB:
    def __init__(self, url: Optional[str], token: Optional[str]):
        self.enabled = bool(url and token)
        self._url = _http_url(url) if url else ""
        self._token = token or ""
        self._q: "queue.Queue[List[Stmt]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        if self.enabled:
            self._init_schema()
            self._thread = threading.Thread(target=self._writer, name="turso-writer", daemon=True)
            self._thread.start()
            atexit.register(self.flush)
            log.info(f"Turso enabled ({self._url})")
            print(f"[turso] connected: {self._url}", flush=True)
        else:
            log.info("Turso not configured (TURSO_DATABASE_URL / TURSO_AUTH_TOKEN) - using local JSON files")
            print("[turso] NOT configured - set TURSO_DATABASE_URL and TURSO_AUTH_TOKEN. "
                  "Using local JSON files (data will NOT persist on hosts that wipe disk).", flush=True)

    # -- low level --
    def _pipeline(self, requests: list) -> list:
        body = json.dumps({"requests": requests + [{"type": "close"}]}).encode()
        req = urllib.request.Request(
            f"{self._url}/v2/pipeline", data=body, method="POST",
            headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            if e.code >= 500 or e.code == 429:
                raise TursoNetworkError(f"HTTP {e.code}: {detail}")
            raise TursoError(f"HTTP {e.code}: {detail}")
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            raise TursoNetworkError(str(e))
        results = data.get("results", [])
        for r in results:
            if r.get("type") == "error":
                raise TursoError(r.get("error", {}).get("message", "unknown error"))
        return results

    def execute(self, stmts: Sequence[Stmt]) -> None:
        """Run statements atomically (all or nothing)."""
        if not stmts:
            return
        steps = [{"stmt": {"sql": "BEGIN"}}]
        for i, (sql, args) in enumerate(stmts):
            steps.append({
                "stmt": {"sql": sql, "args": [_arg(a) for a in args]},
                "condition": {"type": "ok", "step": i},
            })
        last = len(stmts)
        steps.append({"stmt": {"sql": "COMMIT"}, "condition": {"type": "ok", "step": last}})
        steps.append({"stmt": {"sql": "ROLLBACK"},
                      "condition": {"type": "not", "cond": {"type": "ok", "step": last + 1}}})
        res = self._pipeline([{"type": "batch", "batch": {"steps": steps}}])
        result = res[0]["response"]["result"]
        errors = [e for e in result.get("step_errors", []) if e]
        if errors:
            raise TursoError(errors[0].get("message", "batch failed"))

    def query(self, sql: str, args: Sequence[Any] = ()) -> List[list]:
        res = self._pipeline([{"type": "execute",
                               "stmt": {"sql": sql, "args": [_arg(a) for a in args]}}])
        rows = res[0]["response"]["result"].get("rows", [])
        return [[_cell(c) for c in row] for row in rows]

    def _init_schema(self) -> None:
        self.execute([(s, ()) for s in SCHEMA])

    # -- async-ish writes --
    def submit(self, stmts: Sequence[Stmt]) -> None:
        """Queue statements to be written (atomically, in order) by the background thread."""
        if self.enabled and stmts:
            self._q.put(list(stmts))

    def _writer(self) -> None:
        while True:
            stmts = self._q.get()
            delay = 1.0
            while True:
                try:
                    self.execute(stmts)
                    break
                except TursoNetworkError as e:
                    log.warning(f"Turso write failed ({e}); retrying in {delay:.0f}s")
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                except TursoError as e:
                    log.error(f"Turso rejected a write, dropping it: {e}")
                    break
                except Exception:
                    log.exception("Unexpected error writing to Turso, dropping write")
                    break
            self._q.task_done()

    def flush(self, timeout: float = 15.0) -> None:
        """Block until queued writes are done (or timeout). Called automatically at exit."""
        if not self.enabled:
            return
        deadline = time.time() + timeout
        while self._q.unfinished_tasks and time.time() < deadline:
            time.sleep(0.05)
        if self._q.unfinished_tasks:
            log.warning(f"Turso: {self._q.unfinished_tasks} write(s) still pending at shutdown")

    # -- small key/value helpers (used for the backfill checkpoint + migration flags) --
    def kv_get(self, key: str) -> Optional[str]:
        rows = self.query("SELECT value FROM kv WHERE key = ?", [key])
        return rows[0][0] if rows else None

    def kv_set(self, key: str, value: str) -> None:
        self.execute([self._kv_stmt(key, value)])

    def kv_set_async(self, key: str, value: str) -> None:
        self.submit([self._kv_stmt(key, value)])

    @staticmethod
    def _kv_stmt(key: str, value: str) -> Stmt:
        return ("INSERT INTO kv(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value", [key, value])


db = TursoDB(os.getenv("TURSO_DATABASE_URL") or os.getenv("TURSO_URL"), os.getenv("TURSO_AUTH_TOKEN"))
