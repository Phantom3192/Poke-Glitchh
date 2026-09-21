"""
main.py - Watches for wild Pokemon spawn messages (e.g. from Poketwo)
and replies with the species name predicted by OUR model + DB.

How identification works: the spawn image is embedded into a 256-dim vector
with the trained EfficientNet-B0 feature extractor (predict.py / train_model.py)
and matched by cosine similarity against every stored embedding in the DB
(the "feature bank"). Because the bot only compares against that bank, it can
LEARN without retraining: teaching it a new example just adds one more vector
to the bank (saved to the DB and loaded into memory immediately).

Setup:
    1. pip install -r requirements.txt
    2. Set DISCORD_TOKEN (and TURSO_URL / TURSO_AUTH_TOKEN, or DB_PATH) as
       environment variables or in a .env file
    3. Make sure models/pokemon_classifier.pt is your real trained model
    4. python main.py

Commands (owner-only, prefix configurable via COMMAND_PREFIX, default "s!"):
    s!predict            - attach an image, or reply to a message with one, to
                           test identification without needing a live spawn
    s!learn <species>    - teach the bot: attach an image, or reply to a spawn /
                           image message, e.g.  s!learn pikachu
                           (new species: s!learn --new <name>)
    s!forget <species>   - remove the examples you taught for that species
    s!undo               - remove the most recent example you taught
    s!stats              - show how many species/features are loaded
    s!compare            - accuracy check: active backend vs PyTorch on one image
    s!api                - live performance dashboard (activity, active incense,
                           inference/response times, cache hit rate, Pokémon named)
    s!reload             - re-load the model + feature bank from disk/DB
    s!threshold X        - view/change the confidence threshold at runtime

Optional auto-learning (AUTO_LEARN=true): learns from the last spawn in a
channel if the bot guessed it wrong or wasn't confident. Where the "correct"
answer comes from is set by AUTO_LEARN_SOURCE:
    - "catch" (default): the spawn bot's own catch announcement
      ("... You caught a Level 12 Pikachu!").
    - "bot": trust another bot's identification message instead (e.g. a
      second spawn-guessing bot in the same channel). Set AUTO_LEARN_SOURCE_BOT_ID
      to that bot's Discord user ID, and adjust AUTO_LEARN_SOURCE_REGEX if its
      message format differs from the default "Species <emoji>: 99.86%" style.
      AUTO_LEARN_SOURCE_MIN_CONFIDENCE sets how confident that bot must claim
      to be before it's trusted.
See .env.example for all auto-learn variables.
"""

import os
import io
import re
import sys
import gc
import json
import time
import pickle
import hashlib
import uuid
import asyncio
import difflib
import logging
import threading
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Tuple

import aiohttp
import discord
import numpy as np
import torch
from discord.ext import commands
from PIL import Image
from dotenv import load_dotenv

from predict import load_extractor, load_feature_bank, predict_species
from onnx_backend import OnnxExtractor, onnx_matches_source
from train_model import Database, MODEL_OUTPUT

load_dotenv()

# train_model.py (imported above) silences logging in several ways:
#   - replaces logging.basicConfig with a no-op lambda
#   - disables every logger that existed at import time (incl. discord.py's)
#   - redirects sys.stderr to /dev/null
# So basicConfig() can't be used here. Undo it by attaching a stdout handler
# to the root logger directly and re-enabling the disabled loggers.
for _name in list(logging.root.manager.loggerDict.keys()):
    _lg = logging.getLogger(_name)
    _lg.disabled = False
    if _lg.level == logging.CRITICAL:
        _lg.setLevel(logging.NOTSET)

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"))


class _DropVoiceWarning(logging.Filter):
    """discord.py warns that voice isn't supported; irrelevant for this bot."""
    def filter(self, record):
        return "voice will NOT be supported" not in record.getMessage()


_handler.addFilter(_DropVoiceWarning())
_root = logging.getLogger()
_root.handlers = [_handler]
_root.setLevel(logging.INFO)
log = logging.getLogger("discord_bot")

# ── Config ───────────────────────────────────────────────────────────────
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise SystemExit("❌ DISCORD_TOKEN is not set (check your .env file).")

# Default is Poketwo's real bot ID, since that's the most common spawn
# source. Override via .env if you're scanning a different spawn bot.
SPAWN_BOT_ID = int(os.getenv("SPAWN_BOT_ID", "716390085896962058"))
COMMAND_PREFIX = os.getenv("COMMAND_PREFIX", "s!")
BOT_MODEL_PATH = os.getenv("BOT_MODEL_PATH", MODEL_OUTPUT)

# Cosine similarity score (0-1ish) required before the bot replies to a
# spawn automatically. Start conservative and tune down/up once you've
# watched real scores come through with s!predict or the logs.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
TOP_K = int(os.getenv("TOP_K", "5"))

# If true, the bot stays silent on spawns it isn't confident about instead
# of replying with a flagged low-confidence guess. Keep this off while
# you're first tuning CONFIDENCE_THRESHOLD so you can see real scores.
SILENT_BELOW_THRESHOLD = os.getenv("SILENT_BELOW_THRESHOLD", "false").lower() == "true"

# If the single closest stored example is at least this similar, trust it
# over the top-K majority vote. Without this, one freshly-taught example
# (a correction) could be outvoted by 4 older neighbours of another species.
NEAR_EXACT_SIM = float(os.getenv("NEAR_EXACT_SIM", "0.95"))

# s!learn refuses to store an example that is (almost) identical to one the
# species already has, so the bank doesn't fill up with duplicates.
LEARN_DUP_SIM = float(os.getenv("LEARN_DUP_SIM", "0.995"))

# ── Throughput tuning (many channels at once, e.g. 100 incense @ 20s) ───
# os.cpu_count() on Railway & co. reports the HOST's cores, not your quota, so
# torch's default thread count badly oversubscribes a 1 vCPU plan. Defaults
# below are for 1 vCPU; on N vCPU set INFER_WORKERS=N (keep TORCH_THREADS=1).
# BACKEND: "auto" = ONNX Runtime if it works (exported from the same .pt), else PyTorch;
# "onnx" = ONNX or refuse to start; "torch" = always PyTorch. With AUTO_EXPORT_ONNX the
# .onnx is (re)built on startup whenever it's missing or was made from a different .pt.
BACKEND = os.getenv("BACKEND", "auto").lower()
ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH") or os.path.splitext(BOT_MODEL_PATH)[0] + ".onnx"
AUTO_EXPORT_ONNX = os.getenv("AUTO_EXPORT_ONNX", "true").lower() == "true"
# Micro-batching: each worker groups up to BATCH_MAX queued images into one model call,
# waiting at most BATCH_WAIT_MS for stragglers (0 = only group what's already waiting, so
# it never adds latency). Measured on 1 thread it gave no speedup, so it's off (1) by default;
# try BATCH_MAX=4..8 on multi-core hosts and compare inference time in s!api.
BATCH_MAX = max(1, int(os.getenv("BATCH_MAX", "1")))
BATCH_WAIT_MS = max(0.0, float(os.getenv("BATCH_WAIT_MS", "0")))
INFER_WORKERS = max(1, int(os.getenv("INFER_WORKERS", "1")))
TORCH_THREADS = max(1, int(os.getenv("TORCH_THREADS", "1")))
# Skip a queued spawn if it waited longer than this (seconds) - the answer
# would arrive too late to be useful, and skipping lets the queue catch up.
MAX_SPAWN_AGE = float(os.getenv("MAX_SPAWN_AGE", "15"))
# Max cached image results (LRU). Same image bytes/URL -> no inference at all.
CACHE_SIZE = max(0, int(os.getenv("CACHE_SIZE", "5000")))
# Cache is saved here every CACHE_SAVE_INTERVAL seconds and re-loaded on start
# (only if the model + feature bank are unchanged), so restarts don't need a
# warm-up. Set CACHE_FILE="" to disable.
CACHE_FILE = os.getenv("CACHE_FILE", "spawn_cache.pkl")
# Also skip the download when the same image URL was seen before. Off by default: results are
# always cached by the image's exact bytes (provably identical input -> identical answer), a
# URL match is only an assumption that the URL always serves the same image.
URL_CACHE = os.getenv("URL_CACHE", "false").lower() == "true"
CACHE_SAVE_INTERVAL = float(os.getenv("CACHE_SAVE_INTERVAL", "60"))
# s!api dashboard: rolling window for timing stats, and how recently a channel
# must have spawned to count as an "active incense" channel.
METRICS_WINDOW = float(os.getenv("METRICS_WINDOW", "300"))
ACTIVE_INCENSE_WINDOW = float(os.getenv("ACTIVE_INCENSE_WINDOW", "60"))

torch.set_num_threads(TORCH_THREADS)
_executor = ThreadPoolExecutor(max_workers=INFER_WORKERS, thread_name_prefix="infer")

# ── Auto-learning (opt-in) ───────────────────────────────────────────────
AUTO_LEARN = os.getenv("AUTO_LEARN", "false").lower() == "true"
# Where the "ground truth" species comes from: "catch" (the spawn bot's own
# catch announcement) or "bot" (trust another bot's identification message).
AUTO_LEARN_SOURCE = os.getenv("AUTO_LEARN_SOURCE", "catch").strip().lower()
AUTO_LEARN_MAX_AGE = int(os.getenv("AUTO_LEARN_MAX_AGE", "600"))  # seconds

# -- AUTO_LEARN_SOURCE=catch --
# Must capture the species name in group 1. Default matches Poketwo-style
# "... You caught a Level 12 Pikachu! (20.1% IV)". Check your spawn bot's
# real catch message and adjust via the CATCH_REGEX env var if needed.
CATCH_REGEX = os.getenv("CATCH_REGEX", r"caught an? (?:level \d+ )?(.+?)\s*(?:!|\()")
_CATCH_RE = re.compile(CATCH_REGEX, re.IGNORECASE)

# -- AUTO_LEARN_SOURCE=bot --
# Discord user ID of the bot whose identification messages should be trusted
# as ground truth (e.g. another spawn-guessing bot posting in the same
# channel). Right-click / long-press the bot's name in Discord and "Copy
# User ID" (Developer Mode must be on: User Settings > Advanced).
AUTO_LEARN_SOURCE_BOT_ID = int(os.getenv("AUTO_LEARN_SOURCE_BOT_ID", "0") or 0)
# Must capture the species name as group "species" and its stated confidence
# (a bare number, no %) as group "confidence". Default matches messages like
# "Timburr 🔶: 99.86%" / "Timburr: 99.86%\nBest name: Timburr".
AUTO_LEARN_SOURCE_REGEX = os.getenv(
    "AUTO_LEARN_SOURCE_REGEX",
    r"^(?P<species>[A-Za-z][A-Za-z.\-' ]*?)\s+\S*:\s*(?P<confidence>[\d.]+)\s*%",
)
# Only trust the source bot's guess (and learn from it) if it claims at least
# this much confidence in its own answer.
AUTO_LEARN_SOURCE_MIN_CONFIDENCE = float(os.getenv("AUTO_LEARN_SOURCE_MIN_CONFIDENCE", "0.9"))
_AUTO_LEARN_SOURCE_RE = re.compile(AUTO_LEARN_SOURCE_REGEX, re.IGNORECASE) if AUTO_LEARN_SOURCE == "bot" else None

if AUTO_LEARN and AUTO_LEARN_SOURCE == "bot" and not AUTO_LEARN_SOURCE_BOT_ID:
    raise RuntimeError(
        "AUTO_LEARN_SOURCE=bot requires AUTO_LEARN_SOURCE_BOT_ID to be set in your .env "
        "to the Discord user ID of the bot to learn from."
    )
if AUTO_LEARN and AUTO_LEARN_SOURCE not in ("catch", "bot"):
    raise RuntimeError(f"AUTO_LEARN_SOURCE must be 'catch' or 'bot', got {AUTO_LEARN_SOURCE!r}")

# Rows added by s!learn / auto-learn carry this marker in variant_name, which
# is how s!forget / s!undo tell them apart from the original training data.
LEARNED_MARK = "__learned_"

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix=COMMAND_PREFIX, intents=intents)

# ── Model + feature bank (loaded once, refreshed via s!reload) ─────────────
# _species_list / _matrix are always REPLACED (never mutated in place) while
# holding _bank_lock, so worker threads can grab a consistent snapshot.
_bank_lock = threading.RLock()
_extractor = None
_species_list = None
_matrix = None
_db: Optional[Database] = None
_session: Optional[aiohttp.ClientSession] = None

# channel_id -> (timestamp, guessed species, score, query vector) for the most
# recent spawn we successfully analysed; None if we saw a spawn but couldn't.
_last_spawn: Dict[int, Optional[Tuple[float, str, float, np.ndarray]]] = {}

# channel_id -> generation counter; a newer spawn in a channel supersedes older
# queued ones (they're skipped instead of wasting CPU on an outdated image).
_spawn_gen: Dict[int, int] = {}

# Result cache. Cleared whenever the feature bank changes (learn/forget/undo/
# reload) so corrections always take effect. _bank_version stops a worker that
# started before a change from writing a stale result afterwards.
_cache_lock = threading.Lock()
_result_cache: "OrderedDict[str, tuple]" = OrderedDict()   # sha1(image bytes) -> result tuple
_url_cache: "OrderedDict[str, str]" = OrderedDict()        # image url -> sha1
_bank_version = 0
_stats = {"hits": 0, "misses": 0, "coalesced": 0, "dropped": 0, "batches": 0, "batch_items": 0}
_backend_name = "PyTorch CPU"
_cache_dirty = False
# In-flight work, so N channels spawning the same image at once cost ONE download/inference.
_inflight_url: Dict[str, dict] = {}
_inflight_digest: Dict[str, dict] = {}
_saver_started = False


def _cache_clear():
    global _bank_version, _cache_dirty
    with _cache_lock:
        _bank_version += 1
        _cache_dirty = False
        _result_cache.clear()
        _url_cache.clear()


def _cache_version() -> int:
    with _cache_lock:
        return _bank_version


def _cache_get(digest: str):
    with _cache_lock:
        res = _result_cache.get(digest)
        if res is not None:
            _result_cache.move_to_end(digest)
        return res


def _cache_get_by_url(url: str):
    with _cache_lock:
        digest = _url_cache.get(url)
        if digest is None:
            return None
        res = _result_cache.get(digest)
        if res is not None:
            _result_cache.move_to_end(digest)
            _url_cache.move_to_end(url)
        return res


def _cache_put(digest: str, result: tuple, version: int, url: Optional[str] = None):
    global _cache_dirty
    if CACHE_SIZE <= 0:
        return
    with _cache_lock:
        if version != _bank_version:
            return
        _cache_dirty = True
        _result_cache[digest] = result
        _result_cache.move_to_end(digest)
        while len(_result_cache) > CACHE_SIZE:
            _result_cache.popitem(last=False)
        if url:
            _url_cache[url] = digest
            _url_cache.move_to_end(url)
            while len(_url_cache) > CACHE_SIZE:
                _url_cache.popitem(last=False)


async def _in_executor(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(_executor, fn, *args)


# ── s!api metrics (rolling window) ───────────────────────────────────────
_metrics_lock = threading.Lock()
_m_inference: deque = deque()   # (ts, ms) model forward pass, one entry per real inference
_m_store: deque = deque()       # (ts, ms) feature-bank lookup + vote
_m_response: deque = deque()    # (ts, ms) spawn seen -> reply sent
_m_spawns: deque = deque()      # (ts, 1)  every spawn seen
_channel_last_spawn: Dict[int, float] = {}
_named_total = 0


def _m_add(dq: deque, value):
    now = time.time()
    with _metrics_lock:
        dq.append((now, value))
        cutoff = now - METRICS_WINDOW
        while dq and dq[0][0] < cutoff:
            dq.popleft()


def _m_values(dq: deque, since: Optional[float] = None):
    cutoff = time.time() - (METRICS_WINDOW if since is None else since)
    with _metrics_lock:
        return [v for ts, v in dq if ts >= cutoff]


def _fmt_ms(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v / 1000:.3f}s" if v >= 1000 else f"{v:.0f} ms"


def _activity(rate: float):
    """rate = spawns/sec over the last minute -> (label, embed colour name)."""
    if rate <= 0:
        return "⚪ Idle", "light_grey"
    if rate < 2:
        return "🟢 Low Activity", "green"
    if rate < 6:
        return "🟡 Moderate Activity", "gold"
    if rate < 15:
        return "🟠 High Activity", "orange"
    return "🔴 Very High Activity", "red"


def _api_snapshot() -> dict:
    now = time.time()
    inf, store, resp = _m_values(_m_inference), _m_values(_m_store), _m_values(_m_response)
    spawns = len(_m_values(_m_spawns))
    rate = len(_m_values(_m_spawns, since=60)) / 60.0
    for cid in [c for c, t in _channel_last_spawn.items() if now - t > METRICS_WINDOW]:
        del _channel_last_spawn[cid]
    active = sum(1 for t in _channel_last_spawn.values() if now - t <= ACTIVE_INCENSE_WINDOW)

    def stats(vals):
        return (min(vals), max(vals), sum(vals) / len(vals), vals[-1]) if vals else (None,) * 4

    label, color = _activity(rate)
    _, _, matrix = _snapshot()
    return {
        "activity": label, "color": color, "rate": rate, "active": active,
        "hit_rate": None if spawns == 0 else max(0.0, 1.0 - len(inf) / spawns),
        "store": stats(store), "inference": stats(inf), "response": stats(resp),
        "bank": int(matrix.shape[0]), "named": _named_total,
    }


def _build_api_embed() -> discord.Embed:
    d = _api_snapshot()
    embed = discord.Embed(title="API Status", color=getattr(discord.Color, d["color"])())

    def field(name, value):
        embed.add_field(name=name, value=value, inline=True)

    s_min, s_max, s_avg, s_last = d["store"]
    i_min, i_max, i_avg, i_last = d["inference"]
    r_min, r_max, r_avg, r_last = d["response"]
    field("Activity", d["activity"])
    field("Active Incense", f"`{d['active']}`")
    field("Cache Hit Rate", "—" if d["hit_rate"] is None else f"{d['hit_rate'] * 100:.0f}%")
    field("Store (latest)", _fmt_ms(s_last))
    field("Store (avg)", _fmt_ms(s_avg))
    field("Bank Size", f"{d['bank']:,}")
    field("Inference Fastest", _fmt_ms(i_min))
    field("Inference Slowest", _fmt_ms(i_max))
    field("Inference Avg", _fmt_ms(i_avg))
    field("Inference Latest", _fmt_ms(i_last))
    field("Response Fastest", _fmt_ms(r_min))
    field("Response Slowest", _fmt_ms(r_max))
    field("Response Avg", _fmt_ms(r_avg))
    field("Response Latest", _fmt_ms(r_last))
    field("Pokémons Named", f"{d['named']:,}")
    embed.set_footer(
        text=f"Model: {os.path.basename(BOT_MODEL_PATH)} | Backend: {_backend_name} "
             f"(threads={TORCH_THREADS}, workers={INFER_WORKERS}, batch≤{BATCH_MAX}) | "
             f"Window: {METRICS_WINDOW / 60:g} min rolling"
    )
    return embed


def _bank_fingerprint():
    """Identifies the exact model + feature bank a cached result was computed against."""
    with _bank_lock:
        rows = int(_matrix.shape[0])
        total = round(float(_matrix.sum(dtype=np.float64)), 4)
    try:
        model_size = os.path.getsize(BOT_MODEL_PATH)
    except OSError:
        model_size = -1
    return (rows, total, model_size)


def _cache_save():
    global _cache_dirty
    if not CACHE_FILE or CACHE_SIZE <= 0:
        return
    version = _cache_version()
    fingerprint = _bank_fingerprint()  # taken outside _cache_lock (lock order: bank -> cache)
    with _cache_lock:
        if not _cache_dirty or version != _bank_version:
            return
        payload = {"fp": fingerprint, "results": list(_result_cache.items()), "urls": list(_url_cache.items())}
        _cache_dirty = False
    tmp = CACHE_FILE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, CACHE_FILE)


def _cache_load():
    if not CACHE_FILE or CACHE_SIZE <= 0 or not os.path.exists(CACHE_FILE):
        return
    try:
        with open(CACHE_FILE, "rb") as f:
            payload = pickle.load(f)
        if payload.get("fp") != _bank_fingerprint():
            log.info("🗂️  Saved spawn cache doesn't match the current model/bank — ignoring it")
            return
        with _cache_lock:
            for key, val in payload["results"][-CACHE_SIZE:]:
                _result_cache[key] = val
            for url, digest in payload["urls"][-CACHE_SIZE:]:
                if digest in _result_cache:
                    _url_cache[url] = digest
            n = len(_result_cache)
        log.info(f"🗂️  Loaded {n} cached spawn results from {CACHE_FILE}")
    except Exception as e:
        log.warning(f"Couldn't load spawn cache ({type(e).__name__}: {e}) — starting empty")


async def _cache_saver():
    while True:
        await asyncio.sleep(CACHE_SAVE_INTERVAL)
        try:
            await asyncio.to_thread(_cache_save)
        except Exception as e:
            log.warning(f"Couldn't save spawn cache ({type(e).__name__}: {e})")


async def _coalesced(table: Dict[str, dict], key: str, received: float, work, deps=None):
    """Run work(entry) once per key at a time; concurrent callers with the same key share its result."""
    entry = table.get(key)
    if entry is not None:
        entry["latest"] = max(entry["latest"], received)
        _stats["coalesced"] += 1
        return await asyncio.shield(entry["fut"])
    entry = {"fut": asyncio.get_running_loop().create_future(), "latest": received, "deps": list(deps or [])}
    table[key] = entry
    result = None
    try:
        result = await work(entry)
    except Exception as e:
        log.error(f"Identification failed: {type(e).__name__}: {e}")
    finally:
        table.pop(key, None)
        if not entry["fut"].done():
            entry["fut"].set_result(result)
    return result


def _entry_latest(entry: dict) -> float:
    latest = entry["latest"]
    for dep in entry.get("deps", ()):
        latest = max(latest, _entry_latest(dep))
    return latest


async def _identify_spawn(url: str, received: float):
    """Returns (winner, score, neighbors, query_vec) or None. Cache -> coalesce -> download -> hash -> infer."""
    res = _cache_get_by_url(url) if URL_CACHE else None
    if res is not None:
        _stats["hits"] += 1
        return res
    cache_url = url if URL_CACHE else None

    async def work(url_entry):
        image_bytes = await _download(url)
        if not image_bytes:
            return None
        digest = hashlib.sha1(image_bytes).hexdigest()
        version = _cache_version()
        res = _cache_get(digest)
        if res is not None:
            _stats["hits"] += 1
            _cache_put(digest, res, version, cache_url)
            return res

        async def infer(digest_entry):
            out = await _submit_identification(image_bytes, digest_entry)
            if out is not None:
                _stats["misses"] += 1
            return out

        res = await _coalesced(_inflight_digest, digest, received, infer, deps=[url_entry])
        if res is not None:
            _cache_put(digest, res, version, cache_url)
        return res

    return await _coalesced(_inflight_url, url, received, work)


def _try_onnx():
    """Returns an OnnxExtractor that matches BOT_MODEL_PATH, or None (caller falls back to PyTorch)."""
    global _backend_name
    try:
        state = onnx_matches_source(ONNX_MODEL_PATH, BOT_MODEL_PATH)  # True / False / None (no sidecar)
        exists = os.path.exists(ONNX_MODEL_PATH)
        if not exists or state is not True:  # missing, made from a different .pt, or no sidecar to prove it
            if not AUTO_EXPORT_ONNX:
                why = "can't be matched to the current .pt" if exists else "not found"
                log.warning(f"ONNX model {why} and AUTO_EXPORT_ONNX is off")
                return None
            log.info("🔧 Exporting ONNX model (one-time, ~10-30s) ...")
            from export_onnx import export_to_onnx
            torch_ex = load_extractor(BOT_MODEL_PATH)
            stats = export_to_onnx(torch_ex, ONNX_MODEL_PATH, BOT_MODEL_PATH)
            log.info(f"   ✅ ONNX verified vs PyTorch (min cosine {stats['min_cos']:.6f})")
            del torch_ex
            gc.collect()
        ex = OnnxExtractor(ONNX_MODEL_PATH, threads=TORCH_THREADS)
        _backend_name = "ONNX Runtime"
        return ex
    except Exception as e:
        log.warning(f"ONNX backend unavailable ({type(e).__name__}: {e})")
        return None


def _load_extractor():
    global _backend_name
    mode = BACKEND if BACKEND in ("auto", "onnx", "torch") else "auto"
    if mode != "torch":
        ex = _try_onnx()
        if ex is not None:
            return ex
        if mode == "onnx":
            raise SystemExit("❌ BACKEND=onnx but the ONNX model couldn't be loaded (see the warning above).")
        log.info("   Falling back to PyTorch")
    _backend_name = "PyTorch CPU"
    return load_extractor(BOT_MODEL_PATH)


def _load_model_and_bank():
    global _extractor, _species_list, _matrix, _db
    log.info(f"📦 Loading model from {BOT_MODEL_PATH} ...")
    extractor = _load_extractor()
    log.info(f"   Backend: {_backend_name}")

    log.info("🗄️  Connecting to database and loading feature bank ...")
    new_db = Database()
    species_list, matrix = load_feature_bank(new_db)

    with _bank_lock:
        old_db = _db
        _extractor, _db, _species_list, _matrix = extractor, new_db, species_list, matrix
    _cache_clear()
    if old_db is not None:
        old_db.close()
    log.info(f"   Loaded {matrix.shape[0]} feature vectors across {len(set(species_list))} species")


async def _get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    return _session


async def _download(url: str) -> Optional[bytes]:
    try:
        sess = await _get_session()
        async with sess.get(url, timeout=aiohttp.ClientTimeout(total=15, connect=5)) as resp:
            if resp.status == 200:
                return await resp.read()
            log.warning(f"Image download failed ({resp.status}): {url}")
            return None
    except Exception as e:
        log.warning(f"Image download error ({type(e).__name__}): {e}")
        return None


def _is_wild_spawn_embed(embed: discord.Embed) -> bool:
    """
    Heuristic spawn detector, mirroring how most spawn bots phrase it
    ("A wild pokemon has appeared!" style embeds). Loosen/tighten this
    if your target spawn bot phrases things differently.
    """
    title = (embed.title or "").lower()
    description = (embed.description or "").lower()
    footer_text = (embed.footer.text or "").lower() if embed.footer else ""
    combined = f"{title} {description} {footer_text}"
    return ("wild" in combined and "appeared" in combined) or ("guess the" in combined and "catch" in combined)


def _extract_wild_spawn_image_url(message: discord.Message) -> Optional[str]:
    for embed in message.embeds:
        if not _is_wild_spawn_embed(embed):
            continue
        if embed.image and embed.image.url:
            return embed.image.url
        if embed.thumbnail and embed.thumbnail.url:
            return embed.thumbnail.url
    return None


# ── Identification ───────────────────────────────────────────────────────

def _snapshot():
    with _bank_lock:
        return _extractor, _species_list, _matrix


def _embed_image_bytes(image_bytes: bytes) -> np.ndarray:
    extractor, _, _ = _snapshot()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return extractor.extract(img)


def _predict(query_vec: np.ndarray, species_list, matrix):
    """predict_species() + trust a near-exact single match over the majority vote."""
    winner, score, neighbors = predict_species(query_vec, species_list, matrix, top_k=TOP_K)
    top_species, top_sim = neighbors[0]
    if top_species != winner and top_sim >= NEAR_EXACT_SIM:
        winner, score = top_species, top_sim
    return winner, score, neighbors


def _run_identification(image_bytes: bytes):
    """Runs in a worker thread — torch inference + numpy matmul are blocking/CPU-bound."""
    _, species_list, matrix = _snapshot()
    t0 = time.perf_counter()
    query_vec = _embed_image_bytes(image_bytes)
    t1 = time.perf_counter()
    winner, score, neighbors = _predict(query_vec, species_list, matrix)
    t2 = time.perf_counter()
    _m_add(_m_inference, (t1 - t0) * 1000)
    _m_add(_m_store, (t2 - t1) * 1000)
    return winner, score, neighbors, query_vec


def _run_identification_batch(images_bytes):
    """Worker thread: embed a group of images in one model call, then match each. None = unreadable image."""
    extractor, species_list, matrix = _snapshot()
    imgs = []
    for b in images_bytes:
        try:
            imgs.append(Image.open(io.BytesIO(b)).convert("RGB"))
        except Exception:
            imgs.append(None)
    n_valid = sum(1 for im in imgs if im is not None)
    t0 = time.perf_counter()
    if isinstance(extractor, OnnxExtractor):
        vecs = extractor.extract_batch(imgs)  # row i always belongs to imgs[i]
    else:  # PyTorch fallback: one image at a time (its extract_batch can misalign on failures)
        vecs = [extractor.extract(im) if im is not None else None for im in imgs]
    per_image_ms = (time.perf_counter() - t0) * 1000 / max(1, n_valid)

    results = []
    for im, vec in zip(imgs, vecs):
        if im is None or vec is None or float(np.linalg.norm(vec)) < 1e-6:
            results.append(None)
            continue
        t1 = time.perf_counter()
        winner, score, neighbors = _predict(vec, species_list, matrix)
        _m_add(_m_store, (time.perf_counter() - t1) * 1000)
        _m_add(_m_inference, per_image_ms)
        results.append((winner, score, neighbors, vec))
    return results


_batch_queue: Optional[asyncio.Queue] = None
_batch_tasks = []


async def _batch_worker():
    loop = asyncio.get_running_loop()
    while True:
        batch = [await _batch_queue.get()]
        deadline = loop.time() + BATCH_WAIT_MS / 1000.0
        while len(batch) < BATCH_MAX:
            try:
                batch.append(_batch_queue.get_nowait())
                continue
            except asyncio.QueueEmpty:
                pass
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(_batch_queue.get(), remaining))
            except asyncio.TimeoutError:
                break

        live = []
        for image_bytes, entry, fut in batch:
            if fut.done():
                continue
            if time.time() - _entry_latest(entry) > MAX_SPAWN_AGE:  # too old to be useful
                _stats["dropped"] += 1
                fut.set_result(None)
                continue
            live.append((image_bytes, fut))
        if not live:
            continue
        try:
            results = await loop.run_in_executor(_executor, _run_identification_batch, [b for b, _ in live])
        except Exception as e:
            log.error(f"Batch inference failed: {type(e).__name__}: {e}")
            results = [None] * len(live)
        _stats["batches"] += 1
        _stats["batch_items"] += len(live)
        for (_, fut), res in zip(live, results):
            if not fut.done():
                fut.set_result(res)


async def _submit_identification(image_bytes: bytes, entry: dict):
    """Queue an image for the (micro-batching) inference workers; resolves to a result tuple or None."""
    global _batch_queue
    if _batch_queue is None:
        _batch_queue = asyncio.Queue()
        _batch_tasks.extend(asyncio.create_task(_batch_worker()) for _ in range(INFER_WORKERS))
    fut = asyncio.get_running_loop().create_future()
    _batch_queue.put_nowait((image_bytes, entry, fut))
    return await fut


# ── Learning ─────────────────────────────────────────────────────────────

def _key(name: str) -> str:
    """Loose comparison key so 'Mr. Mime', 'mr_mime' and 'mr._mime' all match."""
    return re.sub(r"[^a-z0-9♀♂]+", "", name.lower())


def _new_species_name(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip().lower())


def _display(species: str) -> str:
    return species.replace("_", " ").title()


def _find_species(name: str) -> Optional[str]:
    key = _key(name)
    with _bank_lock:
        for sp in set(_species_list):
            if _key(sp) == key:
                return sp
    return None


def _db_insert_learned(species: str, vec: np.ndarray):
    conn = _db._conn
    cur = conn.cursor()
    try:
        variant = f"{species}{LEARNED_MARK}{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        cur.execute(
            "INSERT INTO pokemon_features (species, variant_name, feature_vector, created_at) "
            "VALUES (?, ?, ?, strftime('%s', 'now'))",
            (species, variant, json.dumps(vec.tolist())),
        )
        cur.execute(
            "INSERT INTO species_info (species, count, last_updated) "
            "VALUES (?, 1, strftime('%s', 'now')) "
            "ON CONFLICT(species) DO UPDATE SET count = count + 1, last_updated = strftime('%s', 'now')",
            (species,),
        )
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            cur.close()
        except Exception:
            pass


def _learn_sync(name: str, vec: np.ndarray, allow_new: bool = False):
    """
    Adds one example to the DB and the in-memory bank.
    Returns (status, species, extra):
      ("learned", species, examples_for_species)
      ("duplicate", species, None)   - bank already has (almost) this exact image
      ("unknown", None, suggestions) - species not in the bank and allow_new=False
      ("bad_image", None, None)      - feature extraction failed (zero vector)
    """
    global _species_list, _matrix
    vec = np.asarray(vec, dtype=np.float32).flatten()
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return "bad_image", None, None
    vec = vec / norm

    with _bank_lock:
        species = _find_species(name)
        if species is None:
            if not allow_new:
                close = difflib.get_close_matches(
                    _new_species_name(name), sorted(set(_species_list)), n=3, cutoff=0.6
                )
                return "unknown", None, close
            species = _new_species_name(name)
        else:
            idx = [i for i, s in enumerate(_species_list) if s == species]
            if idx and float((_matrix[idx] @ vec).max()) >= LEARN_DUP_SIM:
                return "duplicate", species, None

        _db_insert_learned(species, vec)  # DB first, so memory never gets ahead of it
        _matrix = np.vstack([_matrix, vec[None, :]])
        _species_list = _species_list + [species]
        _cache_clear()
        return "learned", species, sum(1 for s in _species_list if s == species)


def _delete_learned(species: Optional[str] = None, last_only: bool = False):
    """Removes learned examples (never the original training data). Returns removed species names."""
    global _species_list, _matrix
    with _bank_lock:
        conn = _db._conn
        cur = conn.cursor()
        try:
            query = "SELECT id, species FROM pokemon_features WHERE instr(variant_name, ?) > 0"
            params = [LEARNED_MARK]
            if species:
                query += " AND species = ?"
                params.append(species)
            query += " ORDER BY id DESC"
            if last_only:
                query += " LIMIT 1"
            cur.execute(query, params)
            rows = [(r[0], r[1]) for r in cur.fetchall()]
            for row_id, sp in rows:
                cur.execute("DELETE FROM pokemon_features WHERE id = ?", (row_id,))
                cur.execute("UPDATE species_info SET count = MAX(count - 1, 0) WHERE species = ?", (sp,))
            conn.commit()
        finally:
            try:
                cur.close()
            except Exception:
                pass
        if rows:
            _species_list, _matrix = load_feature_bank(_db)
            _cache_clear()
        return [sp for _, sp in rows]


def _count_learned() -> int:
    with _bank_lock:
        cur = _db._conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM pokemon_features WHERE instr(variant_name, ?) > 0", (LEARNED_MARK,))
            return int(cur.fetchone()[0])
        finally:
            try:
                cur.close()
            except Exception:
                pass


def _message_text(message: discord.Message) -> str:
    parts = [message.content or ""]
    for embed in message.embeds:
        parts.append(embed.title or "")
        parts.append(embed.description or "")
    return "\n".join(parts)


async def _maybe_auto_learn(message: discord.Message):
    """If this is a 'X caught a Level N <Species>!' message, learn from the last spawn in the channel."""
    text = _message_text(message)
    if "caught" not in text.lower():
        return
    match = _CATCH_RE.search(text)
    if not match:
        log.info(f"Auto-learn: saw a 'caught' message that CATCH_REGEX didn't match: {text[:160]!r}")
        return

    caught = match.group(1).strip()
    entry = _last_spawn.get(message.channel.id)
    if not entry:
        log.info(f"Auto-learn: '{caught}' was caught but I have no analysed spawn for this channel — skipping")
        return
    spawned_at, guess, score, vec = entry
    if time.time() - spawned_at > AUTO_LEARN_MAX_AGE:
        log.info(f"Auto-learn: last spawn is too old to trust — skipping '{caught}'")
        return
    _last_spawn[message.channel.id] = None  # learn from each spawn at most once

    species = _find_species(caught)
    if species is None:
        log.warning(f"Auto-learn: '{caught}' isn't a species in my DB — use s!learn --new {caught.lower()} if it's real")
        return
    if species == guess and score >= CONFIDENCE_THRESHOLD:
        log.info(f"Auto-learn: guessed {species} correctly ({score:.3f}) — nothing to learn")
        return

    status, name, extra = await asyncio.to_thread(_learn_sync, species, vec, False)
    verdict = "wrong" if species != guess else "low confidence"
    log.info(f"Auto-learn: caught={species}, I guessed {guess} ({score:.3f}, {verdict}) -> {status}"
             + (f" ({extra} examples now)" if status == "learned" else ""))


async def _maybe_auto_learn_from_source_bot(message: discord.Message):
    """AUTO_LEARN_SOURCE=bot: trust AUTO_LEARN_SOURCE_BOT_ID's identification
    message as ground truth and learn from the last spawn in this channel if
    our own guess disagreed with it (or wasn't confident)."""
    text = _message_text(message)
    match = _AUTO_LEARN_SOURCE_RE.search(text)
    if not match:
        log.info(f"Auto-learn (source bot): message didn't match AUTO_LEARN_SOURCE_REGEX: {text[:160]!r}")
        return

    claimed = match.group("species").strip()
    try:
        source_conf = float(match.group("confidence")) / 100.0
    except (IndexError, ValueError):
        source_conf = None

    if source_conf is not None and source_conf < AUTO_LEARN_SOURCE_MIN_CONFIDENCE:
        log.info(f"Auto-learn (source bot): '{claimed}' only {source_conf:.1%} confident — not trusted enough to learn from")
        return

    entry = _last_spawn.get(message.channel.id)
    if not entry:
        log.info(f"Auto-learn (source bot): '{claimed}' named but I have no analysed spawn for this channel — skipping")
        return
    spawned_at, guess, score, vec = entry
    if time.time() - spawned_at > AUTO_LEARN_MAX_AGE:
        log.info(f"Auto-learn (source bot): last spawn is too old to trust — skipping '{claimed}'")
        return
    _last_spawn[message.channel.id] = None  # learn from each spawn at most once

    species = _find_species(claimed)
    if species is None:
        log.warning(f"Auto-learn (source bot): '{claimed}' isn't a species in my DB — use s!learn --new {claimed.lower()} if it's real")
        return
    if species == guess and score >= CONFIDENCE_THRESHOLD:
        log.info(f"Auto-learn (source bot): guessed {species} correctly ({score:.3f}) — nothing to learn")
        return

    status, name, extra = await asyncio.to_thread(_learn_sync, species, vec, False)
    verdict = "wrong" if species != guess else "low confidence"
    log.info(f"Auto-learn (source bot): {message.author} said {species} ({source_conf if source_conf is not None else '?'}), "
             f"I guessed {guess} ({score:.3f}, {verdict}) -> {status}"
             + (f" ({extra} examples now)" if status == "learned" else ""))


# ── Discord events ───────────────────────────────────────────────────────

@bot.event
async def on_ready():
    global _saver_started
    await _get_session()
    if CACHE_FILE and CACHE_SIZE > 0 and not _saver_started:
        _saver_started = True
        asyncio.create_task(_cache_saver())
    log.info("=" * 60)
    log.info(f"🤖 Logged in as {bot.user} — scanning for spawns from bot ID {SPAWN_BOT_ID}")
    log.info(f"   Confidence threshold: {CONFIDENCE_THRESHOLD}")
    if AUTO_LEARN:
        source_desc = (
            f"ON (source: bot #{AUTO_LEARN_SOURCE_BOT_ID})"
            if AUTO_LEARN_SOURCE == "bot" else "ON (source: catch announcements)"
        )
    else:
        source_desc = "OFF"
    log.info(f"   Auto-learn: {source_desc}")
    log.info("=" * 60)


@bot.event
async def on_message(message: discord.Message):
    global _named_total
    await bot.process_commands(message)

    if AUTO_LEARN and AUTO_LEARN_SOURCE == "bot" and message.author.id == AUTO_LEARN_SOURCE_BOT_ID:
        try:
            await _maybe_auto_learn_from_source_bot(message)
        except Exception as e:
            log.error(f"Auto-learn (source bot) failed: {type(e).__name__}: {e}")

    if message.author.id != SPAWN_BOT_ID:
        return

    if AUTO_LEARN and AUTO_LEARN_SOURCE == "catch":
        try:
            await _maybe_auto_learn(message)
        except Exception as e:
            log.error(f"Auto-learn failed: {type(e).__name__}: {e}")

    image_url = _extract_wild_spawn_image_url(message)
    if not image_url:
        return

    received = time.time()
    cid = message.channel.id
    _m_add(_m_spawns, 1)
    _channel_last_spawn[cid] = received
    gen = _spawn_gen.get(cid, 0) + 1
    _spawn_gen[cid] = gen  # a newer spawn in this channel supersedes this one if it's still working

    # A new spawn replaces the old one, so forget the previous spawn now; it's
    # only re-set below if we manage to analyse this one (avoids learning a
    # catch against the wrong image).
    _last_spawn[cid] = None

    result = await _identify_spawn(image_url, received)
    if result is None:
        return
    if _spawn_gen.get(cid) != gen:
        return  # superseded while we were working

    winner, score, neighbors, query_vec = result

    _last_spawn[message.channel.id] = (time.time(), winner, score, query_vec)

    elapsed_ms = (time.time() - received) * 1000
    confident = score >= CONFIDENCE_THRESHOLD
    log.info(f"Spawn {message.id}: {winner} (score {score:.3f}, confident={confident}, {elapsed_ms:.0f}ms)")

    if not confident and SILENT_BELOW_THRESHOLD:
        return

    display_name = _display(winner)
    if confident:
        reply_text = f"{display_name} — {score * 100:.1f}%"
    else:
        reply_text = f"{display_name}? — {score * 100:.1f}% (low confidence, might be wrong)"

    try:
        await message.reply(reply_text, allowed_mentions=discord.AllowedMentions.none())
    except discord.HTTPException as e:
        log.warning(f"Failed to reply to spawn message {message.id}: {e}")
        return
    _named_total += 1
    _m_add(_m_response, (time.time() - received) * 1000)


# ── Manual testing / admin commands ─────────────────────────────────────

async def _resolve_message_image(ctx: commands.Context, no_image_error: str):
    for att in ctx.message.attachments:
        if att.content_type and att.content_type.startswith("image/"):
            return await att.read(), None

    if ctx.message.reference is not None:
        replied = ctx.message.reference.resolved
        if not isinstance(replied, discord.Message):
            try:
                replied = await ctx.channel.fetch_message(ctx.message.reference.message_id)
            except discord.HTTPException:
                replied = None
        if replied is not None:
            for att in replied.attachments:
                if att.content_type and att.content_type.startswith("image/"):
                    return await att.read(), None
            image_url = _extract_wild_spawn_image_url(replied) or (
                replied.embeds[0].image.url if replied.embeds and replied.embeds[0].image else None
            )
            if image_url:
                image_bytes = await _download(image_url)
                if image_bytes:
                    return image_bytes, None

    return None, no_image_error


@bot.command(name="predict")
@commands.is_owner()
async def predict_cmd(ctx: commands.Context):
    """Manually test identification: attach an image or reply to one with s!predict"""
    image_bytes, error = await _resolve_message_image(
        ctx, no_image_error="⚠️ Attach an image, or reply to a message that has one, when using this command."
    )
    if image_bytes is None:
        await ctx.send(error)
        return

    start = time.time()
    winner, score, neighbors, _ = await _in_executor(_run_identification, image_bytes)
    elapsed_ms = (time.time() - start) * 1000

    lines = [f"**Best guess:** {_display(winner)} — {score * 100:.1f}%"]
    lines.append(f"Processed in {elapsed_ms:.0f}ms\n")
    lines.append("**Top matches:**")
    for sp, sim in neighbors:
        lines.append(f"  {sp:<20s} {sim * 100:.1f}%")

    embed = discord.Embed(
        title="🔍 Predict Result",
        description="\n".join(lines),
        color=discord.Color.green() if score >= CONFIDENCE_THRESHOLD else discord.Color.orange(),
    )
    await ctx.send(embed=embed)


@bot.command(name="learn")
@commands.is_owner()
async def learn_cmd(ctx: commands.Context, *, species: str):
    """Teach the bot: attach an image or reply to one with s!learn <species> (new species: s!learn --new <name>)"""
    allow_new = False
    if species.lower().startswith("--new "):
        allow_new = True
        species = species[6:].strip()

    image_bytes, error = await _resolve_message_image(
        ctx, no_image_error="⚠️ Attach an image, or reply to a spawn/image message, when using s!learn."
    )
    if image_bytes is None:
        await ctx.send(error)
        return

    try:
        vec = await asyncio.to_thread(_embed_image_bytes, image_bytes)
    except Exception as e:
        await ctx.send(f"❌ Couldn't read that image: {type(e).__name__}")
        return

    try:
        status, name, extra = await asyncio.to_thread(_learn_sync, species, vec, allow_new)
    except Exception as e:
        log.error(f"s!learn failed: {type(e).__name__}: {e}")
        await ctx.send(f"❌ Couldn't save that example: {type(e).__name__}: {e}")
        return

    if status == "bad_image":
        await ctx.send("❌ Couldn't extract features from that image, so nothing was saved.")
    elif status == "unknown":
        hint = f" Did you mean: {', '.join(_display(s) for s in extra)}?" if extra else ""
        await ctx.send(
            f"❓ I don't know a species called **{species}**.{hint}\n"
            f"If it really is a new species, use `{COMMAND_PREFIX}learn --new {species}`."
        )
    elif status == "duplicate":
        await ctx.send(f"ℹ️ I already have this exact image saved for **{_display(name)}** — nothing to add.")
    else:
        _, species_list, matrix = _snapshot()
        winner, score, _ = _predict(vec, species_list, matrix)
        log.info(f"Learned 1 example of {name} ({extra} now, {matrix.shape[0]} total)")
        await ctx.send(
            f"✅ Learned **{_display(name)}** — now {extra} example(s) for it ({matrix.shape[0]} total).\n"
            f"This image now identifies as **{_display(winner)}** ({score * 100:.1f}%)."
        )


@bot.command(name="forget")
@commands.is_owner()
async def forget_cmd(ctx: commands.Context, *, species: str):
    """Remove the examples you taught for a species (original training data is never touched)."""
    found = _find_species(species)
    if found is None:
        await ctx.send(f"❓ I don't know a species called **{species}**.")
        return
    removed = await asyncio.to_thread(_delete_learned, found, False)
    if not removed:
        await ctx.send(f"ℹ️ No taught examples for **{_display(found)}** (original training data is never deleted).")
        return
    await ctx.send(f"🗑️ Removed {len(removed)} taught example(s) of **{_display(found)}**.")


@bot.command(name="undo")
@commands.is_owner()
async def undo_cmd(ctx: commands.Context):
    """Remove the most recent example that was taught (via s!learn or auto-learn)."""
    removed = await asyncio.to_thread(_delete_learned, None, True)
    if not removed:
        await ctx.send("ℹ️ Nothing to undo — no taught examples found.")
        return
    await ctx.send(f"↩️ Removed the latest taught example (**{_display(removed[0])}**).")


@bot.command(name="api")
@commands.is_owner()
async def api_cmd(ctx: commands.Context):
    """Live performance dashboard: activity, active incense, inference/response times, cache hit rate."""
    await ctx.send(embed=_build_api_embed())


_torch_reference = None


def _compare_sync(image_bytes: bytes):
    """Embed one image with the active backend AND the PyTorch reference, and match both against the bank."""
    global _torch_reference
    extractor, species_list, matrix = _snapshot()
    if _torch_reference is None:
        _torch_reference = extractor if not isinstance(extractor, OnnxExtractor) else load_extractor(BOT_MODEL_PATH)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    v_ref, v_act = _torch_reference.extract(img), extractor.extract(img)
    cos = float(np.dot(v_ref, v_act) / max(1e-9, float(np.linalg.norm(v_ref) * np.linalg.norm(v_act))))
    return _predict(v_ref, species_list, matrix), _predict(v_act, species_list, matrix), cos


@bot.command(name="compare")
@commands.is_owner()
async def compare_cmd(ctx: commands.Context):
    """Accuracy check: run the active backend and PyTorch on one image (attach it, or reply to it)."""
    image_bytes, error = await _resolve_message_image(
        ctx, no_image_error="⚠️ Attach an image, or reply to a message that has one, when using this command."
    )
    if image_bytes is None:
        await ctx.send(error)
        return
    async with ctx.typing():
        (w_ref, s_ref, n_ref), (w_act, s_act, n_act), cos = await _in_executor(_compare_sync, image_bytes)
    same = w_ref == w_act
    lines = [
        f"**Active backend ({_backend_name}):** {_display(w_act)} — {s_act * 100:.2f}%",
        f"**PyTorch reference:** {_display(w_ref)} — {s_ref * 100:.2f}%",
        f"**Embedding cosine (1.0 = identical):** {cos:.7f}",
        f"**Same answer:** {'✅ yes' if same else '❌ NO'}",
        "", "**Top matches (active):** " + ", ".join(f"{_display(sp)} {sim * 100:.1f}%" for sp, sim in n_act[:3]),
    ]
    await ctx.send(embed=discord.Embed(
        title="🔬 Backend comparison", description="\n".join(lines),
        color=discord.Color.green() if same else discord.Color.red(),
    ))


@bot.command(name="stats")
@commands.is_owner()
async def stats_cmd(ctx: commands.Context):
    _, species_list, matrix = _snapshot()
    learned = await asyncio.to_thread(_count_learned)
    await ctx.send(
        f"📊 Loaded **{matrix.shape[0]}** feature vectors across "
        f"**{len(set(species_list))}** species from the DB "
        f"(**{learned}** taught by you).\n"
        f"Cache: `{len(_result_cache)}` results · hits `{_stats['hits']}` · inferences `{_stats['misses']}` · "
        f"shared `{_stats['coalesced']}` · dropped `{_stats['dropped']}`\n"
        f"Backend: `{_backend_name}` · workers: `{INFER_WORKERS}` · threads: `{TORCH_THREADS}` · "
        f"avg batch: `{_stats['batch_items'] / max(1, _stats['batches']):.2f}` (max `{BATCH_MAX}`)\n"
        f"Model: `{BOT_MODEL_PATH}`\n"
        f"Confidence threshold: `{CONFIDENCE_THRESHOLD}` · Auto-learn: `{'on (' + AUTO_LEARN_SOURCE + ')' if AUTO_LEARN else 'off'}`"
    )


@bot.command(name="reload")
@commands.is_owner()
async def reload_cmd(ctx: commands.Context):
    """Re-load the model + feature bank without restarting the bot (use after retraining)."""
    await ctx.send("🔄 Reloading model + feature bank ...")
    try:
        await asyncio.to_thread(_load_model_and_bank)
    except Exception as e:
        await ctx.send(f"❌ Reload failed: {e}")
        return
    _, species_list, matrix = _snapshot()
    await ctx.send(f"✅ Reloaded — {matrix.shape[0]} features across {len(set(species_list))} species.")


@bot.command(name="threshold")
@commands.is_owner()
async def threshold_cmd(ctx: commands.Context, value: Optional[float] = None):
    """View or change the confidence threshold at runtime: s!threshold 0.45"""
    global CONFIDENCE_THRESHOLD
    if value is None:
        await ctx.send(f"Current confidence threshold: `{CONFIDENCE_THRESHOLD}`")
        return
    CONFIDENCE_THRESHOLD = value
    await ctx.send(f"✅ Confidence threshold set to `{CONFIDENCE_THRESHOLD}`")


def main():
    _load_model_and_bank()
    _cache_load()
    bot.run(DISCORD_TOKEN, log_handler=None)  # we configured logging above (stdout)


if __name__ == "__main__":
    main()
