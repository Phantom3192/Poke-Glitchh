"""
main.py - Watches for wild Pokemon spawn messages (e.g. from Poketwo)
and replies with the species name predicted by the AI_Model API.

How identification works: this bot is now a PURE API CLIENT. It sends the
spawn image (and image-only requests like s!predict / s!learn) as an HTTP
call to your AI_Model server (server.py from the AI_Model repo), which owns
the model, ONNX Runtime, and the feature bank ("DB"). This bot does not
import torch/onnx/PIL/numpy or touch the database at all - it just uploads
bytes and reads back JSON.

    POST {AI_MODEL_API_URL}/v1/predict        -> single-image identification
    POST {AI_MODEL_API_URL}/v1/predict/batch  -> micro-batched identification
    POST {AI_MODEL_API_URL}/v1/learn          -> teach it a new example
    POST {AI_MODEL_API_URL}/v1/forget         -> remove taught examples
    GET  {AI_MODEL_API_URL}/v1/stats          -> bank size + latency numbers
    GET  {AI_MODEL_API_URL}/health            -> readiness
    POST {AI_MODEL_API_URL}/admin/reload      -> re-read model + feature bank

Setup:
    1. pip install -r requirements.txt
    2. Set DISCORD_TOKEN and AI_MODEL_API_URL (and AI_MODEL_API_KEY if your
       AI_Model server has API_KEY set) as environment variables or in a
       .env file. See .env.example.
    3. Deploy/run the AI_Model server (server.py) somewhere reachable from
       wherever this bot runs, and point AI_MODEL_API_URL at it.
    4. python main.py

Commands (owner-only, prefix configurable via COMMAND_PREFIX, default "s!"):
    s!predict            - attach an image, or reply to a message with one, to
                           test identification without needing a live spawn
    s!learn <species>    - teach the bot: attach an image, or reply to a spawn /
                           image message, e.g.  s!learn pikachu
                           (new species: s!learn --new <n>)
    s!forget <species>   - remove the examples you taught for that species
    s!undo               - remove the most recent example you taught
    s!stats              - show how many species/features the API has loaded
    s!api                - live performance dashboard (activity, active incense,
                           inference/response times, cache hit rate)
    s!reload             - ask the API to re-load its model + feature bank
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
import re
import sys
import json
import time
import pickle
import hashlib
import asyncio
import logging
from collections import OrderedDict, deque
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import discord
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

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

# ---- Config ---------------------------------------------------------------
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise SystemExit("Discord token is not set: set DISCORD_TOKEN (check your .env file).")

# Base URL of your deployed AI_Model server.py, e.g.
# https://my-pokemon-model.up.railway.app  (no trailing slash needed)
AI_MODEL_API_URL = (os.getenv("AI_MODEL_API_URL") or "").rstrip("/")
if not AI_MODEL_API_URL:
    raise SystemExit(
        "AI_MODEL_API_URL is not set (check your .env file). "
        "Point it at your deployed AI_Model server.py, e.g. https://your-model.example.com"
    )
# Only needed if the AI_Model server was started with API_KEY set.
AI_MODEL_API_KEY = os.getenv("AI_MODEL_API_KEY", "")
AI_MODEL_TIMEOUT = float(os.getenv("AI_MODEL_TIMEOUT", "20"))

# Default is Poketwo's real bot ID, since that's the most common spawn
# source. Override via .env if you're scanning a different spawn bot.
SPAWN_BOT_ID = int(os.getenv("SPAWN_BOT_ID", "716390085896962058"))
COMMAND_PREFIX = os.getenv("COMMAND_PREFIX", "s!")

# Cosine similarity score (0-1ish) required before the bot replies to a
# spawn automatically. Also sent to the API as the `threshold` query param,
# so s!threshold changes what the API calls "confident" too.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

# If true, the bot stays silent on spawns it isn't confident about instead
# of replying with a flagged low-confidence guess. Keep this off while
# you're first tuning CONFIDENCE_THRESHOLD so you can see real scores.
SILENT_BELOW_THRESHOLD = os.getenv("SILENT_BELOW_THRESHOLD", "false").lower() == "true"

# ---- Throughput tuning (many channels at once, e.g. 100 incense @ 20s) ----
# Micro-batching: each worker groups up to BATCH_MAX queued images into one
# /v1/predict/batch call, waiting at most BATCH_WAIT_MS for stragglers (0 =
# only group what's already waiting, so it never adds latency).
BATCH_MAX = max(1, int(os.getenv("BATCH_MAX", "4")))
BATCH_WAIT_MS = max(0.0, float(os.getenv("BATCH_WAIT_MS", "25")))
# How many /v1/predict(/batch) calls this bot will have in flight at once.
API_CONCURRENCY = max(1, int(os.getenv("API_CONCURRENCY", "4")))
# Skip a queued spawn if it waited longer than this (seconds) - the answer
# would arrive too late to be useful, and skipping lets the queue catch up.
MAX_SPAWN_AGE = float(os.getenv("MAX_SPAWN_AGE", "15"))
# Max cached image results (LRU). Same image bytes/URL -> no API call at all.
CACHE_SIZE = max(0, int(os.getenv("CACHE_SIZE", "5000")))
# Cache is saved here every CACHE_SAVE_INTERVAL seconds and re-loaded on start
# (only if the API's feature bank is unchanged since), so restarts don't need
# a warm-up. Set CACHE_FILE="" to disable.
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

# ---- Auto-learning (opt-in) ------------------------------------------------
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
# "Timburr [emoji]: 99.86%" / "Timburr: 99.86%\nBest name: Timburr".
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

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix=COMMAND_PREFIX, intents=intents)

# ---- HTTP session + AI_Model API client ------------------------------------
_session: Optional[aiohttp.ClientSession] = None
_api_sem: Optional[asyncio.Semaphore] = None


class ApiError(Exception):
    """Raised for any non-2xx response from the AI_Model API."""
    def __init__(self, status: int, detail: Any):
        self.status = status
        self.detail = detail
        text = detail if isinstance(detail, str) else json.dumps(detail)
        super().__init__(f"HTTP {status}: {text}")


async def _get_session() -> aiohttp.ClientSession:
    global _session, _api_sem
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    if _api_sem is None:
        _api_sem = asyncio.Semaphore(API_CONCURRENCY)
    return _session


async def _api_request(method: str, path: str, **kwargs) -> dict:
    sess = await _get_session()
    headers = kwargs.pop("headers", {}) or {}
    if AI_MODEL_API_KEY:
        headers["X-API-Key"] = AI_MODEL_API_KEY
    url = f"{AI_MODEL_API_URL}{path}"
    timeout = aiohttp.ClientTimeout(total=AI_MODEL_TIMEOUT)
    async with _api_sem:
        async with sess.request(method, url, headers=headers, timeout=timeout, **kwargs) as resp:
            ctype = resp.content_type or ""
            if "json" in ctype:
                body = await resp.json()
            else:
                body = {"raw": (await resp.text())[:500]}
            if resp.status >= 400:
                detail = body.get("detail", body) if isinstance(body, dict) else body
                raise ApiError(resp.status, detail)
            return body


def _file_field(name: str, image_bytes: bytes):
    form = aiohttp.FormData()
    form.add_field(name, image_bytes, filename="image.jpg", content_type="application/octet-stream")
    return form


async def api_health() -> dict:
    return await _api_request("GET", "/health")


async def api_stats() -> dict:
    return await _api_request("GET", "/v1/stats")


async def api_predict(image_bytes: bytes, *, threshold: Optional[float] = None,
                       include_embedding: bool = False) -> dict:
    form = _file_field("file", image_bytes)
    params = {}
    if threshold is not None:
        params["threshold"] = str(threshold)
    if include_embedding:
        params["include_embedding"] = "true"
    return await _api_request("POST", "/v1/predict", data=form, params=params)


async def api_predict_batch(images_bytes: List[bytes], *, threshold: Optional[float] = None,
                             include_embedding: bool = False) -> List[Optional[dict]]:
    form = aiohttp.FormData()
    for i, b in enumerate(images_bytes):
        form.add_field("files", b, filename=f"image{i}.jpg", content_type="application/octet-stream")
    params = {}
    if threshold is not None:
        params["threshold"] = str(threshold)
    if include_embedding:
        params["include_embedding"] = "true"
    body = await _api_request("POST", "/v1/predict/batch", data=form, params=params)
    out = []
    for r in body["results"]:
        out.append(r if r.get("ok") else None)
    return out


async def api_learn(species: str, *, allow_new: bool = False,
                     file_bytes: Optional[bytes] = None,
                     embedding: Optional[List[float]] = None) -> dict:
    form = aiohttp.FormData()
    form.add_field("species", species)
    form.add_field("allow_new", "true" if allow_new else "false")
    if file_bytes is not None:
        form.add_field("file", file_bytes, filename="image.jpg", content_type="application/octet-stream")
    if embedding is not None:
        form.add_field("embedding", json.dumps(list(embedding)))
    return await _api_request("POST", "/v1/learn", data=form)


async def api_forget(*, species: Optional[str] = None, last: bool = False) -> dict:
    payload: dict = {}
    if species:
        payload["species"] = species
    if last:
        payload["last"] = True
    return await _api_request("POST", "/v1/forget", json=payload)


async def api_reload() -> dict:
    return await _api_request("POST", "/admin/reload")


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


# ---- Bank-version tracking (drives cache invalidation) ---------------------
# Every /v1/predict(/batch), /v1/learn and /v1/forget response carries the
# API's current bank_version. Whenever it changes (because we taught/forgot
# something, or someone else did via the API) the cache is stale and gets
# dropped - there's no local feature matrix to fingerprint anymore.
_known_bank_version: Optional[int] = None


async def _note_bank_version(version: Optional[int]):
    global _known_bank_version
    if version is None:
        return
    if _known_bank_version is not None and version != _known_bank_version:
        await _cache_clear()
    _known_bank_version = version


# ---- Result cache (by image hash) ------------------------------------------
_cache_lock = asyncio.Lock()
_result_cache: "OrderedDict[str, tuple]" = OrderedDict()   # sha1(image bytes) -> result tuple
_url_cache: "OrderedDict[str, str]" = OrderedDict()        # image url -> sha1
_cache_dirty = False
_stats = {"hits": 0, "misses": 0, "coalesced": 0, "dropped": 0, "batches": 0, "batch_items": 0}


async def _cache_clear():
    global _cache_dirty
    async with _cache_lock:
        _cache_dirty = False
        _result_cache.clear()
        _url_cache.clear()


async def _cache_get(digest: str):
    async with _cache_lock:
        res = _result_cache.get(digest)
        if res is not None:
            _result_cache.move_to_end(digest)
        return res


async def _cache_get_by_url(url: str):
    async with _cache_lock:
        digest = _url_cache.get(url)
        if digest is None:
            return None
        res = _result_cache.get(digest)
        if res is not None:
            _result_cache.move_to_end(digest)
            _url_cache.move_to_end(url)
        return res


async def _cache_put(digest: str, result: tuple, url: Optional[str] = None):
    global _cache_dirty
    if CACHE_SIZE <= 0:
        return
    async with _cache_lock:
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


def _cache_save():
    global _cache_dirty
    if not CACHE_FILE or CACHE_SIZE <= 0 or not _cache_dirty:
        return
    payload = {"fp": _known_bank_version, "results": list(_result_cache.items()), "urls": list(_url_cache.items())}
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
        if payload.get("fp") != _known_bank_version:
            log.info("Saved spawn cache doesn't match the API's current feature bank - ignoring it")
            return
        for key, val in payload["results"][-CACHE_SIZE:]:
            _result_cache[key] = val
        for url, digest in payload["urls"][-CACHE_SIZE:]:
            if digest in _result_cache:
                _url_cache[url] = digest
        log.info(f"Loaded {len(_result_cache)} cached spawn results from {CACHE_FILE}")
    except Exception as e:
        log.warning(f"Couldn't load spawn cache ({type(e).__name__}: {e}) - starting empty")


async def _cache_saver():
    while True:
        await asyncio.sleep(CACHE_SAVE_INTERVAL)
        try:
            await asyncio.to_thread(_cache_save)
        except Exception as e:
            log.warning(f"Couldn't save spawn cache ({type(e).__name__}: {e})")


# ---- s!api metrics (rolling window) ----------------------------------------
_m_inference: deque = deque()   # (ts, ms) API embed time, one entry per real inference
_m_store: deque = deque()       # (ts, ms) API feature-bank match time
_m_response: deque = deque()    # (ts, ms) spawn seen -> reply sent
_m_spawns: deque = deque()      # (ts, 1)  every spawn seen
_channel_last_spawn: Dict[int, float] = {}
_named_total = 0


def _m_add(dq: deque, value):
    now = time.time()
    dq.append((now, value))
    cutoff = now - METRICS_WINDOW
    while dq and dq[0][0] < cutoff:
        dq.popleft()


def _m_values(dq: deque, since: Optional[float] = None):
    cutoff = time.time() - (METRICS_WINDOW if since is None else since)
    return [v for ts, v in dq if ts >= cutoff]


def _fmt_ms(v: Optional[float]) -> str:
    if v is None:
        return "-"
    return f"{v / 1000:.3f}s" if v >= 1000 else f"{v:.0f} ms"


def _activity(rate: float):
    """rate = spawns/sec over the last minute -> (label, embed colour name)."""
    if rate <= 0:
        return "Idle", "light_grey"
    if rate < 2:
        return "Low Activity", "green"
    if rate < 6:
        return "Moderate Activity", "gold"
    if rate < 15:
        return "High Activity", "orange"
    return "Very High Activity", "red"


async def _api_snapshot() -> dict:
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
    try:
        health = await api_health()
        bank_size = health.get("vectors", 0)
    except Exception:
        bank_size = 0
    return {
        "activity": label, "color": color, "rate": rate, "active": active,
        "hit_rate": None if spawns == 0 else max(0.0, 1.0 - len(inf) / spawns),
        "store": stats(store), "inference": stats(inf), "response": stats(resp),
        "bank": int(bank_size), "named": _named_total,
    }


async def _build_api_embed() -> discord.Embed:
    d = await _api_snapshot()
    embed = discord.Embed(title="API Status", color=getattr(discord.Color, d["color"])())

    def field(name, value):
        embed.add_field(name=name, value=value, inline=True)

    s_min, s_max, s_avg, s_last = d["store"]
    i_min, i_max, i_avg, i_last = d["inference"]
    r_min, r_max, r_avg, r_last = d["response"]
    field("Activity", d["activity"])
    field("Active Incense", f"`{d['active']}`")
    field("Cache Hit Rate", "-" if d["hit_rate"] is None else f"{d['hit_rate'] * 100:.0f}%")
    field("Store (latest)", _fmt_ms(s_last))
    field("Store (avg)", _fmt_ms(s_avg))
    field("Bank Size", f"{d['bank']:,}")
    field("Embed Fastest", _fmt_ms(i_min))
    field("Embed Slowest", _fmt_ms(i_max))
    field("Embed Avg", _fmt_ms(i_avg))
    field("Embed Latest", _fmt_ms(i_last))
    field("Response Fastest", _fmt_ms(r_min))
    field("Response Slowest", _fmt_ms(r_max))
    field("Response Avg", _fmt_ms(r_avg))
    field("Response Latest", _fmt_ms(r_last))
    field("Pokemons Named", f"{d['named']:,}")
    embed.set_footer(
        text=f"API: {AI_MODEL_API_URL} | concurrency={API_CONCURRENCY}, batch<={BATCH_MAX} | "
             f"Window: {METRICS_WINDOW / 60:g} min rolling"
    )
    return embed


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


# ---- Identification (via API) ----------------------------------------------
# A "result" tuple throughout this file is (winner, score, neighbors, embedding),
# matching the shape the old local-inference code used, so the rest of the bot
# (auto-learn, caching, display) didn't need to change shape.

def _result_from_predict(data: dict) -> tuple:
    neighbors = [(n["species"], n["score"]) for n in data.get("neighbors", [])]
    return data["species"], data["score"], neighbors, data.get("embedding")


async def _identify_one(image_bytes: bytes) -> Optional[tuple]:
    try:
        data = await api_predict(image_bytes, threshold=CONFIDENCE_THRESHOLD, include_embedding=True)
    except ApiError as e:
        if e.status != 422:  # 422 = unreadable image, not worth logging as an error
            log.error(f"Predict API error: {e}")
        return None
    except Exception as e:
        log.error(f"Predict API unreachable ({type(e).__name__}): {e}")
        return None
    await _note_bank_version(data.get("bank_version"))
    _m_add(_m_inference, data.get("timing_ms", {}).get("embed", 0.0))
    _m_add(_m_store, data.get("timing_ms", {}).get("match", 0.0))
    return _result_from_predict(data)


async def _identify_batch(images_bytes: List[bytes]) -> List[Optional[tuple]]:
    try:
        results = await api_predict_batch(images_bytes, threshold=CONFIDENCE_THRESHOLD, include_embedding=True)
    except Exception as e:
        log.error(f"Batch predict API error ({type(e).__name__}): {e}")
        return [None] * len(images_bytes)
    out = []
    for r in results:
        if r is None:
            out.append(None)
            continue
        await _note_bank_version(r.get("bank_version"))
        _m_add(_m_inference, r.get("timing_ms", {}).get("embed", 0.0))
        _m_add(_m_store, r.get("timing_ms", {}).get("match", 0.0))
        out.append(_result_from_predict(r))
    return out


_batch_queue: Optional[asyncio.Queue] = None
_batch_tasks = []


def _entry_latest(entry: dict) -> float:
    return entry["latest"] if entry else time.time()


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
            results = await _identify_batch([b for b, _ in live])
        except Exception as e:
            log.error(f"Batch inference failed: {type(e).__name__}: {e}")
            results = [None] * len(live)
        _stats["batches"] += 1
        _stats["batch_items"] += len(live)
        for (_, fut), res in zip(live, results):
            if not fut.done():
                fut.set_result(res)


async def _submit_identification(image_bytes: bytes, entry: dict):
    """Queue an image for the (micro-batching) API workers; resolves to a result tuple or None."""
    global _batch_queue
    if _batch_queue is None:
        _batch_queue = asyncio.Queue()
        _batch_tasks.extend(asyncio.create_task(_batch_worker()) for _ in range(API_CONCURRENCY))
    fut = asyncio.get_running_loop().create_future()
    _batch_queue.put_nowait((image_bytes, entry, fut))
    return await fut


_inflight_url: Dict[str, dict] = {}
_inflight_digest: Dict[str, dict] = {}


async def _coalesced(table: Dict[str, dict], key: str, received: float, work, deps=None):
    """Run work(entry) once per key at a time; concurrent callers with the same key share its result."""
    entry = table.get(key)
    if entry is None:
        entry = {"latest": received, "fut": None, "deps": deps or []}
        table[key] = entry
    else:
        entry["latest"] = max(entry["latest"], received)
        entry["deps"] = (entry.get("deps") or []) + (deps or [])
        _stats["coalesced"] += 1

    if entry["fut"] is None:
        entry["fut"] = asyncio.get_running_loop().create_future()
        try:
            result = await work(entry)
            if not entry["fut"].done():
                entry["fut"].set_result(result)
        except Exception as e:
            if not entry["fut"].done():
                entry["fut"].set_exception(e)
            raise
        finally:
            if table.get(key) is entry:
                del table[key]
    return await entry["fut"]


async def _identify_spawn(url: str, received: float):
    """Returns (winner, score, neighbors, embedding) or None. Cache -> coalesce -> download -> hash -> infer."""
    res = await _cache_get_by_url(url) if URL_CACHE else None
    if res is not None:
        _stats["hits"] += 1
        return res
    cache_url = url if URL_CACHE else None

    async def work(url_entry):
        image_bytes = await _download(url)
        if not image_bytes:
            return None
        digest = hashlib.sha1(image_bytes).hexdigest()
        res = await _cache_get(digest)
        if res is not None:
            _stats["hits"] += 1
            await _cache_put(digest, res, cache_url)
            return res

        async def infer(digest_entry):
            out = await _submit_identification(image_bytes, digest_entry)
            if out is not None:
                _stats["misses"] += 1
            return out

        res = await _coalesced(_inflight_digest, digest, received, infer, deps=[url_entry])
        if res is not None:
            await _cache_put(digest, res, cache_url)
        return res

    return await _coalesced(_inflight_url, url, received, work)


# ---- Learning (via API) -----------------------------------------------------

def _key(name: str) -> str:
    """Loose comparison key so 'Mr. Mime', 'mr_mime' and 'mr._mime' all match."""
    return re.sub(r"[^a-z0-9\u2640\u2642]+", "", name.lower())


def _display(species: str) -> str:
    return species.replace("_", " ").title()


async def _learn(species: str, *, allow_new: bool = False,
                  file_bytes: Optional[bytes] = None, embedding: Optional[List[float]] = None) -> dict:
    data = await api_learn(species, allow_new=allow_new, file_bytes=file_bytes, embedding=embedding)
    await _note_bank_version(data.get("bank_version"))
    return data


async def _forget(*, species: Optional[str] = None, last: bool = False) -> dict:
    data = await api_forget(species=species, last=last)
    await _note_bank_version(data.get("bank_version"))
    return data


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
        log.info(f"Auto-learn: '{caught}' was caught but I have no analysed spawn for this channel - skipping")
        return
    spawned_at, guess, score, vec = entry
    if time.time() - spawned_at > AUTO_LEARN_MAX_AGE:
        log.info(f"Auto-learn: last spawn is too old to trust - skipping '{caught}'")
        return
    _last_spawn[message.channel.id] = None  # learn from each spawn at most once

    if score >= CONFIDENCE_THRESHOLD and _key(caught) == _key(guess):
        log.info(f"Auto-learn: guessed {guess} correctly ({score:.3f}) - nothing to learn")
        return
    if vec is None:
        log.warning("Auto-learn: no embedding available for the last spawn - skipping")
        return

    try:
        data = await _learn(caught, allow_new=False, embedding=vec)
    except ApiError as e:
        log.warning(f"Auto-learn: API rejected '{caught}': {e}")
        return
    status = data.get("status")
    verdict = "wrong" if _key(caught) != _key(guess) else "low confidence"
    if status == "unknown":
        suggestions = data.get("suggestions") or []
        log.warning(f"Auto-learn: '{caught}' isn't a species the API knows - "
                    f"use s!learn --new {caught.lower()} if it's real"
                    + (f" (did you mean: {', '.join(suggestions)}?)" if suggestions else ""))
        return
    examples = data.get("examples")
    log.info(f"Auto-learn: caught={caught}, I guessed {guess} ({score:.3f}, {verdict}) -> {status}"
             + (f" ({examples} examples now)" if status == "learned" else ""))


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
        log.info(f"Auto-learn (source bot): '{claimed}' only {source_conf:.1%} confident - not trusted enough")
        return

    entry = _last_spawn.get(message.channel.id)
    if not entry:
        log.info(f"Auto-learn (source bot): '{claimed}' named but I have no analysed spawn for this channel - skipping")
        return
    spawned_at, guess, score, vec = entry
    if time.time() - spawned_at > AUTO_LEARN_MAX_AGE:
        log.info(f"Auto-learn (source bot): last spawn is too old to trust - skipping '{claimed}'")
        return
    _last_spawn[message.channel.id] = None  # learn from each spawn at most once

    if score >= CONFIDENCE_THRESHOLD and _key(claimed) == _key(guess):
        log.info(f"Auto-learn (source bot): guessed {guess} correctly ({score:.3f}) - nothing to learn")
        return
    if vec is None:
        log.warning("Auto-learn (source bot): no embedding available for the last spawn - skipping")
        return

    try:
        data = await _learn(claimed, allow_new=False, embedding=vec)
    except ApiError as e:
        log.warning(f"Auto-learn (source bot): API rejected '{claimed}': {e}")
        return
    status = data.get("status")
    verdict = "wrong" if _key(claimed) != _key(guess) else "low confidence"
    if status == "unknown":
        suggestions = data.get("suggestions") or []
        log.warning(f"Auto-learn (source bot): '{claimed}' isn't a species the API knows - "
                    f"use s!learn --new {claimed.lower()} if it's real"
                    + (f" (did you mean: {', '.join(suggestions)}?)" if suggestions else ""))
        return
    examples = data.get("examples")
    log.info(f"Auto-learn (source bot): {message.author} said {claimed} "
             f"({source_conf if source_conf is not None else '?'}), "
             f"I guessed {guess} ({score:.3f}, {verdict}) -> {status}"
             + (f" ({examples} examples now)" if status == "learned" else ""))


# ---- Discord events ---------------------------------------------------------

_last_spawn: Dict[int, Optional[Tuple[float, str, float, Optional[List[float]]]]] = {}
_spawn_gen: Dict[int, int] = {}
_saver_started = False


@bot.event
async def on_ready():
    global _saver_started, _known_bank_version
    await _get_session()

    try:
        health = await api_health()
        _known_bank_version = health.get("bank_version")
        log.info(f"Connected to AI_Model API: {AI_MODEL_API_URL}")
        log.info(f"   ready={health.get('ready')} backend={health.get('backend')} "
                 f"vectors={health.get('vectors')} species={health.get('species')}")
    except Exception as e:
        log.warning(f"Could not reach AI_Model API at startup ({type(e).__name__}: {e}) - "
                    f"will keep retrying on each spawn")

    await asyncio.to_thread(_cache_load)
    if CACHE_FILE and CACHE_SIZE > 0 and not _saver_started:
        _saver_started = True
        asyncio.create_task(_cache_saver())

    log.info("=" * 60)
    log.info(f"Logged in as {bot.user} - scanning for spawns from bot ID {SPAWN_BOT_ID}")
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

    winner, score, neighbors, vec = result

    _last_spawn[message.channel.id] = (time.time(), winner, score, vec)

    elapsed_ms = (time.time() - received) * 1000
    confident = score >= CONFIDENCE_THRESHOLD
    log.info(f"Spawn {message.id}: {winner} (score {score:.3f}, confident={confident}, {elapsed_ms:.0f}ms)")

    if not confident and SILENT_BELOW_THRESHOLD:
        return

    display_name = _display(winner)
    if confident:
        reply_text = f"{display_name} - {score * 100:.1f}%"
    else:
        reply_text = f"{display_name}? - {score * 100:.1f}% (low confidence, might be wrong)"

    try:
        await message.reply(reply_text, allowed_mentions=discord.AllowedMentions.none())
    except discord.HTTPException as e:
        log.warning(f"Failed to reply to spawn message {message.id}: {e}")
        return
    _named_total += 1
    _m_add(_m_response, (time.time() - received) * 1000)


# ---- Manual testing / admin commands ----------------------------------------

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
        ctx, no_image_error="Attach an image, or reply to a message that has one, when using this command."
    )
    if image_bytes is None:
        await ctx.send(error)
        return

    start = time.time()
    async with ctx.typing():
        result = await _identify_one(image_bytes)
    elapsed_ms = (time.time() - start) * 1000

    if result is None:
        await ctx.send("Couldn't get a prediction (unreadable image, or the AI_Model API is unreachable - check logs).")
        return
    winner, score, neighbors, _ = result

    lines = [f"**Best guess:** {_display(winner)} - {score * 100:.1f}%"]
    lines.append(f"Processed in {elapsed_ms:.0f}ms\n")
    lines.append("**Top matches:**")
    for sp, sim in neighbors:
        lines.append(f"  {sp:<20s} {sim * 100:.1f}%")

    embed = discord.Embed(
        title="Predict Result",
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
        ctx, no_image_error="Attach an image, or reply to a spawn/image message, when using s!learn."
    )
    if image_bytes is None:
        await ctx.send(error)
        return

    try:
        async with ctx.typing():
            data = await _learn(species, allow_new=allow_new, file_bytes=image_bytes)
    except ApiError as e:
        await ctx.send(f"Couldn't save that example: {e}")
        return
    except Exception as e:
        log.error(f"s!learn failed: {type(e).__name__}: {e}")
        await ctx.send(f"Couldn't reach the AI_Model API: {type(e).__name__}: {e}")
        return

    status, name, extra = data.get("status"), data.get("species"), data.get("examples")
    if status == "bad_image":
        await ctx.send("Couldn't extract features from that image, so nothing was saved.")
    elif status == "unknown":
        suggestions = data.get("suggestions") or []
        hint = f" Did you mean: {', '.join(_display(s) for s in suggestions)}?" if suggestions else ""
        await ctx.send(
            f"I don't know a species called **{species}**.{hint}\n"
            f"If it really is a new species, use `{COMMAND_PREFIX}learn --new {species}`."
        )
    elif status == "duplicate":
        await ctx.send(f"I already have this exact image saved for **{_display(name)}** - nothing to add.")
    else:
        result = await _identify_one(image_bytes)
        if result is not None:
            winner, score, _n, _v = result
            log.info(f"Learned 1 example of {name} ({extra} now)")
            await ctx.send(
                f"Learned **{_display(name)}** - now {extra} example(s) for it.\n"
                f"This image now identifies as **{_display(winner)}** ({score * 100:.1f}%)."
            )
        else:
            await ctx.send(f"Learned **{_display(name)}** - now {extra} example(s) for it.")


@bot.command(name="forget")
@commands.is_owner()
async def forget_cmd(ctx: commands.Context, *, species: str):
    """Remove the examples you taught for a species (original training data is never touched)."""
    try:
        data = await _forget(species=species)
    except ApiError as e:
        if e.status == 404:
            suggestions = e.detail.get("suggestions") if isinstance(e.detail, dict) else None
            hint = f" Did you mean: {', '.join(_display(s) for s in suggestions)}?" if suggestions else ""
            await ctx.send(f"I don't know a species called **{species}**.{hint}")
        else:
            await ctx.send(f"{e}")
        return
    removed = data.get("species") or []
    if not removed:
        await ctx.send(f"No taught examples for **{_display(species)}** (original training data is never deleted).")
        return
    await ctx.send(f"Removed {data.get('removed', len(removed))} taught example(s) of **{_display(removed[0])}**.")


@bot.command(name="undo")
@commands.is_owner()
async def undo_cmd(ctx: commands.Context):
    """Remove the most recent example that was taught (via s!learn or auto-learn)."""
    try:
        data = await _forget(last=True)
    except ApiError as e:
        await ctx.send(f"{e}")
        return
    removed = data.get("species") or []
    if not removed:
        await ctx.send("Nothing to undo - no taught examples found.")
        return
    await ctx.send(f"Removed the latest taught example (**{_display(removed[0])}**).")


@bot.command(name="api")
@commands.is_owner()
async def api_cmd(ctx: commands.Context):
    """Live performance dashboard: activity, active incense, inference/response times, cache hit rate."""
    await ctx.send(embed=await _build_api_embed())


@bot.command(name="stats")
@commands.is_owner()
async def stats_cmd(ctx: commands.Context):
    try:
        data = await api_stats()
    except ApiError as e:
        await ctx.send(f"Couldn't fetch stats from the API: {e}")
        return
    except Exception as e:
        await ctx.send(f"Couldn't reach the AI_Model API: {type(e).__name__}: {e}")
        return
    learned = data.get("learned_vectors", "?")
    settings = data.get("settings", {})
    counters = data.get("counters", {})
    await ctx.send(
        f"API reports **{data.get('vectors', '?')}** feature vectors across "
        f"**{data.get('species', '?')}** species (**{learned}** taught).\n"
        f"Bot-side cache: `{len(_result_cache)}` results - hits `{_stats['hits']}` - "
        f"API calls `{_stats['misses']}` - shared `{_stats['coalesced']}` - dropped `{_stats['dropped']}`\n"
        f"Backend: `{data.get('backend', '?')}` - model: `{data.get('model', '?')}` - "
        f"avg batch: `{_stats['batch_items'] / max(1, _stats['batches']):.2f}` (max `{BATCH_MAX}`)\n"
        f"API: `{AI_MODEL_API_URL}` - predictions: `{counters.get('predictions', '?')}`\n"
        f"Confidence threshold: `{CONFIDENCE_THRESHOLD}` (server default `{settings.get('confidence_threshold', '?')}`) - "
        f"Auto-learn: `{'on (' + AUTO_LEARN_SOURCE + ')' if AUTO_LEARN else 'off'}`"
    )


@bot.command(name="reload")
@commands.is_owner()
async def reload_cmd(ctx: commands.Context):
    """Ask the AI_Model API to re-load its model + feature bank (use after retraining)."""
    global _known_bank_version
    await ctx.send("Asking the API to reload its model + feature bank ...")
    try:
        result = await api_reload()
    except ApiError as e:
        await ctx.send(f"Reload failed: {e}")
        return
    except Exception as e:
        await ctx.send(f"Couldn't reach the AI_Model API: {type(e).__name__}: {e}")
        return
    await _cache_clear()
    try:
        health = await api_health()
        _known_bank_version = health.get("bank_version")
    except Exception:
        pass
    await ctx.send(
        f"Reloaded - {result.get('vectors', '?')} features across {result.get('species', '?')} species."
        + (f"\nNote: {result['note']}" if result.get("note") else "")
    )


@bot.command(name="threshold")
@commands.is_owner()
async def threshold_cmd(ctx: commands.Context, value: Optional[float] = None):
    """View or change the confidence threshold at runtime: s!threshold 0.45"""
    global CONFIDENCE_THRESHOLD
    if value is None:
        await ctx.send(f"Current confidence threshold: `{CONFIDENCE_THRESHOLD}`")
        return
    CONFIDENCE_THRESHOLD = value
    await ctx.send(f"Confidence threshold set to `{CONFIDENCE_THRESHOLD}` (sent to the API on every prediction).")


def main():
    bot.run(DISCORD_TOKEN, log_handler=None)  # we configured logging above (stdout)


if __name__ == "__main__":
    main()
