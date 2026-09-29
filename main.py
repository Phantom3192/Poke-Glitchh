"""
main.py - Watches for wild Pokemon spawn messages (e.g. from Poketwo)
and replies with the species name predicted by the AI_Model API.

How identification works: this bot is a PURE API CLIENT. It sends the
spawn image (and image-only requests like s!predict / s!learn) as an HTTP
call to your AI_Model server (server.py from the AI_Model repo), which owns
the model, ONNX Runtime, and the feature bank ("DB"). This bot does not
import torch/onnx/PIL/numpy or touch the database at all - it just uploads
bytes and reads back JSON.

    POST {AI_MODEL_API_URL}/v1/predict        -> single-image identification
    POST {AI_MODEL_API_URL}/v1/predict/batch  -> micro-batched identification
    POST {AI_MODEL_API_URL}/v1/learn          -> teach it a new example
    POST {AI_MODEL_API_URL}/v1/learn/batch    -> teach many examples in one call
    POST {AI_MODEL_API_URL}/v1/forget         -> remove taught examples
    GET  {AI_MODEL_API_URL}/v1/stats          -> bank size + latency numbers
    GET  {AI_MODEL_API_URL}/v1/species        -> per-species taught-image counts
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
    s!checklist [search] - browse per-species image counts ("Charizard - 1000
                           images"), searchable, sortable high/low via buttons
    s!stats              - show how many species/features the API has loaded
    s!api                - live performance dashboard
    s!reload             - ask the API to re-load its model + feature bank
    s!threshold X        - view/change the confidence threshold at runtime
    s!naming [on|off]    - turn spawn identification off/on for this server
    s!backfill [scope] [max_per_channel] [dry] [fresh] [from:<channel_id>]
                         - scan old spawn history and self-label it for free.
                           Runs BACKFILL_CHANNEL_CONCURRENCY channels in parallel,
                           batches /v1/learn calls BACKFILL_BATCH_SIZE at a time,
                           and downloads images BACKFILL_DOWNLOAD_CONCURRENCY at
                           a time - so scan speed is limited only by Discord's
                           history endpoint, not by the model server.
                           s!backfill stop  - pause a running backfill (progress
                                              is checkpointed; re-run the same
                                              command to resume).
    s!bulklearn [dry] [known-only]
                         - attach a .zip/.rar/.7z (or reply to a message that
                           has one), laid out one folder per species just like
                           AI_Model's "Extra pokemons/" convention:
                               my_pokemons.zip
                                 pikachu/img1.jpg
                                 charizard/img1.jpg ...
                           Extracts it, then teaches every image via
                           /v1/learn/batch - purely additive (appends to
                           bank/learned.jsonl on the server, never touches
                           base_features.npy/base_species.npy, no retrain
                           needed). "dry" previews species/counts without
                           teaching anything. "known-only" skips folders whose
                           name doesn't already match a known species instead
                           of auto-creating them (protects against typos).

Optional auto-learning (AUTO_LEARN=true): learns from the last spawn in a
channel if the bot guessed it wrong or wasn't confident. Where the "correct"
answer comes from is set by AUTO_LEARN_SOURCE:
    - "catch" (default): the spawn bot's own catch announcement.
    - "bot": trust another bot's identification message instead.
See .env.example for all auto-learn variables.
"""

import io
import os
import math
import re
import atexit
import signal
import sys
import json
import time
import pickle
import shutil
import tempfile
import zipfile
import hashlib
import asyncio
import logging
from pathlib import Path
from collections import OrderedDict, deque
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlsplit, parse_qsl, urlencode

import aiohttp
import discord
from discord.ext import commands
from dotenv import load_dotenv

import pokedata
from guild_store import store as guild_store
from turso_db import db as turso

try:
    import psutil  # optional: CPU/RAM numbers for s!ping. Bot works fine without it.
except ImportError:
    psutil = None

_BOT_START_TIME = time.time()

try:
    import orjson  # Faster JSON parser
except ImportError:
    orjson = None  # Fallback to standard json if not installed

try:  # optional: shrinks uploads. Without Pillow the original image is sent (same results, bigger upload)
    from PIL import Image as _PILImage
except ImportError:
    _PILImage = None

load_dotenv()

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"))
_handler.setLevel(logging.ERROR)


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
AI_MODEL_TIMEOUT = None  # no timeout on bot -> API requests (wait as long as the server needs)

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

# TEMPORARY: show a neighbour-agreement "confidence" in replies instead of the raw
# similarity score. Set SHOW_CONFIDENCE=false in .env to go back to similarity.
# Only the number that is *displayed* changes; the CONFIDENCE_THRESHOLD check
# still uses the similarity score.
SHOW_CONFIDENCE = os.getenv("SHOW_CONFIDENCE", "true").lower() == "true"
CONFIDENCE_TEMP = float(os.getenv("CONFIDENCE_TEMP", "0.02"))  # smaller = stricter


def display_confidence(winner: str, score: float, neighbors) -> float:
    """
    Share of the top-k nearest neighbours (weighted by similarity) that belong to
    the winning species. 1.0 = every close neighbour agrees; it drops when another
    species has a similar match. Falls back to the similarity score if unavailable.
    """
    if not SHOW_CONFIDENCE or not neighbors:
        return score
    top = max(sim for _, sim in neighbors)
    total = won = 0.0
    for sp, sim in neighbors:
        w = math.exp((sim - top) / CONFIDENCE_TEMP)
        total += w
        if sp == winner:
            won += w
    return won / total if total > 0 else score

# ---- Throughput tuning (many channels at once, e.g. 100 incense @ 20s) ----
# Micro-batching: each worker groups up to BATCH_MAX queued images into one
# /v1/predict/batch call, waiting at most BATCH_WAIT_MS for stragglers (0 =
# only group what's already waiting, so it never adds latency).
BATCH_MAX = 8
# 0 = only group spawns that are already queued at the exact same instant (no added latency).
# A small non-zero wait lets nearby spawns (e.g. several channels firing within milliseconds of
# each other) share ONE network round trip to the API instead of each paying it separately -
# this is the main lever for cutting total round trips without moving either deployment.
BATCH_WAIT_MS = float(os.getenv("BATCH_WAIT_MS", "30.0"))
# How many /v1/predict(/batch) calls this bot will have in flight at once.
API_CONCURRENCY = 8
# The model's input resolution (must match the server's classifier - see
# INPUT_SIZE in the AI_Model repo's onnx_backend.py). Requesting the spawn
# image at this size from Discord's media proxy means we download a much
# smaller payload and skip a resize the server would otherwise have to do.
SPAWN_IMAGE_SIZE = 224
# Skip a queued spawn if it waited longer than this (seconds) - the answer
# would arrive too late to be useful, and skipping lets the queue catch up.
MAX_SPAWN_AGE = 15.0
# Max cached image results (LRU). Same image bytes/URL -> no API call at all.
CACHE_SIZE = 5000
# Cache is saved here every CACHE_SAVE_INTERVAL seconds and re-loaded on start
# (only if the API's feature bank is unchanged since), so restarts don't need
# a warm-up. Set CACHE_FILE="" to disable.
CACHE_FILE = "spawn_cache.pkl"
# Also skip the download when the same image URL was seen before. Off by default: results are
# always cached by the image's exact bytes (provably identical input -> identical answer), a
# URL match is only an assumption that the URL always serves the same image.
URL_CACHE = True
# Send the image URL to the API and let the SERVER download it (no download + upload through the bot).
# Falls back to the download+upload path per spawn if the server can't fetch that URL.
URL_MODE = os.getenv("URL_MODE", "true").lower() == "true"
CACHE_SAVE_INTERVAL = 15.0
# s!api dashboard: rolling window for timing stats, and how recently a channel
# must have spawned to count as an "active incense" channel.
METRICS_WINDOW = 300.0
ACTIVE_INCENSE_WINDOW = 60.0

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

# -- Spawn reply decoration (type emojis / best name / copyable catch command) --
# What a catch command looks like when we hand it to you copy-pasteable, e.g.
# "c {name}" (Poketwo's real shorthand) or "@Poketwo#8236 c {name}" if you'd
# rather it look like the old bot's output. {name} is filled with the "best name".
CATCH_COMMAND_TEMPLATE = os.getenv("CATCH_COMMAND_TEMPLATE", "c {name}")
# Regex used to guess whether a spawn is shiny, for s!sh pings. Poketwo/clones
# usually put a sparkle or the word "shiny" somewhere in the spawn embed when
# it's shiny - adjust SHINY_REGEX in .env if your spawn bot phrases it differently.
SHINY_REGEX = os.getenv("SHINY_REGEX", r"[\u2728\u2b50]|shiny")
_SHINY_RE = re.compile(SHINY_REGEX, re.IGNORECASE)

# -- s!backfill --
# Poketwo (and most clones) name the PREVIOUS spawn when the NEXT one appears:
# "Wild Glalie fled. A new wild pokemon has appeared!" - so every spawn image
# is ground-truth-labeled by the very next spawn message in that channel.
# This lets s!backfill turn a channel's entire history into training data
# without needing a single catch.
FLED_REGEX = os.getenv("FLED_REGEX", r"[Ww]ild\s+(.+?)\s+fled")
_FLED_RE = re.compile(FLED_REGEX)

# Backfill tuning (hardcoded - no .env needed).
# These are the values that put backfill right at Discord's practical speed
# ceiling without producing visible rate-limit log spam.
BACKFILL_CHANNEL_CONCURRENCY = 6        # channels scanned in parallel
BACKFILL_DOWNLOAD_CONCURRENCY = 16      # concurrent image downloads
BACKFILL_BATCH_SIZE = 32                # pairs per /v1/learn/batch call
BACKFILL_QUEUE_MAX = 512                # bounded queue so a fast reader can't OOM
BACKFILL_CHANNEL_DELAY = 0.25           # seconds between history pages per channel

# Backfill runtime state + checkpoint files.
_backfill_cancel = False                # set by `s!backfill stop`; checked between messages
_backfill_active = False                # only one backfill run at a time
BACKFILL_CHECKPOINT_PATH = os.getenv("BACKFILL_CHECKPOINT_PATH", "backfill_checkpoint.json")
BACKFILL_CHECKPOINT_INTERVAL = float(os.getenv("BACKFILL_CHECKPOINT_INTERVAL", "10"))
_backfill_checkpoint_lock = asyncio.Lock()

# -- s!bulklearn --
# Teach the bot from an uploaded archive of images (one folder per species)
# instead of scanning Discord history. Same /v1/learn/batch endpoint as
# s!backfill, just sourced from a zip/rar/7z instead of channel messages.
BULKLEARN_BATCH_SIZE = 32                # pairs per /v1/learn/batch call
BULKLEARN_WORKERS = 4                    # concurrent batch calls in flight
BULKLEARN_MAX_ARCHIVE_MB = int(os.getenv("BULKLEARN_MAX_ARCHIVE_MB", "500"))
_BULKLEARN_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}
_BULKLEARN_ARCHIVE_EXT = (".zip", ".rar", ".7z")
_bulklearn_active = False

if AUTO_LEARN and AUTO_LEARN_SOURCE == "bot" and not AUTO_LEARN_SOURCE_BOT_ID:
    raise RuntimeError(
        "AUTO_LEARN_SOURCE=bot requires AUTO_LEARN_SOURCE_BOT_ID to be set in your .env "
        "to the Discord user ID of the bot to learn from."
    )
if AUTO_LEARN and AUTO_LEARN_SOURCE not in ("catch", "bot"):
    raise RuntimeError(f"AUTO_LEARN_SOURCE must be 'catch' or 'bot', got {AUTO_LEARN_SOURCE!r}")

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix=COMMAND_PREFIX, intents=intents, help_command=None)

# -- per-guild naming on/off --
DISABLED_GUILDS_FILE = os.getenv("DISABLED_GUILDS_FILE", "disabled_guilds.json")

def _load_disabled_guilds_file() -> set:
    try:
        with open(DISABLED_GUILDS_FILE, "r") as f:
            return {int(x) for x in json.load(f)}
    except FileNotFoundError:
        return set()
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        log.error(f"Couldn't parse {DISABLED_GUILDS_FILE}, starting with nothing disabled: {e}")
        return set()

def _load_disabled_guilds() -> set:
    if not turso.enabled:
        return _load_disabled_guilds_file()
    found = {int(r[0]) for r in turso.query("SELECT guild_id FROM disabled_guilds")}
    if not found and turso.kv_get("migrated:disabled_guilds") is None:
        found = _load_disabled_guilds_file()          # one-time import of the old JSON file
        if found:
            turso.execute([("INSERT OR IGNORE INTO disabled_guilds(guild_id) VALUES(?)", [str(g)])
                           for g in found])
            log.info(f"Imported {len(found)} disabled guild(s) from {DISABLED_GUILDS_FILE} into Turso")
        turso.kv_set("migrated:disabled_guilds", "1")
    return found

def _save_disabled_guilds() -> None:
    if turso.enabled:
        # replace the whole (tiny) set atomically, in order, on the background writer
        turso.submit([("DELETE FROM disabled_guilds", [])] +
                     [("INSERT INTO disabled_guilds(guild_id) VALUES(?)", [str(g)])
                      for g in sorted(DISABLED_GUILDS)])
        return
    try:
        with open(DISABLED_GUILDS_FILE, "w") as f:
            json.dump(sorted(DISABLED_GUILDS), f)
    except OSError as e:
        log.error(f"Failed to save {DISABLED_GUILDS_FILE}: {e}")

DISABLED_GUILDS: set = _load_disabled_guilds()

def _shrink_for_upload(data: bytes) -> bytes:
    """
    Resize to the model's 224x224 input here (same PIL convert("RGB") + BILINEAR resize the server does)
    and send it as lossless PNG. The server's own resize is then a no-op, so embeddings and answers are
    bit-identical, but the upload is several times smaller. Falls back to the original bytes on any problem.
    """
    if _PILImage is None:
        return data
    try:
        im = _PILImage.open(io.BytesIO(data)).convert("RGB").resize((224, 224), _PILImage.Resampling.BILINEAR)
        out = io.BytesIO()
        im.save(out, "PNG", compress_level=1)
        small = out.getvalue()
        return small if len(small) < len(data) else data
    except Exception:
        return data


# ---- s!bulklearn: archive -> (species, image_bytes) pairs -----------------
def _normalize_species_folder(name: str) -> str:
    """Same convention AI_Model's train_model.py uses for 'Extra pokemons/<species>/'
    folders, so anything taught here lines up with species names used in training."""
    return name.replace("_", " ").strip().lower()


def _extract_archive_to(archive_path: Path, dest_dir: Path) -> None:
    """Extract .zip / .rar / .7z into dest_dir. Mirrors AI_Model/train_model.py's
    extractor so the same archive formats work on both the trainer and the bot."""
    ext = archive_path.suffix.lower()
    if ext == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    elif ext == ".rar":
        try:
            import rarfile
            with rarfile.RarFile(archive_path) as rf:
                rf.extractall(dest_dir)
        except Exception:
            import subprocess
            subprocess.run(["unrar", "x", "-y", str(archive_path), str(dest_dir)],
                            capture_output=True, check=False)
    elif ext == ".7z":
        try:
            import py7zr
            with py7zr.SevenZipFile(archive_path, "r") as sz:
                sz.extractall(dest_dir)
        except Exception:
            import subprocess
            subprocess.run(["7z", "x", "-y", str(archive_path), f"-o{dest_dir}"],
                            capture_output=True, check=False)
    else:
        raise ValueError(f"Unsupported archive type: {ext}")


def _find_species_root(extracted_dir: Path) -> Path:
    """
    Zipping a folder often wraps everything in one parent directory, e.g.
    zipping 'Extra pokemons/' produces 'Extra pokemons/pikachu/...' once
    extracted. If extracted_dir has exactly one subdirectory and no loose
    files, descend into it so species folders are found either way.
    """
    entries = [p for p in extracted_dir.iterdir() if not p.name.startswith("__MACOSX")]
    dirs = [p for p in entries if p.is_dir()]
    files = [p for p in entries if p.is_file()]
    if len(dirs) == 1 and not files:
        return dirs[0]
    return extracted_dir


def _iter_species_images(root: Path):
    """Yield (species, image_path) for every image directly inside a species subfolder of root."""
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        species = _normalize_species_folder(folder.name)
        if not species:
            continue
        for img_path in sorted(folder.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in _BULKLEARN_IMAGE_EXT:
                yield species, img_path


def _bulklearn_status_line(done: int, total: int, stats: dict) -> str:
    pct = (done / total * 100) if total else 100
    return (
        f"Teaching... {done}/{total} ({pct:.0f}%) - "
        f"learned {stats['learned']:,}, dup {stats['duplicate']:,}, "
        f"unknown {stats['unknown']:,}, errors {stats['errors']:,}"
    )


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
        _session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ttl_dns_cache=300, keepalive_timeout=60, limit=0)
        )
    if _api_sem is None:
        _api_sem = asyncio.Semaphore(API_CONCURRENCY)
    return _session


async def _api_request(method: str, path: str, **kwargs) -> dict:
    sess = await _get_session()
    headers = kwargs.pop("headers", {}) or {}
    if AI_MODEL_API_KEY:
        headers["X-API-Key"] = AI_MODEL_API_KEY
    url = f"{AI_MODEL_API_URL}{path}"
    timeout = aiohttp.ClientTimeout(total=None, connect=None, sock_connect=None, sock_read=None)
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


async def api_species_counts() -> dict:
    """GET /v1/species -> {"species": {name: image_count, ...}, "total_species": N, "total_vectors": N}"""
    return await _api_request("GET", "/v1/species")


# ---- Full Prediction Cache (1 hour TTL) ----
_prediction_cache: Dict[str, Tuple[dict, float]] = {}  # {image_hash: (full_result, timestamp)}
_embedding_cache: Dict[str, Tuple[List[float], float]] = {}  # {image_hash: (embedding, timestamp)}
PREDICTION_CACHE_TTL = 3600  # 1 hour in seconds
# Embedding cache TTL: 7 days default. If model is updated, clear cache to re-learn.
EMBEDDING_CACHE_TTL = int(os.getenv("EMBEDDING_CACHE_TTL_SECONDS", str(86400 * 7)))


def _get_image_hash(image_bytes: bytes) -> str:
    """Generate SHA1 hash of image bytes for cache lookup."""
    return hashlib.sha1(image_bytes).hexdigest()


def _get_cached_prediction(image_hash: str) -> Optional[dict]:
    """Return cached full prediction result if exists and not expired, else None."""
    if image_hash in _prediction_cache:
        result, timestamp = _prediction_cache[image_hash]
        if time.time() - timestamp < PREDICTION_CACHE_TTL:
            return result
        else:
            # Expired, remove it
            del _prediction_cache[image_hash]
    return None


def _cache_prediction(image_hash: str, result: dict) -> None:
    """Store complete prediction result in cache with current timestamp."""
    _prediction_cache[image_hash] = (result, time.time())


def _get_cached_embedding(image_hash: str) -> Optional[List[float]]:
    """Return cached embedding if exists and not expired, else None."""
    if image_hash in _embedding_cache:
        embedding, timestamp = _embedding_cache[image_hash]
        if time.time() - timestamp < EMBEDDING_CACHE_TTL:
            return embedding
        else:
            # Expired, remove it
            del _embedding_cache[image_hash]
    return None


def _cache_embedding(image_hash: str, embedding: List[float]) -> None:
    """Store embedding in cache with current timestamp."""
    _embedding_cache[image_hash] = (embedding, time.time())


async def api_predict(image_bytes: bytes, *, threshold: Optional[float] = None,
                       include_embedding: bool = False) -> dict:
    # Check if this EXACT image was predicted before (full result cache)
    image_hash = _get_image_hash(image_bytes)
    cached_result = _get_cached_prediction(image_hash)
    if cached_result is not None:
        # Found in cache! Return immediately (saves ~500ms API call)
        return cached_result
    
    form = _file_field("file", image_bytes)
    params = {}
    if threshold is not None:
        params["threshold"] = str(threshold)
    if include_embedding:
        params["include_embedding"] = "true"
    
    result = await _api_request("POST", "/v1/predict", data=form, params=params)
    
    # Cache the FULL result for next time
    _cache_prediction(image_hash, result)
    
    # Also cache the embedding if present (for hybrid usage)
    if include_embedding and "embedding" in result and isinstance(result.get("embedding"), list):
        _cache_embedding(image_hash, result["embedding"])
    
    return result


async def api_predict_batch(images_bytes: List[bytes], *, threshold: Optional[float] = None,
                             include_embedding: bool = False) -> List[Optional[dict]]:
    """
    OPTIMIZED batch prediction with embedding cache shortcut.
    
    For each image:
      1. Check if embedding is cached (from prior learning)
      2. If YES -> return instantly (0ms, no API call)
      3. If NO  -> send to API for inference + embedding extraction
    
    This optimization is especially powerful for repeat spawns and incense clusters.
    Impact: repeat spawns go from 500ms (API) -> 1ms (cache), saving 99% latency.
    """
    
    # Partition images: cached vs uncached
    cached_results = {}  # {index: result_dict}
    uncached_indices = []  # indices that need API call
    uncached_bytes = []    # image bytes for uncached images
    
    for i, image_bytes in enumerate(images_bytes):
        image_hash = _get_image_hash(image_bytes)
        cached_emb = _get_cached_embedding(image_hash)
        
        if cached_emb is not None:
            # Embedding already learned! Return it instantly without API call
            _stats["embedding_cache_hits"] += 1
            cached_results[i] = {
                "ok": True,
                "species": None,  # We only cached the embedding, not the full prediction
                "embedding": cached_emb,
                "_source": "embedding_cache",  # Signal that this came from cache
                "_latency_ms": 1  # Instant
            }
            log.debug(f"Embedding cache HIT for image {i} (hash: {image_hash[:8]}...)")
        else:
            # Need API call for this image
            _stats["embedding_cache_misses"] += 1
            uncached_indices.append(i)
            uncached_bytes.append(image_bytes)
    
    # If ALL images are cached, return immediately (HUGE speedup for repeat clusters!)
    if not uncached_bytes:
        out = [None] * len(images_bytes)
        for i, result in cached_results.items():
            out[i] = result
        log.debug(f"Embedding cache ALL HIT: {len(cached_results)} images, skipped API call")
        return out
    
    # Call API only for UNCACHED images
    if uncached_bytes:
        log.debug(f"Embedding cache PARTIAL HIT: {len(cached_results)}/{len(images_bytes)} cached, sending {len(uncached_bytes)} to API")
        
        form = aiohttp.FormData()
        for i, b in enumerate(uncached_bytes):
            form.add_field("files", b, filename=f"image{i}.jpg", content_type="application/octet-stream")
        params = {}
        if threshold is not None:
            params["threshold"] = str(threshold)
        if include_embedding:
            params["include_embedding"] = "true"
        
        body = await _api_request("POST", "/v1/predict/batch", data=form, params=params)
        results = body.get("results") if isinstance(body, dict) else None
        if not isinstance(results, list):
            snippet = json.dumps(body)[:300] if not isinstance(body, str) else body[:300]
            raise ApiError(200, f"unexpected /v1/predict/batch response (no 'results'): {snippet}")
        
        # Cache embeddings from API for next time, merge with cached results
        for uncached_idx, api_result in zip(uncached_indices, results):
            if api_result and isinstance(api_result, dict) and api_result.get("ok"):
                # Cache the embedding for future repeats
                image_hash = _get_image_hash(uncached_bytes[uncached_indices.index(uncached_idx)])
                if include_embedding and "embedding" in api_result:
                    _cache_embedding(image_hash, api_result["embedding"])
                cached_results[uncached_idx] = api_result
            else:
                cached_results[uncached_idx] = None
    
    # Reconstruct output in original order
    out = []
    for i in range(len(images_bytes)):
        if i in cached_results:
            out.append(cached_results[i])
        else:
            out.append(None)
    
    return out


async def api_predict_urls(urls: List[str], *, threshold: Optional[float] = None,
                           include_embedding: bool = False) -> List[dict]:
    params = {}
    if threshold is not None:
        params["threshold"] = str(threshold)
    if include_embedding:
        params["include_embedding"] = "true"
    body = await _api_request("POST", "/v1/predict/urls", json={"urls": urls}, params=params)
    results = body.get("results") if isinstance(body, dict) else None
    if not isinstance(results, list) or len(results) != len(urls):
        raise ApiError(200, f"unexpected /v1/predict/urls response: {json.dumps(body)[:300]}")
    return results


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


async def api_learn_batch(items: List[Tuple[str, bytes]], *, allow_new: bool = True) -> List[dict]:
    """
    POST /v1/learn/batch with up to BACKFILL_BATCH_SIZE (species, image_bytes) pairs.
    Returns a list of per-item result dicts, same order as `items`.
    """
    if not items:
        return []
    form = aiohttp.FormData()
    for species, img_bytes in items:
        form.add_field("species", species)
        form.add_field("files", img_bytes, filename="image.jpg", content_type="application/octet-stream")
    form.add_field("allow_new", "true" if allow_new else "false")
    body = await _api_request("POST", "/v1/learn/batch", data=form)
    results = body.get("results") if isinstance(body, dict) else None
    if not isinstance(results, list) or len(results) != len(items):
        snippet = json.dumps(body)[:300] if not isinstance(body, str) else body[:300]
        raise ApiError(200, f"unexpected /v1/learn/batch response: {snippet}")
    await _note_bank_version(body.get("bank_version"))
    return results


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


async def _download_parallel(urls: List[str], max_concurrent: int = 4) -> Dict[str, Optional[bytes]]:
    """Download multiple URLs in parallel. Returns {url: bytes or None}."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def fetch_one(url: str) -> Tuple[str, Optional[bytes]]:
        async with semaphore:
            result = await _download(url)
            return url, result
    
    tasks = [fetch_one(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return {url: data for url, data in results}


# ---- Bank-version tracking (drives cache invalidation) ---------------------
_known_bank_version: Optional[int] = None
_known_bank_fp: Optional[str] = None


async def _note_bank_version(version: Optional[int]):
    global _known_bank_version, _known_bank_fp
    if version is None:
        return
    if _known_bank_version is not None and version != _known_bank_version:
        await _cache_clear()
        _known_bank_fp = None
        asyncio.create_task(_refresh_bank_fp())
    _known_bank_version = version


async def _refresh_bank_fp():
    global _known_bank_fp
    try:
        h = await api_health()
        if h.get("bank_version") == _known_bank_version:
            _known_bank_fp = h.get("bank_fingerprint")
    except Exception:
        pass


# ---- Result cache (by image hash) ------------------------------------------
_cache_lock = asyncio.Lock()
_result_cache: "OrderedDict[str, tuple]" = OrderedDict()
_url_cache: "OrderedDict[str, str]" = OrderedDict()
_cache_dirty = False
_stats = {"hits": 0, "misses": 0, "coalesced": 0, "dropped": 0, "batches": 0, "batch_items": 0,
          "embedding_cache_hits": 0, "embedding_cache_misses": 0}


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
    if not CACHE_FILE or CACHE_SIZE <= 0 or not _cache_dirty or not _known_bank_fp:
        return
    payload = {"fp": _known_bank_fp, "results": list(_result_cache.items()), "urls": list(_url_cache.items())}
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
        if not _known_bank_fp or payload.get("fp") != _known_bank_fp:
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
_m_inference: deque = deque()
_m_store: deque = deque()
_m_response: deque = deque()
_m_spawns: deque = deque()
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
    title = (embed.title or "").lower()
    description = (embed.description or "").lower()
    footer_text = (embed.footer.text or "").lower() if embed.footer else ""
    combined = f"{title} {description} {footer_text}"
    return ("wild" in combined and "appeared" in combined) or ("guess the" in combined and "catch" in combined)


def _to_media_proxy(url: str) -> str:
    """
    cdn.discordapp.com (raw storage) measured 150-750ms and erratic from the AI_Model
    host; media.discordapp.net (Discord's image proxy) measured a consistent ~150ms
    from the same host. Route every fetch through the proxy instead of raw storage.

    Also asks the proxy to resize down to SPAWN_IMAGE_SIZE - the model resizes to
    this anyway, so fetching it pre-shrunk cuts the download payload and skips
    that resize work server-side. Discord's media proxy fits-within (preserves
    aspect ratio, no stretching/cropping), so this is lossless in the sense that
    it doesn't change what the model would have seen after its own resize step.
    Any existing query params (e.g. the cdn signature tokens ex/is/hm) are kept.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    host = (parts.hostname or "").lower()
    if host in ("cdn.discordapp.com", "images-ext-1.discordapp.net", "media.discordapp.net"):
        query = dict(parse_qsl(parts.query))
        query["width"] = str(SPAWN_IMAGE_SIZE)
        query["height"] = str(SPAWN_IMAGE_SIZE)
        parts = parts._replace(netloc="media.discordapp.net", query=urlencode(query))
        return parts.geturl()
    return url


def _extract_wild_spawn_image_url(message: discord.Message) -> Optional[str]:
    for embed in message.embeds:
        if not _is_wild_spawn_embed(embed):
            continue
        if embed.image and embed.image.url:
            return _to_media_proxy(embed.image.url)
        if embed.thumbnail and embed.thumbnail.url:
            return _to_media_proxy(embed.thumbnail.url)
    return None


# ---- Identification (via API) ----------------------------------------------

def _result_from_predict(data: dict) -> tuple:
    neighbors = [(n["species"], n["score"]) for n in data.get("neighbors", [])]
    return data["species"], data["score"], neighbors, data.get("embedding")


async def _identify_one(image_bytes: bytes) -> Optional[tuple]:
    try:
        data = await api_predict(image_bytes, threshold=CONFIDENCE_THRESHOLD, include_embedding=AUTO_LEARN)
    except ApiError as e:
        if e.status != 422:
            log.error(f"Predict API error: {e}")
        return None
    except Exception as e:
        log.error(f"Predict API unreachable ({type(e).__name__}): {e}")
        return None
    await _note_bank_version(data.get("bank_version"))
    _m_add(_m_inference, data.get("timing_ms", {}).get("embed", 0.0))
    _m_add(_m_store, data.get("timing_ms", {}).get("match", 0.0))
    return _result_from_predict(data)


async def _identify_batch(images_bytes: List[bytes], info: Optional[dict] = None) -> List[Optional[tuple]]:
    t_api = time.perf_counter()
    try:
        results = await api_predict_batch(images_bytes, threshold=CONFIDENCE_THRESHOLD, include_embedding=AUTO_LEARN)
    except Exception as e:
        log.error(f"Batch predict API error ({type(e).__name__}): {e}")
        return [None] * len(images_bytes)
    if info is not None:
        info["api"] = (time.perf_counter() - t_api) * 1000
        info["srv"] = sum((r.get("timing_ms", {}).get("embed", 0.0) + r.get("timing_ms", {}).get("match", 0.0))
                          for r in results if r) / max(1, sum(1 for r in results if r))
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
            if time.time() - _entry_latest(entry) > MAX_SPAWN_AGE:
                _stats["dropped"] += 1
                fut.set_result(None)
                continue
            live.append((image_bytes, fut, entry))
        if not live:
            continue
        try:
            now_pc = time.perf_counter()
            info = {"n": len(live)}
            for _, _, e in live:
                if e is not None:
                    e["t_queue"] = (now_pc - e.get("t_put", now_pc)) * 1000
            results = await _identify_batch([b for b, _, _ in live], info)
            for _, _, e in live:
                if e is not None:
                    e.update(info)
        except Exception as e:
            log.error(f"Batch inference failed: {type(e).__name__}: {e}")
            results = [None] * len(live)
        _stats["batches"] += 1
        _stats["batch_items"] += len(live)
        for (_, fut, _), res in zip(live, results):
            if not fut.done():
                fut.set_result(res)


async def _submit_identification(image_bytes: bytes, entry: dict):
    global _batch_queue
    if _batch_queue is None:
        _batch_queue = asyncio.Queue()
        _batch_tasks.extend(asyncio.create_task(_batch_worker()) for _ in range(API_CONCURRENCY))
    fut = asyncio.get_running_loop().create_future()
    if entry is not None:
        entry["t_put"] = time.perf_counter()
    _batch_queue.put_nowait((image_bytes, entry, fut))
    return await fut


_url_blocked_hosts: set = set()


def _url_host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


async def _identify_via_server_fetch(url: str, timing: Optional[dict]):
    global URL_MODE
    host = _url_host(url)
    t = time.perf_counter()
    try:
        r = (await api_predict_urls([url], threshold=CONFIDENCE_THRESHOLD, include_embedding=AUTO_LEARN))[0]
    except ApiError as e:
        if e.status in (404, 405):
            log.warning("API has no /v1/predict/urls - switching to download+upload mode")
            URL_MODE = False
        else:
            log.warning(f"URL predict API error: {e}")
        return None
    except Exception as e:
        log.warning(f"URL predict unreachable ({type(e).__name__}): {e}")
        return None
    api_ms = (time.perf_counter() - t) * 1000
    if not r.get("ok"):
        if r.get("code") == "host_not_allowed" and host:
            _url_blocked_hosts.add(host)
            log.info(f"API won't fetch from {host}; using download+upload for that host")
        return None
    await _note_bank_version(r.get("bank_version"))
    tm = r.get("timing_ms", {})
    _m_add(_m_inference, tm.get("embed", 0.0))
    _m_add(_m_store, tm.get("match", 0.0))
    if timing is not None:
        timing["api"] = api_ms
        timing["srv"] = tm.get("embed", 0.0) + tm.get("match", 0.0)
        timing["fetch"] = r.get("fetch_ms", 0.0)
    return _result_from_predict(r), r.get("sha1")


_inflight_url: Dict[str, dict] = {}
_inflight_digest: Dict[str, dict] = {}


async def _coalesced(table: Dict[str, dict], key: str, received: float, work, deps=None):
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


async def _identify_spawn(url: str, received: float, timing: Optional[dict] = None):
    res = await _cache_get_by_url(url) if URL_CACHE else None
    if res is not None:
        _stats["hits"] += 1
        return res
    cache_url = url if URL_CACHE else None

    async def work(url_entry):
        # Prefetch the download in parallel with the API call to reduce latency.
        # If API succeeds, drop the download. If API fails, use pre-fetched bytes.
        download_task = asyncio.create_task(_download(url))
        
        if URL_MODE and _url_host(url) not in _url_blocked_hosts:
            try:
                got = await _identify_via_server_fetch(url, timing)
                if got is not None:
                    download_task.cancel()  # Cancel parallel download, not needed.
                    res, digest = got
                    _stats["misses"] += 1
                    if digest:
                        await _cache_put(digest, res, cache_url)
                    return res
            except Exception:
                pass  # Fall through to download+upload path
            if timing is not None:
                timing.pop("api", None)
        
        # Fallback: retrieve the pre-fetched download.
        t_dl = time.perf_counter()
        try:
            image_bytes = await asyncio.wait_for(download_task, timeout=15.0)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            image_bytes = None
        if timing is not None:
            timing["dl"] = (time.perf_counter() - t_dl) * 1000
        if not image_bytes:
            return None
        digest = hashlib.sha1(image_bytes).hexdigest()
        res = await _cache_get(digest)
        if res is not None:
            _stats["hits"] += 1
            await _cache_put(digest, res, cache_url)
            return res

        async def infer(digest_entry):
            small = await asyncio.get_running_loop().run_in_executor(None, _shrink_for_upload, image_bytes)
            if timing is not None:
                timing["kb"] = len(small) / 1024
            out = await _submit_identification(small, digest_entry)
            if timing is not None:
                for k in ("t_queue", "api", "srv", "n"):
                    if k in digest_entry:
                        timing[k] = digest_entry[k]
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
    _last_spawn[message.channel.id] = None

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
    _last_spawn[message.channel.id] = None

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
    global _saver_started, _known_bank_version, _known_bank_fp
    await _get_session()

    try:
        health = await api_health()
        _known_bank_version = health.get("bank_version")
        _known_bank_fp = health.get("bank_fingerprint")
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
    log.info(f"   Backfill: channels={BACKFILL_CHANNEL_CONCURRENCY}, "
             f"downloads={BACKFILL_DOWNLOAD_CONCURRENCY}, batch={BACKFILL_BATCH_SIZE}")
    log.info("=" * 60)


@bot.event
async def on_message(message: discord.Message):
    global _named_total
    await bot.process_commands(message)

    if message.guild is not None and message.guild.id in DISABLED_GUILDS:
        return

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
    _spawn_gen[cid] = gen

    _last_spawn[cid] = None

    timing: dict = {}
    result = await _identify_spawn(image_url, received, timing)
    if result is None:
        return
    if _spawn_gen.get(cid) != gen:
        return

    winner, score, neighbors, vec = result

    _last_spawn[message.channel.id] = (time.time(), winner, score, vec)

    elapsed_ms = (time.time() - received) * 1000
    confident = score >= CONFIDENCE_THRESHOLD
    api_ms = timing.get("api", 0.0)
    bot_ms = max(elapsed_ms - api_ms, 0.0)
    breakdown = []
    if "fetch" in timing:
        breakdown.append(f"CDN fetch {timing['fetch']:.0f}ms")
    if "srv" in timing:
        breakdown.append(f"embed+match {timing['srv']:.0f}ms")
    if "dl" in timing:
        breakdown.append(f"bot download {timing['dl']:.0f}ms")
    host = _url_host(image_url)
    detail = (
        f" (bot analysis {bot_ms:.0f}ms, API travel time {api_ms:.0f}ms"
        f"{', ' + ', '.join(breakdown) if breakdown else ''}, host={host})"
    )
    log.info(f"Spawn {message.id}: {winner} (score {score:.3f}, confident={confident}, {elapsed_ms:.0f}ms){detail}")

    if not confident and SILENT_BELOW_THRESHOLD:
        return

    display_name = _display(winner)
    conf_pct = display_confidence(winner, score, neighbors) * 100

    # Awaited (with a short timeout) so THIS species' own data is what gets used -
    # never a leftover value from whatever species was looked up before it. Capped
    # at 2s so a slow/unreachable PokeAPI can't stall the spawn reply; if it times
    # out we just fall back to no emoji/plain name for this one reply, and the
    # fetch keeps running in the background so the next sighting has it cached.
    pdata = await pokedata.get_species_data(winner, display_name)
    type_emojis = pokedata.emojis_for(pdata)
    embed_color = pokedata.color_for(pdata, fallback=0x57F287 if confident else 0xFEE75C)

    bar_len = 10
    filled = round((conf_pct / 100) * bar_len) if confident or conf_pct > 0 else 0
    conf_bar = "▰" * filled + "▱" * (bar_len - filled)

    # NOTE: custom emoji only render in an embed's description/field values -
    # Discord's client shows them as literal "<:Name:id>" text anywhere else
    # (author name, title, footer), so the type emoji goes in the description,
    # not in set_author.
    emoji_prefix = "".join(type_emojis) + " " if type_emojis else ""
    embed = discord.Embed(color=embed_color)
    embed.description = (
        f"## {emoji_prefix}{display_name}{'' if confident else ' (unsure)'}\n"
        f"{conf_bar} **{conf_pct:.2f}%**"
    )
    # Always use the species' own main name for the catch command - not the
    # shorter alt-language "best_name" pokedata can find, since that can be a
    # completely different-looking string (e.g. a Japanese romanization) that
    # doesn't match what's actually on screen.
    catch_cmd = CATCH_COMMAND_TEMPLATE.format(name=display_name)
    embed.add_field(name="Command", value=f"```{catch_cmd}```", inline=False)
    embed.set_thumbnail(url=image_url)
    footer = f"{elapsed_ms:.0f}ms"
    if not confident:
        footer += " · low confidence, might be wrong"
    embed.set_footer(text=footer)

    # Who gets pinged: a reserve on this species always wins over a plain collection
    # entry (reserve = "this one's claimed"), plus anyone currently shiny hunting it
    # if this spawn looks shiny.
    ping_ids: Set[int] = set()
    guild_id = message.guild.id if message.guild else None
    if guild_id is not None:
        reserved_to = guild_store.reserve_matches(guild_id, winner)
        ping_ids |= reserved_to if reserved_to else guild_store.collection_matches(guild_id, winner)
        if _SHINY_RE.search(_message_text(message)):
            ping_ids |= guild_store.shiny_matches(guild_id, winner)

    ping_text = " ".join(f"<@{uid}>" for uid in ping_ids)
    copy_line = f"**Click to Copy ->** `{catch_cmd}`"
    content = f"{ping_text}\n{copy_line}" if ping_text else copy_line
    try:
        await message.reply(
            content=content, embed=embed,
            allowed_mentions=discord.AllowedMentions(users=True, everyone=False, roles=False),
        )
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

    lines = [f"**Best guess:** {_display(winner)} - {display_confidence(winner, score, neighbors) * 100:.1f}% (similarity {score * 100:.1f}%)"]
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


def _backfill_status_line(stats: dict, channel: discord.abc.GuildChannel) -> str:
    return (
        f"Backfill scanning **#{getattr(channel, 'name', channel.id)}** "
        f"({stats['channels_done']}/{stats['channels_total']} channels)\n"
        f"Spawns seen: {stats['spawns']:,} | Pairs found: {stats['pairs']:,} | "
        f"Learned: {stats['learned']:,} | Dup: {stats['duplicate']:,} | "
        f"Unknown: {stats['unknown']:,} | Errors: {stats['errors']:,}"
    )


_BACKFILL_AUDIT_PATH = os.getenv("BACKFILL_AUDIT_PATH", "backfill_audit.csv")
_backfill_audit_lock = asyncio.Lock()


async def _backfill_audit_log(channel_name: str, message_id: int, species: str, image_url: str, status: str):
    """Appends every (species, image) pair backfill has ever seen to a CSV, dry-run or not,
    so you can spot-check real pairs later - open the CSV, click a few image_url values,
    and eyeball whether they actually show the species column next to them."""
    row = f'{time.strftime("%Y-%m-%d %H:%M:%S")},{channel_name},{message_id},"{species}",{image_url},{status}\n'
    async with _backfill_audit_lock:
        try:
            new_file = not os.path.exists(_BACKFILL_AUDIT_PATH)
            with open(_BACKFILL_AUDIT_PATH, "a", encoding="utf-8") as f:
                if new_file:
                    f.write("timestamp,channel,message_id,species,image_url,status\n")
                f.write(row)
        except OSError as e:
            log.warning(f"Backfill: couldn't write audit log: {e}")


_BACKFILL_CK_KEY = "backfill_checkpoint"


def _load_backfill_checkpoint_file() -> dict:
    try:
        with open(BACKFILL_CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        log.error(f"Couldn't parse {BACKFILL_CHECKPOINT_PATH}, starting with no checkpoint: {e}")
        return {}


def _load_backfill_checkpoint() -> dict:
    if not turso.enabled:
        return _load_backfill_checkpoint_file()
    try:
        raw = turso.kv_get(_BACKFILL_CK_KEY)
        if raw is not None:
            return json.loads(raw)
        if turso.kv_get("migrated:backfill_checkpoint") is None:   # one-time import of old file
            ck = _load_backfill_checkpoint_file()
            if ck:
                turso.kv_set(_BACKFILL_CK_KEY, json.dumps(ck))
                log.info(f"Imported {BACKFILL_CHECKPOINT_PATH} into Turso")
            turso.kv_set("migrated:backfill_checkpoint", "1")
            return ck
        return {}
    except Exception as e:
        log.error(f"Couldn't load backfill checkpoint from Turso: {e}")
        return {}


async def _save_backfill_checkpoint(checkpoint: dict) -> None:
    """Turso: queued to the background writer (atomic upsert, never blocks the loop).
    Fallback: atomic tmp-file + rename so a crash mid-save can't corrupt the file."""
    async with _backfill_checkpoint_lock:
        if turso.enabled:
            turso.kv_set_async(_BACKFILL_CK_KEY, json.dumps(checkpoint))
            return
        try:
            tmp = BACKFILL_CHECKPOINT_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f)
            os.replace(tmp, BACKFILL_CHECKPOINT_PATH)
        except OSError as e:
            log.warning(f"Backfill: couldn't save checkpoint: {e}")


@bot.command(name="backfill")
@commands.is_owner()
async def backfill_cmd(ctx: commands.Context, scope: str = "channel", limit: Optional[int] = None,
                        *flags: str):
    """Scan old spawn history and self-label it (see s!help backfill).
    s!backfill [channel|guild|all] [max_per_channel] [dry] [fresh] [from:<channel_id>]
    s!backfill stop - pause a running backfill; progress is checkpointed.

    Reads BACKFILL_CHANNEL_CONCURRENCY channels in parallel and batches
    /v1/learn calls BACKFILL_BATCH_SIZE at a time, so scan speed is limited by
    Discord's history endpoint, not by the model server.
    """
    global _backfill_cancel, _backfill_active

    scope = scope.lower()
    if scope in ("stop", "cancel", "pause"):
        if not _backfill_active:
            await ctx.send("No backfill is currently running.")
            return
        _backfill_cancel = True
        await ctx.send("Stopping... it'll finish the message it's on, save a checkpoint, then stop. "
                        f"Run `{COMMAND_PREFIX}backfill` again later (same scope) to resume from there.")
        return

    if _backfill_active:
        await ctx.send(f"A backfill is already running. Use `{COMMAND_PREFIX}backfill stop` to pause it first.")
        return

    if scope not in ("channel", "guild", "all"):
        await ctx.send('Scope must be `channel`, `guild`, or `all` (e.g. `s!backfill guild 20000`).')
        return

    flag_words = {f.strip().lower() for f in flags}
    dry_run = bool(flag_words & {"dry", "dryrun", "dry-run", "preview"})
    fresh = bool(flag_words & {"fresh", "restart", "reset"})

    start_channel_id: Optional[int] = None
    for f in flags:
        fl = f.strip().lower()
        if fl.startswith(("from:", "from=", "start:", "start=")):
            _, _, val = fl.partition(":") if ":" in fl else fl.partition("=")
            try:
                start_channel_id = int(val.strip())
            except ValueError:
                await ctx.send(f"Couldn't read a channel ID out of `{f}` - expected e.g. `from:1490688685342068766`.")
                return

    if scope == "channel":
        channels = [ctx.channel]
    elif scope == "guild":
        if ctx.guild is None:
            await ctx.send("This isn't a server channel.")
            return
        channels = [c for c in ctx.guild.text_channels
                    if c.permissions_for(ctx.guild.me).read_message_history]
    else:
        channels = []
        for g in bot.guilds:
            channels += [c for c in g.text_channels if c.permissions_for(g.me).read_message_history]

    if start_channel_id is not None:
        idx = next((i for i, c in enumerate(channels) if c.id == start_channel_id), None)
        if idx is None:
            await ctx.send(
                f"Couldn't find channel `{start_channel_id}` in scope `{scope}` "
                f"(wrong ID, no read-history permission, wrong server, or not a text channel)."
            )
            return
        skipped = channels[:idx]
        channels = channels[idx:]
        if skipped:
            await ctx.send(f"Skipping {len(skipped)} channel(s) before <#{start_channel_id}> "
                            f"(not touched, not marked done - just left alone this run).")

    if not channels:
        await ctx.send("No readable text channels found for that scope.")
        return
    history_limit = limit if (limit and limit > 0) else None

    checkpoint = {} if fresh else _load_backfill_checkpoint()
    resuming = (not fresh) and any(str(c.id) in checkpoint for c in channels)

    stats = {"channels_total": len(channels), "channels_done": 0, "spawns": 0,
             "pairs": 0, "learned": 0, "duplicate": 0, "unknown": 0, "errors": 0}

    status_msg = await ctx.send(
        f"Backfill starting across {len(channels)} channel(s) "
        f"(channel_concurrency={BACKFILL_CHANNEL_CONCURRENCY}, batch={BACKFILL_BATCH_SIZE})"
        f"{' (resuming from checkpoint)' if resuming else ''}..."
    )
    last_edit = 0.0
    _backfill_cancel = False
    _backfill_active = True

    # ---- shared queue + batch workers -----------------------------------
    pair_queue: asyncio.Queue = asyncio.Queue(maxsize=BACKFILL_QUEUE_MAX)
    download_sem = asyncio.Semaphore(BACKFILL_DOWNLOAD_CONCURRENCY)
    workers_stop = asyncio.Event()

    def _bump(status: str):
        if status in stats:
            stats[status] += 1
        else:
            stats["errors"] += 1

    async def _process_item(channel_name: str, message_id: int, url: str, species: str):
        """Download + resize one image. Returns (species, bytes) or None on failure."""
        async with download_sem:
            try:
                img = await _download(url)
            except Exception as e:
                stats["errors"] += 1
                log.warning(f"Backfill: download failed for {url}: {type(e).__name__}: {e}")
                await _backfill_audit_log(channel_name, message_id, species, url, "error:download")
                return None
        if img is None:
            stats["errors"] += 1
            await _backfill_audit_log(channel_name, message_id, species, url, "error:download_none")
            return None
        try:
            small = await asyncio.get_running_loop().run_in_executor(None, _shrink_for_upload, img)
        except Exception:
            small = img
        return species, small

    async def _batch_worker(worker_id: int):
        """Pulls up to BACKFILL_BATCH_SIZE items from the queue, submits them in one call."""
        while not workers_stop.is_set():
            try:
                first = await asyncio.wait_for(pair_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                return
            batch_meta = [first]
            while len(batch_meta) < BACKFILL_BATCH_SIZE:
                try:
                    batch_meta.append(pair_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            if dry_run:
                for channel_name, message_id, url, species in batch_meta:
                    log.info(f"Backfill [DRY] #{channel_name}: would learn '{species}' <- {url}")
                    await _backfill_audit_log(channel_name, message_id, species, url, "dry_run")
                continue

            results = await asyncio.gather(
                *(_process_item(cn, mid, u, sp) for cn, mid, u, sp in batch_meta),
                return_exceptions=False,
            )
            items_for_learn = []
            kept_meta = []
            for meta, res in zip(batch_meta, results):
                if res is None:
                    continue
                items_for_learn.append(res)
                kept_meta.append(meta)

            if not items_for_learn:
                continue

            try:
                api_results = await api_learn_batch(items_for_learn, allow_new=True)
            except Exception as e:
                stats["errors"] += len(items_for_learn)
                log.warning(f"Backfill: /v1/learn/batch failed ({type(e).__name__}: {e})")
                for cn, mid, u, sp in kept_meta:
                    await _backfill_audit_log(cn, mid, sp, u, "error:api")
                continue

            for (cn, mid, u, sp), r in zip(kept_meta, api_results):
                status = r.get("status", "error")
                _bump(status)
                log.info(f"Backfill #{cn}: '{sp}' -> {status}")
                await _backfill_audit_log(cn, mid, sp, u, status)

    # ---- per-channel scanner -------------------------------------------
    async def _scan_channel(channel) -> bool:
        key = str(channel.id)
        entry = checkpoint.get(key)
        if entry and entry.get("done") and not fresh:
            stats["channels_done"] += 1
            return True

        pending_url: Optional[str] = entry.get("pending_url") if entry else None
        pending_msg_id: Optional[int] = entry.get("pending_msg_id") if entry else None
        last_seen_id: Optional[int] = entry.get("last_message_id") if entry else None
        after_obj = discord.Object(id=last_seen_id) if last_seen_id else None
        cancelled = False

        try:
            async for message in channel.history(limit=history_limit, oldest_first=True, after=after_obj):
                if _backfill_cancel:
                    cancelled = True
                    break
                last_seen_id = message.id

                if message.author.id == SPAWN_BOT_ID and any(_is_wild_spawn_embed(e) for e in message.embeds):
                    stats["spawns"] += 1
                    fled_match = _FLED_RE.search(_message_text(message))
                    if fled_match and pending_url:
                        species = fled_match.group(1).strip().strip(".!").strip()
                        if species:
                            stats["pairs"] += 1
                            await pair_queue.put(
                                (getattr(channel, "name", str(channel.id)), pending_msg_id, pending_url, species)
                            )
                    pending_url = _extract_wild_spawn_image_url(message)
                    pending_msg_id = message.id

                if BACKFILL_CHANNEL_DELAY > 0 and last_seen_id and (last_seen_id % 100 == 0):
                    await asyncio.sleep(BACKFILL_CHANNEL_DELAY)

        except discord.Forbidden:
            log.warning(f"Backfill: no permission to read history in #{getattr(channel, 'name', channel.id)}")
        except Exception as e:
            log.error(f"Backfill: error scanning #{getattr(channel, 'name', channel.id)}: {type(e).__name__}: {e}")

        checkpoint[key] = {"last_message_id": last_seen_id, "pending_url": pending_url,
                           "pending_msg_id": pending_msg_id, "done": not cancelled}
        await _save_backfill_checkpoint(checkpoint)

        if not cancelled:
            stats["channels_done"] += 1
        return not cancelled

    # ---- orchestrator ---------------------------------------------------
    worker_tasks = [asyncio.create_task(_batch_worker(i))
                    for i in range(max(1, BACKFILL_CHANNEL_CONCURRENCY))]

    channel_sem = asyncio.Semaphore(max(1, BACKFILL_CHANNEL_CONCURRENCY))

    async def _guarded_scan(channel):
        async with channel_sem:
            ok = await _scan_channel(channel)
            return channel, ok

    try:
        scan_tasks = [asyncio.create_task(_guarded_scan(c)) for c in channels]
        pending_scans = set(scan_tasks)
        cancelled_any = False
        while pending_scans:
            done, pending_scans = await asyncio.wait(
                pending_scans, timeout=8, return_when=asyncio.FIRST_COMPLETED
            )
            for t in done:
                try:
                    channel, ok = t.result()
                    if not ok:
                        cancelled_any = True
                except Exception as e:
                    log.error(f"Backfill: channel task crashed: {type(e).__name__}: {e}")
            try:
                any_channel = channels[0] if channels else ctx.channel
                await status_msg.edit(content=_backfill_status_line(stats, any_channel))
            except discord.HTTPException:
                pass

        while not pair_queue.empty():
            await asyncio.sleep(0.5)
            try:
                any_channel = channels[0] if channels else ctx.channel
                await status_msg.edit(content=_backfill_status_line(stats, any_channel))
            except discord.HTTPException:
                pass

        await asyncio.sleep(1.0)

        if cancelled_any or _backfill_cancel:
            await ctx.send(
                f"**Backfill stopped** (checkpoint saved).\n"
                f"Spawns seen: {stats['spawns']:,} | Pairs found: {stats['pairs']:,} | "
                f"Learned: {stats['learned']:,} | Already had: {stats['duplicate']:,} | "
                f"Unknown species: {stats['unknown']:,} | Errors: {stats['errors']:,}\n"
                f"Run `{COMMAND_PREFIX}backfill {scope}"
                f"{' ' + str(limit) if limit else ''}` to resume from here, "
                f"or add `fresh` to start that channel over."
            )
            return

        await ctx.send(
            f"**Backfill complete** across {stats['channels_total']} channel(s).\n"
            f"Spawns seen: {stats['spawns']:,} | Pairs found: {stats['pairs']:,} | "
            f"Learned: {stats['learned']:,} | Already had: {stats['duplicate']:,} | "
            f"Unknown species: {stats['unknown']:,} | Errors: {stats['errors']:,}\n"
            + (f"Run `{COMMAND_PREFIX}reload` if the API doesn't pick up new examples automatically."
               if stats["learned"] else "")
        )
        for channel in channels:
            checkpoint.pop(str(channel.id), None)
        await _save_backfill_checkpoint(checkpoint)
    finally:
        workers_stop.set()
        for t in worker_tasks:
            t.cancel()
        _backfill_active = False


@bot.command(name="bulklearn", aliases=["learnfolder", "learnarchive"])
@commands.is_owner()
async def bulklearn_cmd(ctx: commands.Context, *flags: str):
    """
    Bulk-teach the bot from an uploaded archive, one folder per species:

        my_pokemons.zip
          pikachu/
            img1.jpg
            img2.png
          charizard/
            img1.jpg

    Attach the .zip/.rar/.7z to your message (or reply to a message that has
    one) and run:
        s!bulklearn              - teach everything, auto-creating new species
        s!bulklearn dry          - preview species/counts, teaches nothing
        s!bulklearn known-only   - skip folders that don't match an existing
                                    species instead of auto-creating them

    Uses the same POST /v1/learn/batch endpoint as s!learn and s!backfill, so
    this is purely additive: it appends to bank/learned.jsonl on the AI_Model
    server. It never touches base_features.npy / base_species.npy and never
    needs a retrain - the API picks it up immediately.
    """
    global _bulklearn_active

    if _bulklearn_active:
        await ctx.send("A bulklearn run is already in progress - wait for it to finish.")
        return

    flag_words = {f.strip().lower() for f in flags}
    dry_run = bool(flag_words & {"dry", "dryrun", "dry-run", "preview"})
    allow_new = not bool(flag_words & {"known-only", "known", "no-new", "strict"})

    # ---- find the archive attachment (this message, or the one replied to) ----
    attachment = None
    for att in ctx.message.attachments:
        if att.filename.lower().endswith(_BULKLEARN_ARCHIVE_EXT):
            attachment = att
            break
    if attachment is None and ctx.message.reference is not None:
        replied = ctx.message.reference.resolved
        if not isinstance(replied, discord.Message):
            try:
                replied = await ctx.channel.fetch_message(ctx.message.reference.message_id)
            except discord.HTTPException:
                replied = None
        if replied is not None:
            for att in replied.attachments:
                if att.filename.lower().endswith(_BULKLEARN_ARCHIVE_EXT):
                    attachment = att
                    break

    if attachment is None:
        await ctx.send(
            "Attach a `.zip`, `.rar`, or `.7z` file (or reply to a message that has one) "
            f"when using `{COMMAND_PREFIX}bulklearn`. It should contain one folder per "
            "species, e.g. `pikachu/img1.jpg`."
        )
        return

    size_mb = attachment.size / (1024 * 1024)
    if size_mb > BULKLEARN_MAX_ARCHIVE_MB:
        await ctx.send(
            f"That archive is {size_mb:.0f} MB - larger than the {BULKLEARN_MAX_ARCHIVE_MB} MB "
            "safety cap for this command. Split it into smaller archives, or raise "
            "BULKLEARN_MAX_ARCHIVE_MB in your .env if this bot's host has room for it."
        )
        return

    _bulklearn_active = True
    status_msg = await ctx.send(f"Downloading `{attachment.filename}` ({size_mb:.1f} MB)...")
    tmp_dir = Path(tempfile.mkdtemp(prefix="bulklearn_"))

    try:
        archive_path = tmp_dir / attachment.filename
        try:
            await attachment.save(archive_path)
        except discord.HTTPException as e:
            await status_msg.edit(content=f"Couldn't download the attachment: {e}")
            return

        await status_msg.edit(content=f"Extracting `{attachment.filename}`...")
        extract_dir = tmp_dir / "extracted"
        extract_dir.mkdir(exist_ok=True)
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, _extract_archive_to, archive_path, extract_dir
            )
        except Exception as e:
            await status_msg.edit(content=f"Couldn't extract that archive: {type(e).__name__}: {e}")
            return

        species_root = _find_species_root(extract_dir)
        pairs = list(_iter_species_images(species_root))
        if not pairs:
            await status_msg.edit(
                content="No images found. Make sure the archive has one folder per "
                        "species with images directly inside (e.g. `pikachu/img1.jpg`)."
            )
            return

        by_species: Dict[str, int] = {}
        for sp, _ in pairs:
            by_species[sp] = by_species.get(sp, 0) + 1

        if dry_run:
            lines = [f"**Dry run** - found {len(pairs)} image(s) across {len(by_species)} species:"]
            for sp, n in sorted(by_species.items(), key=lambda kv: -kv[1])[:25]:
                lines.append(f"  {sp:<20s} {n}")
            if len(by_species) > 25:
                lines.append(f"  ...and {len(by_species) - 25} more species")
            lines.append(f"\nRun `{COMMAND_PREFIX}bulklearn` (no `dry`) to actually teach these.")
            await status_msg.edit(content="\n".join(lines))
            return

        stats = {"learned": 0, "duplicate": 0, "unknown": 0, "errors": 0}
        unknown_species: set = set()
        total = len(pairs)
        done = 0
        last_edit = 0.0

        queue: asyncio.Queue = asyncio.Queue(maxsize=BULKLEARN_BATCH_SIZE * BULKLEARN_WORKERS * 2)

        async def _producer():
            for sp, path in pairs:
                try:
                    raw = await asyncio.get_running_loop().run_in_executor(None, path.read_bytes)
                except Exception:
                    stats["errors"] += 1
                    continue
                try:
                    small = await asyncio.get_running_loop().run_in_executor(None, _shrink_for_upload, raw)
                except Exception:
                    small = raw
                await queue.put((sp, small))
            for _ in range(BULKLEARN_WORKERS):
                await queue.put(None)

        async def _worker():
            nonlocal done, last_edit
            while True:
                first = await queue.get()
                if first is None:
                    return
                batch = [first]
                while len(batch) < BULKLEARN_BATCH_SIZE:
                    try:
                        item = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if item is None:
                        await queue.put(None)  # let sibling workers see the sentinel too
                        break
                    batch.append(item)

                try:
                    results = await api_learn_batch(batch, allow_new=allow_new)
                except Exception as e:
                    stats["errors"] += len(batch)
                    log.warning(f"bulklearn: /v1/learn/batch failed: {type(e).__name__}: {e}")
                    results = []

                for (sp, _), r in zip(batch, results):
                    status = r.get("status", "error")
                    if status in stats:
                        stats[status] += 1
                    else:
                        stats["errors"] += 1
                    if status == "unknown":
                        unknown_species.add(sp)
                done += len(batch)

                now = time.time()
                if now - last_edit > 3:
                    last_edit = now
                    try:
                        await status_msg.edit(content=_bulklearn_status_line(done, total, stats))
                    except discord.HTTPException:
                        pass

        producer_task = asyncio.create_task(_producer())
        worker_tasks = [asyncio.create_task(_worker()) for _ in range(BULKLEARN_WORKERS)]
        await producer_task
        await asyncio.gather(*worker_tasks)

        summary = (
            f"**Bulklearn complete** - {total} image(s) across {len(by_species)} species.\n"
            f"Learned: {stats['learned']:,} | Already had: {stats['duplicate']:,} | "
            f"Unknown species: {stats['unknown']:,} | Errors: {stats['errors']:,}"
        )
        if unknown_species and not allow_new:
            sample = ", ".join(sorted(unknown_species)[:15])
            summary += (
                f"\nSkipped unknown species (folder name didn't match any existing species): "
                f"{sample}{'...' if len(unknown_species) > 15 else ''}\n"
                f"Re-run without `known-only` to auto-create them, or fix the folder names."
            )
        await status_msg.edit(content=summary)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        _bulklearn_active = False


@bot.command(name="naming")
@commands.is_owner()
async def naming_cmd(ctx: commands.Context, state: Optional[str] = None):
    """s!naming [on|off] - turn spawn identification (and auto-learn) on/off for this server."""
    if ctx.guild is None:
        await ctx.send("This isn't a server channel.")
        return

    if state is None:
        off = ctx.guild.id in DISABLED_GUILDS
        await ctx.send(f"Naming is currently **{'off' if off else 'on'}** in {ctx.guild.name}.")
        return

    state = state.lower()
    if state in ("off", "disable", "false", "0"):
        if ctx.guild.id not in DISABLED_GUILDS:
            DISABLED_GUILDS.add(ctx.guild.id)
            _save_disabled_guilds()
        await ctx.send(f"Naming turned **off** in {ctx.guild.name}. The bot will ignore spawns here until `{COMMAND_PREFIX}naming on`.")
    elif state in ("on", "enable", "true", "1"):
        if ctx.guild.id in DISABLED_GUILDS:
            DISABLED_GUILDS.discard(ctx.guild.id)
            _save_disabled_guilds()
        await ctx.send(f"Naming turned **on** in {ctx.guild.name}.")
    else:
        await ctx.send(f"Usage: `{COMMAND_PREFIX}naming on` or `{COMMAND_PREFIX}naming off`.")


@bot.command(name="forget")
@commands.is_owner()
async def forget_cmd(ctx: commands.Context, *, species: str):
    """Remove the examples you taught for a species."""
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
    """Remove the most recent example that was taught."""
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
    """Live performance dashboard."""
    await ctx.send(embed=await _build_api_embed())


CHECKLIST_PAGE_SIZE = 15


def _pretty_species(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").title()


class _ChecklistSearchModal(discord.ui.Modal, title="Search Pokedex Checklist"):
    query = discord.ui.TextInput(
        label="Name contains...",
        placeholder="e.g. char, pika, eevee (leave blank to clear)",
        required=False,
        max_length=64,
    )

    def __init__(self, view: "ChecklistView"):
        super().__init__()
        self.checklist_view = view
        self.query.default = view.search or ""

    async def on_submit(self, interaction: discord.Interaction):
        self.checklist_view.search = self.query.value.strip() or None
        self.checklist_view.page = 0
        await self.checklist_view.refresh(interaction)


class ChecklistView(discord.ui.View):
    """
    Interactive browser for s!checklist: search by name, sort by image
    count (ascending/descending), and page through the results.
    """

    def __init__(self, owner_id: int, counts: Dict[str, int], *, timeout: float = 180.0):
        super().__init__(timeout=timeout)
        self.owner_id = owner_id
        self.all_counts = counts  # species -> taught image count (unfiltered)
        self.search: Optional[str] = None
        self.descending = True  # True = highest image count first
        self.page = 0
        self.message: Optional[discord.Message] = None
        self._sync_buttons()

    def _filtered_sorted(self) -> List[Tuple[str, int]]:
        items = list(self.all_counts.items())
        if self.search:
            q = self.search.lower()
            items = [(sp, n) for sp, n in items if q in sp.lower()]
        # Secondary key keeps ties in a stable, readable (alphabetical) order.
        items.sort(key=lambda kv: (kv[1], kv[0].lower()), reverse=self.descending)
        return items

    def _sync_buttons(self):
        self.sort_button.label = "Highest first" if self.descending else "Lowest first"
        self.sort_button.emoji = "🔽" if self.descending else "🔼"
        self.search_button.label = f'Search: "{self.search}"' if self.search else "Search"
        self.clear_button.disabled = self.search is None

    def build_embed(self) -> discord.Embed:
        items = self._filtered_sorted()
        total = len(items)
        pages = max(1, math.ceil(total / CHECKLIST_PAGE_SIZE))
        self.page = max(0, min(self.page, pages - 1))
        start = self.page * CHECKLIST_PAGE_SIZE
        chunk = items[start:start + CHECKLIST_PAGE_SIZE]

        self.prev_button.disabled = self.page <= 0
        self.next_button.disabled = self.page >= pages - 1

        title = "Pokedex Checklist"
        if self.search:
            title += f' - matching "{self.search}"'
        embed = discord.Embed(title=title, color=discord.Color.blurple())

        if not chunk:
            embed.description = "No Pokemon match that search." if self.search else "Nothing taught yet."
        else:
            embed.description = "\n".join(
                f"**{_pretty_species(sp)}** - {n:,} image{'s' if n != 1 else ''}" for sp, n in chunk
            )

        order = "highest to lowest" if self.descending else "lowest to highest"
        scope = f" (of {len(self.all_counts):,} total)" if self.search else ""
        embed.set_footer(text=f"Page {self.page + 1}/{pages} - {total:,} species{scope} - sorted {order}")
        self._sync_buttons()
        return embed

    async def refresh(self, interaction: discord.Interaction):
        embed = self.build_embed()
        if interaction.response.is_done():
            await interaction.edit_original_response(embed=embed, view=self)
        else:
            await interaction.response.edit_message(embed=embed, view=self)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message(
                "Only the person who ran s!checklist can use these buttons.", ephemeral=True
            )
            return False
        return True

    async def on_timeout(self):
        for item in self.children:
            item.disabled = True
        if self.message:
            try:
                await self.message.edit(view=self)
            except discord.HTTPException:
                pass

    @discord.ui.button(label="Prev", style=discord.ButtonStyle.secondary, emoji="⬅")
    async def prev_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.page -= 1
        await self.refresh(interaction)

    @discord.ui.button(label="Next", style=discord.ButtonStyle.secondary, emoji="➡")
    async def next_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.page += 1
        await self.refresh(interaction)

    @discord.ui.button(label="Highest first", style=discord.ButtonStyle.primary, emoji="🔽")
    async def sort_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.descending = not self.descending
        self.page = 0
        await self.refresh(interaction)

    @discord.ui.button(label="Search", style=discord.ButtonStyle.success, emoji="🔍")
    async def search_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(_ChecklistSearchModal(self))

    @discord.ui.button(label="Clear", style=discord.ButtonStyle.danger, emoji="✖", disabled=True)
    async def clear_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.search = None
        self.page = 0
        await self.refresh(interaction)


@bot.command(name="checklist", aliases=["dex", "pokedex"])
@commands.is_owner()
async def checklist_cmd(ctx: commands.Context, *, search: Optional[str] = None):
    """
    Browse per-species taught-image counts, e.g. "Charizard - 1000 images".
    s!checklist            - full list, most images first
    s!checklist charizard  - only species with "charizard" in the name
    Buttons on the message let you flip Highest/Lowest-first sort, search
    without retyping the command, and page through the results.
    """
    try:
        data = await api_species_counts()
    except ApiError as e:
        await ctx.send(f"Couldn't fetch the species list from the API: {e}")
        return
    except Exception as e:
        await ctx.send(f"Couldn't reach the AI_Model API: {type(e).__name__}: {e}")
        return

    counts = data.get("species")
    if not isinstance(counts, dict) or not counts:
        await ctx.send("The API doesn't know any species yet (empty feature bank).")
        return

    view = ChecklistView(ctx.author.id, counts)
    if search:
        view.search = search.strip() or None
    embed = view.build_embed()
    view.message = await ctx.send(embed=embed, view=view)


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
    """Ask the AI_Model API to re-load its model + feature bank."""
    global _known_bank_version, _known_bank_fp
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
        _known_bank_fp = health.get("bank_fingerprint")
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


def _parse_species_list(raw: str) -> List[str]:
    """'rayquaza, vanillite  charizard' -> ['rayquaza', 'vanillite', 'charizard']"""
    parts = re.split(r"[,\n]+", raw)
    out: List[str] = []
    for p in parts:
        out.extend(p.split())
    return [p.strip() for p in out if p.strip()]


async def _has_res_role(ctx: commands.Context) -> bool:
    if ctx.guild is None:
        return False
    if ctx.author.guild_permissions.manage_guild or await bot.is_owner(ctx.author):
        return True
    role_id = guild_store.get_res_role(ctx.guild.id)
    if role_id is None:
        return False
    return any(r.id == role_id for r in getattr(ctx.author, "roles", []))


@bot.command(name="ping")
async def ping_cmd(ctx: commands.Context):
    """Bot performance: websocket latency, API latency, CPU/RAM, uptime."""
    ws_ms = bot.latency * 1000
    d = await _api_snapshot()
    _, _, r_avg, r_last = d["response"]
    uptime = int(time.time() - _BOT_START_TIME)
    days, rem = divmod(uptime, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)

    embed = discord.Embed(title="🏓 Pong!", color=discord.Color.blurple())
    embed.add_field(name="WebSocket Latency", value=f"`{ws_ms:.2f} ms`")
    embed.add_field(name="API Latency", value=f"`{_fmt_ms(r_last)}`" if r_last else "`-`")
    if psutil:
        proc = psutil.Process()
        embed.add_field(name="CPU Usage", value=f"`{proc.cpu_percent(interval=0.1):.1f}%`")
        mem = proc.memory_info().rss / (1024 * 1024)
        total = psutil.virtual_memory().total / (1024 * 1024)
        embed.add_field(name="RAM Usage", value=f"`{mem:,.0f} MB / {total:,.0f} MB`")
    embed.add_field(name="System Uptime", value=f"`{days}d, {hours:02}:{minutes:02}:{seconds:02}`")
    await ctx.send(embed=embed)


@bot.command(name="model")
async def model_cmd(ctx: commands.Context):
    """Which model/backend the AI_Model API is running, plus recent latency."""
    try:
        health = await api_health()
        stats = await api_stats()
    except Exception as e:
        await ctx.send(f"Couldn't reach the AI_Model API: {type(e).__name__}: {e}")
        return
    d = await _api_snapshot()
    _, _, i_avg, i_last = d["inference"]
    embed = discord.Embed(title="Model Info", color=discord.Color.blurple())
    embed.add_field(name="Backend", value=f"`{health.get('backend') or stats.get('backend', '?')}`")
    embed.add_field(name="Model", value=f"`{stats.get('model', '?')}`")
    embed.add_field(name="Species Loaded", value=f"`{health.get('species', stats.get('species', '?'))}`")
    embed.add_field(name="Feature Vectors", value=f"`{health.get('vectors', stats.get('vectors', '?')):,}`"
                     if isinstance(health.get('vectors', stats.get('vectors')), int) else "`?`")
    embed.add_field(name="Embed Latency (latest)", value=f"`{_fmt_ms(i_last)}`")
    embed.add_field(name="Embed Latency (avg)", value=f"`{_fmt_ms(i_avg)}`")
    embed.set_footer(text=f"API: {AI_MODEL_API_URL}")
    await ctx.send(embed=embed)


@bot.group(name="cl", invoke_without_command=True)
async def cl_cmd(ctx: commands.Context, action: Optional[str] = None, *, species: str = ""):
    """
    s!cl add <pokemon> [pokemon2 ...]   - get pinged in this server when it spawns
    s!cl remove <pokemon> [pokemon2 ...]
    s!cl list
    s!cl clear                          - empty your whole collection in this server
    """
    if ctx.guild is None:
        await ctx.send("This only works in a server.")
        return
    action = (action or "").lower()
    names = _parse_species_list(species)
    if action == "add" and names:
        added = guild_store.collection_add(ctx.guild.id, ctx.author.id, names)
        await ctx.send(f"**Added:** {', '.join(_display(n) for n in added)}" if added
                        else "Already in your collection.")
    elif action == "remove" and names:
        removed = guild_store.collection_remove(ctx.guild.id, ctx.author.id, names)
        await ctx.send(f"**Removed:** {', '.join(_display(n) for n in removed)}" if removed
                        else "None of those were in your collection.")
    elif action == "clear":
        n = guild_store.collection_clear(ctx.guild.id, ctx.author.id)
        await ctx.send(f"Cleared your collection ({n} Pokemon removed)." if n
                        else "Your collection is already empty.")
    elif action in ("list", "", None):
        mine = guild_store.collection_list(ctx.guild.id, ctx.author.id)
        await ctx.send("Your collection is empty." if not mine else
                        "**Your collection:** " + ", ".join(_display(n) for n in mine))
    else:
        await ctx.send(f"Usage: `{COMMAND_PREFIX}cl add <pokemon>`, `{COMMAND_PREFIX}cl remove <pokemon>`, "
                        f"`{COMMAND_PREFIX}cl list`, or `{COMMAND_PREFIX}cl clear`.")


@bot.group(name="res", invoke_without_command=True)
async def res_cmd(ctx: commands.Context, action: Optional[str] = None, *, rest: str = ""):
    """
    s!res add <pokemon> [pokemon2 ...] @user   - reserve species to @user (silences their collection ping)
    s!res remove <pokemon> [pokemon2 ...] @user
    s!res list [@user]
    s!res search <pokemon> [pokemon2 ...]       - see who has each species reserved
    s!res clear                                 - wipe ALL reservations in this server (res role / Manage Server)
    s!res role @role                            - (Manage Server) set who's allowed to use res add/remove
    """
    if ctx.guild is None:
        await ctx.send("This only works in a server.")
        return
    action = (action or "").lower()

    if action == "role":
        if not ctx.author.guild_permissions.manage_guild:
            await ctx.send("Only someone with **Manage Server** can set the res role.")
            return
        if ctx.message.role_mentions:
            role = ctx.message.role_mentions[0]
            guild_store.set_res_role(ctx.guild.id, role.id)
            await ctx.send(f"**{role.name}** can now use `{COMMAND_PREFIX}res add`/`remove`.")
        else:
            guild_store.set_res_role(ctx.guild.id, None)
            await ctx.send("Res role cleared - only Manage Server members can use res add/remove now.")
        return

    if action in ("add", "remove"):
        if not await _has_res_role(ctx):
            await ctx.send("You don't have the role allowed to use `res` in this server.")
            return
        target = ctx.message.mentions[0] if ctx.message.mentions else None
        if target is None:
            await ctx.send(f"Mention who this reservation is for, e.g. "
                            f"`{COMMAND_PREFIX}res add rayquaza @user`.")
            return
        text_without_mention = re.sub(r"<@!?\d+>", "", rest).strip()
        names = _parse_species_list(text_without_mention)
        if not names:
            await ctx.send("Give at least one Pokemon to reserve.")
            return
        if action == "add":
            added = guild_store.reserve_add(ctx.guild.id, target.id, names)
            await ctx.send(f"**Added reserves for** {', '.join(_display(n) for n in added)} "
                            f"**to** {target.mention}." if added else "Already reserved.")
        else:
            removed = guild_store.reserve_remove(ctx.guild.id, target.id, names)
            await ctx.send(f"**Removed reserves for** {', '.join(_display(n) for n in removed)} "
                            f"**from** {target.mention}." if removed else "None of those were reserved.")
        return

    if action == "clear":
        if not await _has_res_role(ctx):
            await ctx.send("You don't have the role allowed to use `res` in this server.")
            return
        n = guild_store.reserve_clear_guild(ctx.guild.id)
        await ctx.send(f"Cleared all reservations in this server ({n} removed)." if n
                        else "There were no reservations in this server.")
        return

    if action == "search":
        names = _parse_species_list(rest)
        if not names:
            await ctx.send(f"Usage: `{COMMAND_PREFIX}res search <pokemon> [pokemon2 ...]`")
            return
        lines = []
        for n in names:
            holders = guild_store.reserve_matches(ctx.guild.id, n)
            who = ", ".join(f"<@{uid}>" for uid in sorted(holders)) if holders else "nobody"
            lines.append(f"**{_display(n)}** - reserved by {who}")
        # allowed_mentions=none: show the names without actually pinging anyone
        await ctx.send("\n".join(lines)[:1990], allowed_mentions=discord.AllowedMentions.none())
        return

    if action in ("list", "", None):
        target = ctx.message.mentions[0] if ctx.message.mentions else ctx.author
        mine = guild_store.reserve_list(ctx.guild.id, target.id)
        await ctx.send(f"{target.mention} has no reserves." if not mine else
                        f"**{target.display_name}'s reserves:** " + ", ".join(_display(n) for n in mine))
        return

    await ctx.send(f"Usage: `{COMMAND_PREFIX}res add <pokemon> @user`, `{COMMAND_PREFIX}res remove <pokemon> @user`, "
                    f"`{COMMAND_PREFIX}res list [@user]`, `{COMMAND_PREFIX}res search <pokemon>`, "
                    f"`{COMMAND_PREFIX}res clear`, or `{COMMAND_PREFIX}res role @role`.")


@bot.command(name="sh")
async def shiny_hunt_cmd(ctx: commands.Context, *, species: Optional[str] = None):
    """
    s!sh <pokemon>  - shiny hunt one Pokemon at a time in this server (replaces any previous target)
    s!sh clear      - stop shiny hunting
    s!sh            - show your current target
    """
    if ctx.guild is None:
        await ctx.send("This only works in a server.")
        return
    if species is None:
        current = guild_store.shiny_get(ctx.guild.id, ctx.author.id)
        await ctx.send(f"You're shiny hunting **{_display(current)}**." if current
                        else "You're not shiny hunting anything right now.")
        return
    if species.strip().lower() == "clear":
        guild_store.shiny_set(ctx.guild.id, ctx.author.id, None)
        await ctx.send("Shiny hunt cleared.")
        return
    name = species.strip()
    guild_store.shiny_set(ctx.guild.id, ctx.author.id, name)
    await ctx.send(f"You are now shiny hunting **{_display(name)}**.")


@bot.command(name="help")
async def help_cmd(ctx: commands.Context):
    embed = discord.Embed(
        title="Poke-Glitch Commands",
        description=f"Prefix: `{COMMAND_PREFIX}`",
        color=discord.Color.blurple(),
    )
    embed.add_field(name="Everyone", value=(
        f"`{COMMAND_PREFIX}cl add/remove/list <pokemon>` - ping me when it spawns\n"
        f"`{COMMAND_PREFIX}cl clear` - empty my whole collection\n"
        f"`{COMMAND_PREFIX}sh <pokemon>` / `clear` - shiny hunt one Pokemon\n"
        f"`{COMMAND_PREFIX}res list [@user]` - view reserves\n"
        f"`{COMMAND_PREFIX}res search <pokemon>` - who has it reserved\n"
        f"`{COMMAND_PREFIX}ping` - bot latency/health\n"
        f"`{COMMAND_PREFIX}model` - which AI model is running\n"
        f"`{COMMAND_PREFIX}naming` - is spawn-naming on in this server?"
    ), inline=False)
    embed.add_field(name="Allowed res role / Manage Server", value=(
        f"`{COMMAND_PREFIX}res add <pokemon> @user` - reserve a species to someone "
        f"(silences their collection ping for it)\n"
        f"`{COMMAND_PREFIX}res remove <pokemon> @user`\n"
        f"`{COMMAND_PREFIX}res clear` - wipe every reservation in this server\n"
        f"`{COMMAND_PREFIX}res role @role` - set who's allowed to use res add/remove"
    ), inline=False)
    embed.add_field(name="Bot owner", value=(
        f"`{COMMAND_PREFIX}predict`, `{COMMAND_PREFIX}learn`, `{COMMAND_PREFIX}forget`, `{COMMAND_PREFIX}undo`, "
        f"`{COMMAND_PREFIX}checklist`, `{COMMAND_PREFIX}stats`, `{COMMAND_PREFIX}api`, `{COMMAND_PREFIX}reload`, "
        f"`{COMMAND_PREFIX}threshold`, `{COMMAND_PREFIX}naming on/off`, `{COMMAND_PREFIX}backfill`, "
        f"`{COMMAND_PREFIX}bulklearn`"
    ), inline=False)
    await ctx.send(embed=embed)


@bot.command(name="bestname")
@commands.is_owner()
async def bestname_cmd(ctx: commands.Context, *, species: str):
    """Debug: show exactly what pokedata resolves for one species (bypasses the spawn path)."""
    display_name = _display(species.strip().lower().replace(" ", "_"))
    pdata = await pokedata.get_species_data(species, display_name, timeout=5.0)
    await ctx.send(
        f"species=`{species}` slug=`{pokedata._slugify(species)}` -> "
        f"types=`{pdata.get('types')}` best_name=`{pdata.get('best_name')}`"
    )


@bot.command(name="pokedataclear")
@commands.is_owner()
async def pokedata_clear_cmd(ctx: commands.Context, *, species: Optional[str] = None):
    """Debug: drop one species (or the whole cache) from pokedata_cache.json and re-fetch it clean."""
    if species:
        key = species.strip().lower()
        pokedata._cache.pop(key, None)
        pokedata._save_cache()
        await ctx.send(f"Cleared cached pokedata for `{species}`. It'll be re-fetched next time it's needed.")
    else:
        pokedata._cache.clear()
        pokedata._save_cache()
        await ctx.send("Cleared the entire pokedata cache.")


def _on_sigterm(signum, frame):
    raise SystemExit(0)


def main():
    atexit.register(_cache_save)
    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except (ValueError, OSError):
        pass
    bot.run(DISCORD_TOKEN, log_handler=None)


if __name__ == "__main__":
    main()
