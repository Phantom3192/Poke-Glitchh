"""
pokedata.py - Two small, independent lookups used to decorate a spawn reply:

  1. get_type_emojis(species)  -> list[str] of your server's custom type emojis
     (edit TYPE_EMOJIS below with the emoji IDs from Developer Portal > Emojis).

  2. get_best_name(species)    -> the shortest plain a-z name PokeAPI has on file
     for that species across every language it's localized in (English, the
     official Japanese romanization, German, French, Spanish, Italian, Korean
     romanized, ...). This is genuinely useful because a lot of species have a
     much shorter "easy" name in some other language than their English name -
     e.g. Shellder's official Japanese romanization is "Sheruda" - and it's
     always a real, Nintendo-sanctioned localization, never a made-up string.

Both are looked up once per species ever, then cached to disk
(pokedata_cache.json) so a restart doesn't re-hit the network. Network calls
happen lazily, off the hot spawn path is not blocking: main.py should call
ensure_species(species) as a fire-and-forget task the moment a spawn is
identified, and only use whatever is already cached when building that
spawn's reply (so the very first sighting of a species has no emoji/best name
yet, but every sighting after that does, forever).
"""

import asyncio
import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp

log = logging.getLogger("pokedata")

POKEAPI_BASE = os.getenv("POKEAPI_BASE", "https://pokeapi.co/api/v2")
CACHE_PATH = Path(os.getenv("POKEDATA_CACHE_FILE", "pokedata_cache.json"))

# ---------------------------------------------------------------------------
# 1) Type emojis
#
# From your Developer Portal > Applications > Poke-Glitch > Emojis page:
# right-click (or long-press) each emoji there and "Copy ID", then paste it
# in below in the form "<:Name:ID>". I've filled in the 8 you showed me -
# add the remaining 10 the same way (fire, water, grass, electric, ice,
# fighting, poison, ground, rock, steel). Anything left as None just won't
# show an emoji for that type - it won't error out.
# ---------------------------------------------------------------------------
TYPE_EMOJIS: Dict[str, Optional[str]] = {
    "normal": "<:normal:1553515794090958898>",
    "fire": "<:fire:1553515678827282564>",
    "water": "<:water:1553515888521642085>",
    "electric": "<:electric:1553515075904602192>",
    "grass": "<:grass:1553515282524283050>",
    "ice": "<:ice:1553515738269229187>",
    "fighting": "<:fighting:1553515634766385192>",
    "poison": "<:poison:1553515829776228402>",
    "ground": "<:ground:1553515337624985670>",
    "flying": "<:flying:1553515168359780352>",
    "psychic": "<:psychic:1553515394684293160>",
    "bug": "<:bug:1553514984317657098>",
    "rock": "<:rock:1553515433829859441>",
    "ghost": "<:ghost:1553515222327627807>",
    "dragon": "<:dragon:1553515545469390950>",
    "dark": "<:dark:1553515035979022346>",
    "steel": "<:steel:1553515886382678097>",
    "fairy": "<:fairy:1553515593812934826>",
}


# Standard competitive-Pokemon type colors, used to color the spawn embed by
# the species' primary type (falls back to a neutral color if type is unknown/not cached yet).
TYPE_COLORS: Dict[str, int] = {
    "normal": 0xA8A77A, "fire": 0xEE8130, "water": 0x6390F0, "electric": 0xF7D02C,
    "grass": 0x7AC74C, "ice": 0x96D9D6, "fighting": 0xC22E28, "poison": 0xA33EA1,
    "ground": 0xE2BF65, "flying": 0xA98FF3, "psychic": 0xF95587, "bug": 0xA6B91A,
    "rock": 0xB6A136, "ghost": 0x735797, "dragon": 0x6F35FC, "dark": 0x705746,
    "steel": 0xB7B7CE, "fairy": 0xD685AD,
}

# ---------------------------------------------------------------------------
# Species-name -> PokeAPI slug. PokeAPI wants lowercase, hyphenated, no
# accents/symbols. This covers the common special cases; anything else just
# gets lowercased/hyphenated automatically, which is right most of the time.
# ---------------------------------------------------------------------------
_SLUG_OVERRIDES = {
    "nidoranf": "nidoran-f", "nidoran♀": "nidoran-f",
    "nidoranm": "nidoran-m", "nidoran♂": "nidoran-m",
    "mr mime": "mr-mime", "mr. mime": "mr-mime",
    "mime jr": "mime-jr", "mime jr.": "mime-jr",
    "mr rime": "mr-rime", "mr. rime": "mr-rime",
    "farfetchd": "farfetchd", "farfetch'd": "farfetchd",
    "sirfetchd": "sirfetchd", "sirfetch'd": "sirfetchd",
    "type null": "type-null", "type: null": "type-null",
    "ho oh": "ho-oh", "porygon z": "porygon-z",
    "jangmo o": "jangmo-o", "hakamo o": "hakamo-o", "kommo o": "kommo-o",
    "tapu koko": "tapu-koko", "tapu lele": "tapu-lele",
    "tapu bulu": "tapu-bulu", "tapu fini": "tapu-fini",
}


def _slugify(species: str) -> str:
    key = species.strip().lower()
    if key in _SLUG_OVERRIDES:
        return _SLUG_OVERRIDES[key]
    key = unicodedata.normalize("NFKD", key).encode("ascii", "ignore").decode()
    key = re.sub(r"[.']", "", key)
    key = re.sub(r"[^a-z0-9]+", "-", key).strip("-")
    return key


_ASCII_WORD_RE = re.compile(r"^[a-z]+$")


def _pick_best_name(display_name: str, api_names: List[dict]) -> str:
    """Shortest plain a-z candidate name across every language PokeAPI has;
    falls back to the species' own display name if nothing shorter/cleaner exists."""
    candidates = {display_name.replace(" ", "").lower(): display_name}
    for entry in api_names:
        raw = (entry.get("name") or "").strip()
        ascii_name = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode()
        ascii_name = ascii_name.replace(" ", "").replace("-", "")
        if ascii_name and _ASCII_WORD_RE.match(ascii_name.lower()):
            candidates.setdefault(ascii_name.lower(), ascii_name.capitalize())
    return min(candidates.values(), key=lambda n: (len(n), n.lower()))


# ---------------------------------------------------------------------------
# Cache (species-key -> {"types": [...], "best_name": "..."})
# ---------------------------------------------------------------------------
_cache: Dict[str, dict] = {}
_cache_lock = asyncio.Lock()
_inflight: Dict[str, asyncio.Task] = {}
_session: Optional[aiohttp.ClientSession] = None


def _load_cache() -> None:
    global _cache
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            _cache = json.load(f)
        log.info(f"pokedata: loaded {len(_cache)} cached species from {CACHE_PATH}")
    except FileNotFoundError:
        _cache = {}
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"pokedata: couldn't read {CACHE_PATH} ({e}), starting empty")
        _cache = {}


def _save_cache() -> None:
    try:
        tmp = CACHE_PATH.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_cache, f)
        tmp.replace(CACHE_PATH)
    except OSError as e:
        log.warning(f"pokedata: couldn't save {CACHE_PATH}: {e}")


async def _get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            headers={"User-Agent": "poke-glitch-bot/1.0"},
        )
    return _session


async def _fetch(key: str, species: str, display_name: str) -> dict:
    slug = _slugify(species)
    sess = await _get_session()
    types: List[str] = []
    best_name = display_name
    resolved_from = "fallback (display name)"
    try:
        async with sess.get(f"{POKEAPI_BASE}/pokemon/{slug}") as resp:
            if resp.status == 200:
                body = await resp.json()
                types = [t["type"]["name"] for t in body.get("types", [])]
            else:
                log.info(f"pokedata: GET /pokemon/{slug} -> HTTP {resp.status} for species {species!r}")
    except Exception as e:
        log.info(f"pokedata: type lookup failed for {species!r} ({slug}): {type(e).__name__}: {e}")

    try:
        async with sess.get(f"{POKEAPI_BASE}/pokemon-species/{slug}") as resp:
            if resp.status == 200:
                body = await resp.json()
                api_species_name = (body.get("name") or "").strip()
                if api_species_name and api_species_name != slug:
                    # PokeAPI resolved our slug to a DIFFERENT species (bad slug guess) -
                    # don't use its names, they'd belong to the wrong Pokemon.
                    log.warning(f"pokedata: slug {slug!r} for {species!r} resolved to "
                                f"{api_species_name!r} instead - ignoring its names")
                else:
                    best_name = _pick_best_name(display_name, body.get("names", []))
                    resolved_from = f"PokeAPI /pokemon-species/{slug}"
            else:
                log.info(f"pokedata: GET /pokemon-species/{slug} -> HTTP {resp.status} for species {species!r}")
    except Exception as e:
        log.info(f"pokedata: name lookup failed for {species!r} ({slug}): {type(e).__name__}: {e}")

    log.info(f"pokedata: {species!r} (slug={slug!r}) -> types={types} best_name={best_name!r} [{resolved_from}]")
    entry = {"types": types, "best_name": best_name}
    async with _cache_lock:
        _cache[key] = entry
        _save_cache()
    return entry


async def get_species_data(species: str, display_name: str, timeout: float = 2.0) -> dict:
    """
    The one function main.py should call. Returns {"types": [...], "best_name": "..."}
    for THIS species - either instantly from cache/an in-flight request for the exact
    same species, or a fresh fetch. Never reuses another species' data: every call is
    keyed strictly off this species' own normalized name, so two different Pokemon can
    never end up sharing a cached entry.

    Waits up to `timeout` seconds for a first-time lookup so the very first sighting of
    a species already gets its emoji/best name (not just every sighting after it). If
    PokeAPI is slow/unreachable, returns the safe fallback (no emoji, best_name =
    display_name) without blocking the spawn reply, while the fetch keeps running in
    the background to populate the cache for next time.
    """
    key = species.strip().lower()
    cached = _cache.get(key)
    if cached is not None:
        return cached

    task = _inflight.get(key)
    if task is None:
        task = asyncio.ensure_future(_fetch(key, species, display_name))
        _inflight[key] = task
        task.add_done_callback(lambda t, k=key: _inflight.pop(k, None))

    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    except asyncio.TimeoutError:
        return {"types": [], "best_name": display_name}


def emojis_for(pdata: dict) -> List[str]:
    return [e for t in pdata.get("types", []) if (e := TYPE_EMOJIS.get(t))]


def color_for(pdata: dict, fallback: int = 0x2B2D31) -> int:
    types = pdata.get("types") or []
    return TYPE_COLORS.get(types[0], fallback) if types else fallback


async def close() -> None:
    if _session and not _session.closed:
        await _session.close()


_load_cache()
