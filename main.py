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
    s!reload             - re-load the model + feature bank from disk/DB
    s!threshold X        - view/change the confidence threshold at runtime

Optional auto-learning (AUTO_LEARN=true): when the spawn bot announces who
caught what ("... You caught a Level 12 Pikachu!"), the bot learns from the
last spawn in that channel if it guessed wrong or wasn't confident.
"""

import os
import io
import re
import sys
import json
import time
import uuid
import asyncio
import difflib
import logging
import threading
from typing import Dict, Optional, Tuple

import aiohttp
import discord
import numpy as np
from discord.ext import commands
from PIL import Image
from dotenv import load_dotenv

from predict import load_extractor, load_feature_bank, predict_species
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

# ── Auto-learning (opt-in) ───────────────────────────────────────────────
AUTO_LEARN = os.getenv("AUTO_LEARN", "false").lower() == "true"
# Must capture the species name in group 1. Default matches Poketwo-style
# "... You caught a Level 12 Pikachu! (20.1% IV)". Check your spawn bot's
# real catch message and adjust via the CATCH_REGEX env var if needed.
CATCH_REGEX = os.getenv("CATCH_REGEX", r"caught an? (?:level \d+ )?(.+?)\s*(?:!|\()")
AUTO_LEARN_MAX_AGE = int(os.getenv("AUTO_LEARN_MAX_AGE", "600"))  # seconds
_CATCH_RE = re.compile(CATCH_REGEX, re.IGNORECASE)

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


def _load_model_and_bank():
    global _extractor, _species_list, _matrix, _db
    log.info(f"📦 Loading model from {BOT_MODEL_PATH} ...")
    extractor = load_extractor(BOT_MODEL_PATH)

    log.info("🗄️  Connecting to database and loading feature bank ...")
    new_db = Database()
    species_list, matrix = load_feature_bank(new_db)

    with _bank_lock:
        old_db = _db
        _extractor, _db, _species_list, _matrix = extractor, new_db, species_list, matrix
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
    query_vec = _embed_image_bytes(image_bytes)
    winner, score, neighbors = _predict(query_vec, species_list, matrix)
    return winner, score, neighbors, query_vec


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


# ── Discord events ───────────────────────────────────────────────────────

@bot.event
async def on_ready():
    await _get_session()
    log.info("=" * 60)
    log.info(f"🤖 Logged in as {bot.user} — scanning for spawns from bot ID {SPAWN_BOT_ID}")
    log.info(f"   Confidence threshold: {CONFIDENCE_THRESHOLD}")
    log.info(f"   Auto-learn: {'ON' if AUTO_LEARN else 'OFF'}")
    log.info("=" * 60)


@bot.event
async def on_message(message: discord.Message):
    await bot.process_commands(message)

    if message.author.id != SPAWN_BOT_ID:
        return

    if AUTO_LEARN:
        try:
            await _maybe_auto_learn(message)
        except Exception as e:
            log.error(f"Auto-learn failed: {type(e).__name__}: {e}")

    image_url = _extract_wild_spawn_image_url(message)
    if not image_url:
        return

    # A new spawn replaces the old one, so forget the previous spawn now; it's
    # only re-set below if we manage to analyse this one (avoids learning a
    # catch against the wrong image).
    _last_spawn[message.channel.id] = None

    start = time.time()
    image_bytes = await _download(image_url)
    if not image_bytes:
        return

    try:
        winner, score, neighbors, query_vec = await asyncio.to_thread(_run_identification, image_bytes)
    except Exception as e:
        log.error(f"Inference failed for spawn message {message.id}: {e}")
        return

    _last_spawn[message.channel.id] = (time.time(), winner, score, query_vec)

    elapsed_ms = (time.time() - start) * 1000
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
    winner, score, neighbors, _ = await asyncio.to_thread(_run_identification, image_bytes)
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


@bot.command(name="stats")
@commands.is_owner()
async def stats_cmd(ctx: commands.Context):
    _, species_list, matrix = _snapshot()
    learned = await asyncio.to_thread(_count_learned)
    await ctx.send(
        f"📊 Loaded **{matrix.shape[0]}** feature vectors across "
        f"**{len(set(species_list))}** species from the DB "
        f"(**{learned}** taught by you).\n"
        f"Model: `{BOT_MODEL_PATH}`\n"
        f"Confidence threshold: `{CONFIDENCE_THRESHOLD}` · Auto-learn: `{'on' if AUTO_LEARN else 'off'}`"
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
    bot.run(DISCORD_TOKEN, log_handler=None)  # we configured logging above (stdout)


if __name__ == "__main__":
    main()
