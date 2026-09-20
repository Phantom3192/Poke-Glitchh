"""
discord_bot.py - Watches for wild Pokemon spawn messages (e.g. from Poketwo)
and replies with the species name predicted by OUR model + DB.

Mechanics note: this uses a completely different identification method than
a typical image-hash matcher bot. Instead of comparing perceptual hashes,
it embeds the spawn image into a 256-dim vector with the trained
EfficientNet-B0 feature extractor (predict.py / train_model.py) and finds
the closest match by cosine similarity against every stored training
embedding in the DB. The two approaches aren't directly comparable number
-for-number (a hash "distance" and a cosine "similarity" mean different
things), so the CONFIDENCE_THRESHOLD below needs its own tuning based on
what scores you actually observe once this is running against real spawns.

Setup:
    1. pip install -r requirements-bot.txt
    2. Copy .env.example to .env and fill in DISCORD_TOKEN (and DB creds,
       same as train_model.py: TURSO_URL / TURSO_AUTH_TOKEN or DB_PATH)
    3. Make sure models/pokemon_classifier.pt is your real trained model
    4. python discord_bot.py

Commands (owner-only, prefix configurable via COMMAND_PREFIX, default "s!"):
    s!predict     - attach an image, or reply to a message with one, to test
                    identification manually without needing a live spawn
    s!stats       - show how many species/features are currently loaded
    s!reload      - re-load the model + feature bank from disk/DB without
                    restarting the bot (use after retraining)
    s!threshold X - view/change the confidence threshold at runtime
"""

import os
import io
import sys
import time
import asyncio
import logging
from typing import Optional

import aiohttp
import discord
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

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix=COMMAND_PREFIX, intents=intents)

# ── Model + feature bank (loaded once, refreshed via s!reload) ─────────────
_extractor = None
_species_list = None
_matrix = None
_db: Optional[Database] = None
_session: Optional[aiohttp.ClientSession] = None


def _load_model_and_bank():
    global _extractor, _species_list, _matrix, _db
    log.info(f"📦 Loading model from {BOT_MODEL_PATH} ...")
    _extractor = load_extractor(BOT_MODEL_PATH)

    log.info("🗄️  Connecting to database and loading feature bank ...")
    if _db is not None:
        _db.close()
    _db = Database()
    _species_list, _matrix = load_feature_bank(_db)
    log.info(f"   Loaded {_matrix.shape[0]} feature vectors across {len(set(_species_list))} species")


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


def _run_identification(image_bytes: bytes):
    """Runs in a worker thread — torch inference + numpy matmul are blocking/CPU-bound."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    query_vec = _extractor.extract(img)
    return predict_species(query_vec, _species_list, _matrix, top_k=TOP_K)


@bot.event
async def on_ready():
    await _get_session()
    log.info("=" * 60)
    log.info(f"🤖 Logged in as {bot.user} — scanning for spawns from bot ID {SPAWN_BOT_ID}")
    log.info(f"   Confidence threshold: {CONFIDENCE_THRESHOLD}")
    log.info("=" * 60)


@bot.event
async def on_message(message: discord.Message):
    await bot.process_commands(message)

    if message.author.id != SPAWN_BOT_ID:
        return

    image_url = _extract_wild_spawn_image_url(message)
    if not image_url:
        return

    start = time.time()
    image_bytes = await _download(image_url)
    if not image_bytes:
        return

    try:
        winner, score, neighbors = await asyncio.to_thread(_run_identification, image_bytes)
    except Exception as e:
        log.error(f"Inference failed for spawn message {message.id}: {e}")
        return

    elapsed_ms = (time.time() - start) * 1000
    confident = score >= CONFIDENCE_THRESHOLD
    log.info(f"Spawn {message.id}: {winner} (score {score:.3f}, confident={confident}, {elapsed_ms:.0f}ms)")

    if not confident and SILENT_BELOW_THRESHOLD:
        return

    display_name = winner.replace("_", " ").title()
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
    winner, score, neighbors = await asyncio.to_thread(_run_identification, image_bytes)
    elapsed_ms = (time.time() - start) * 1000

    lines = [f"**Best guess:** {winner.replace('_', ' ').title()} — {score * 100:.1f}%"]
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


@bot.command(name="stats")
@commands.is_owner()
async def stats_cmd(ctx: commands.Context):
    await ctx.send(
        f"📊 Loaded **{_matrix.shape[0]}** feature vectors across "
        f"**{len(set(_species_list))}** species from the DB.\n"
        f"Model: `{BOT_MODEL_PATH}`\n"
        f"Confidence threshold: `{CONFIDENCE_THRESHOLD}`"
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
    await ctx.send(f"✅ Reloaded — {_matrix.shape[0]} features across {len(set(_species_list))} species.")


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
