"""
guild_store.py - per-guild ping lists, persisted in Turso (libSQL).

  collection[guild][user]  -> species this user wants pinged for
  reserve[guild][user]     -> species reserved to this user (a species in
                              ANYONE's reserve in this guild suppresses the
                              collection ping for it - reserve wins)
  shiny[guild][user]       -> the single species this user is shiny hunting
  res_role[guild]          -> role ID allowed to use s!res add/remove (None =
                              falls back to "Manage Server" permission)

Everything is loaded into memory at startup (so lookups on every spawn are
instant) and every change is written through to Turso in the background.

If TURSO_DATABASE_URL / TURSO_AUTH_TOKEN aren't set, it falls back to the old
local file (guild_data.json) so local dev still works.

First run with Turso enabled: if the Turso tables are empty and a
guild_data.json exists, it's imported automatically (one time only).

Species keys are normalized (see _key) so "Mr. Mime", "mr mime" and
"MR-MIME" all collide correctly.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from turso_db import db, Stmt

log = logging.getLogger("guild_store")

STORE_PATH = Path(os.getenv("GUILD_STORE_FILE", "guild_data.json"))
_MIGRATED_FLAG = "migrated:guild_data.json"


def _key(species: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", species.lower())


def _new_guild() -> dict:
    return {"collection": {}, "reserve": {}, "shiny": {}, "res_role": None}


class GuildStore:
    def __init__(self, path: Path = STORE_PATH):
        self.path = path
        self.data: Dict[str, dict] = {}
        self.load()

    # ---------------- persistence ----------------
    def load(self) -> None:
        if db.enabled:
            self._load_turso()
        else:
            self._load_file()

    def _load_file(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except FileNotFoundError:
            self.data = {}
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"guild_store: couldn't read {self.path} ({e}), starting empty")
            self.data = {}

    def _load_turso(self) -> None:
        self.data = {}
        for table in ("collection", "reserve"):
            for gid, uid, species in db.query(
                f"SELECT guild_id, user_id, species FROM {table} ORDER BY rowid"
            ):
                self._guild_raw(gid)[table].setdefault(uid, []).append(species)
        for gid, uid, species in db.query("SELECT guild_id, user_id, species FROM shiny"):
            self._guild_raw(gid)["shiny"][uid] = species
        for gid, role in db.query("SELECT guild_id, res_role FROM guild_settings"):
            self._guild_raw(gid)["res_role"] = int(role) if role else None

        if not self.data and db.kv_get(_MIGRATED_FLAG) is None:
            self._migrate_from_file()
        log.info(f"guild_store: loaded {len(self.data)} guild(s) from Turso")

    def _migrate_from_file(self) -> None:
        if not self.path.exists():
            db.kv_set(_MIGRATED_FLAG, "1")
            return
        self._load_file()
        stmts: List[Stmt] = []
        for gid, g in self.data.items():
            for table in ("collection", "reserve"):
                for uid, species_list in g.get(table, {}).items():
                    for sp in species_list:
                        stmts.append(self._add_stmt(table, gid, uid, sp))
            for uid, sp in g.get("shiny", {}).items():
                stmts.append(self._shiny_stmt(gid, uid, sp))
            if g.get("res_role") is not None:
                stmts.append(self._role_stmt(gid, g["res_role"]))
        for i in range(0, len(stmts), 200):
            db.execute(stmts[i:i + 200])
        db.kv_set(_MIGRATED_FLAG, "1")
        log.info(f"guild_store: imported {self.path} into Turso ({len(stmts)} rows). "
                 f"The file is no longer used and can be deleted.")

    def save(self) -> None:
        """File fallback only. With Turso, writes happen per-change via _persist()."""
        if db.enabled:
            return
        try:
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f)
            tmp.replace(self.path)
        except OSError as e:
            log.warning(f"guild_store: couldn't save {self.path}: {e}")

    def _persist(self, stmts: List[Stmt]) -> None:
        if db.enabled:
            db.submit(stmts)
        else:
            self.save()

    # ---------------- statement builders ----------------
    @staticmethod
    def _add_stmt(table: str, gid, uid, species: str) -> Stmt:
        return (f"INSERT OR IGNORE INTO {table}(guild_id, user_id, species_key, species) "
                f"VALUES(?, ?, ?, ?)", [str(gid), str(uid), _key(species), species])

    @staticmethod
    def _del_stmt(table: str, gid, uid, species: str) -> Stmt:
        return (f"DELETE FROM {table} WHERE guild_id = ? AND user_id = ? AND species_key = ?",
                [str(gid), str(uid), _key(species)])

    @staticmethod
    def _shiny_stmt(gid, uid, species: str) -> Stmt:
        return ("INSERT INTO shiny(guild_id, user_id, species) VALUES(?, ?, ?) "
                "ON CONFLICT(guild_id, user_id) DO UPDATE SET species = excluded.species",
                [str(gid), str(uid), species])

    @staticmethod
    def _role_stmt(gid, role_id: Optional[int]) -> Stmt:
        return ("INSERT INTO guild_settings(guild_id, res_role) VALUES(?, ?) "
                "ON CONFLICT(guild_id) DO UPDATE SET res_role = excluded.res_role",
                [str(gid), None if role_id is None else str(role_id)])

    # ---------------- internals ----------------
    def _guild_raw(self, gid: str) -> dict:
        g = self.data.setdefault(str(gid), _new_guild())
        for field in ("collection", "reserve", "shiny"):
            g.setdefault(field, {})
        g.setdefault("res_role", None)
        return g

    def _guild(self, guild_id: int) -> dict:
        return self._guild_raw(str(guild_id))

    def _list_add(self, table: str, guild_id: int, user_id: int, species: list) -> list:
        bucket = self._guild(guild_id)[table].setdefault(str(user_id), [])
        existing = {_key(s) for s in bucket}
        added = []
        for sp in species:
            if _key(sp) not in existing:
                bucket.append(sp)
                existing.add(_key(sp))
                added.append(sp)
        if added:
            self._persist([self._add_stmt(table, guild_id, user_id, sp) for sp in added])
        return added

    def _list_remove(self, table: str, guild_id: int, user_id: int, species: list) -> list:
        g = self._guild(guild_id)
        bucket = g[table].get(str(user_id), [])
        targets = {_key(s) for s in species}
        removed = [s for s in bucket if _key(s) in targets]
        g[table][str(user_id)] = [s for s in bucket if _key(s) not in targets]
        if removed:
            self._persist([self._del_stmt(table, guild_id, user_id, sp) for sp in removed])
        return removed

    def _list_get(self, table: str, guild_id: int, user_id: int) -> list:
        return list(self._guild(guild_id)[table].get(str(user_id), []))

    def _list_matches(self, table: str, guild_id: int, species: str) -> Set[int]:
        k = _key(species)
        return {int(uid) for uid, sp_list in self._guild(guild_id)[table].items()
                if any(_key(s) == k for s in sp_list)}

    # ---------------- collection ----------------
    def collection_add(self, guild_id: int, user_id: int, species: list) -> list:
        return self._list_add("collection", guild_id, user_id, species)

    def collection_remove(self, guild_id: int, user_id: int, species: list) -> list:
        return self._list_remove("collection", guild_id, user_id, species)

    def collection_list(self, guild_id: int, user_id: int) -> list:
        return self._list_get("collection", guild_id, user_id)

    def collection_matches(self, guild_id: int, species: str) -> Set[int]:
        """User IDs (in this guild) whose collection contains this species."""
        return self._list_matches("collection", guild_id, species)

    # ---------------- reserve ----------------
    def reserve_add(self, guild_id: int, user_id: int, species: list) -> list:
        return self._list_add("reserve", guild_id, user_id, species)

    def reserve_remove(self, guild_id: int, user_id: int, species: list) -> list:
        return self._list_remove("reserve", guild_id, user_id, species)

    def reserve_list(self, guild_id: int, user_id: int) -> list:
        return self._list_get("reserve", guild_id, user_id)

    def reserve_matches(self, guild_id: int, species: str) -> Set[int]:
        return self._list_matches("reserve", guild_id, species)

    # ---------------- reserve role ----------------
    def set_res_role(self, guild_id: int, role_id: Optional[int]) -> None:
        self._guild(guild_id)["res_role"] = role_id
        self._persist([self._role_stmt(guild_id, role_id)])

    def get_res_role(self, guild_id: int) -> Optional[int]:
        return self._guild(guild_id).get("res_role")

    # ---------------- shiny hunt (one species per user) ----------------
    def shiny_set(self, guild_id: int, user_id: int, species: Optional[str]) -> None:
        g = self._guild(guild_id)
        if species is None:
            g["shiny"].pop(str(user_id), None)
            self._persist([("DELETE FROM shiny WHERE guild_id = ? AND user_id = ?",
                            [str(guild_id), str(user_id)])])
        else:
            g["shiny"][str(user_id)] = species
            self._persist([self._shiny_stmt(guild_id, user_id, species)])

    def shiny_get(self, guild_id: int, user_id: int) -> Optional[str]:
        return self._guild(guild_id)["shiny"].get(str(user_id))

    def shiny_matches(self, guild_id: int, species: str) -> Set[int]:
        k = _key(species)
        return {int(uid) for uid, sp in self._guild(guild_id)["shiny"].items() if _key(sp) == k}


store = GuildStore()
