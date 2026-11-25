import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from tag_classifier import SLOT_ORDER


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class LoopStorage:
    """SQLite persistence layer for prompt-loop batches, generations, tag weights, and clusters."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=15.0,
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self.init_db()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def init_db(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=10000")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS batches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    prompt_mode TEXT NOT NULL DEFAULT 'auto',
                    progress_text TEXT NOT NULL DEFAULT '',
                    error_message TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS generations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id INTEGER NOT NULL,
                    slot_index INTEGER NOT NULL,
                    seed INTEGER NOT NULL,
                    is_dropout_round INTEGER NOT NULL DEFAULT 0,
                    positive_prompt TEXT NOT NULL,
                    negative_prompt TEXT NOT NULL,
                    sampled_tags_json TEXT NOT NULL,
                    detected_tags_json TEXT NOT NULL DEFAULT '[]',
                    emergent_tags_json TEXT NOT NULL DEFAULT '[]',
                    ratings_json TEXT NOT NULL DEFAULT '[]',
                    image_rel_path TEXT NOT NULL DEFAULT '',
                    comfy_prompt_id TEXT NOT NULL DEFAULT '',
                    gen_elapsed_ms REAL NOT NULL DEFAULT 0.0,
                    tag_elapsed_ms REAL NOT NULL DEFAULT 0.0,
                    user_rating INTEGER,
                    is_corrupted INTEGER NOT NULL DEFAULT 0,
                    weight_applied INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(batch_id) REFERENCES batches(id)
                );

                CREATE INDEX IF NOT EXISTS idx_generations_batch_id
                    ON generations(batch_id);

                CREATE TABLE IF NOT EXISTS tag_stats (
                    tag TEXT PRIMARY KEY,
                    slot TEXT NOT NULL,
                    cluster TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    active_count INTEGER NOT NULL DEFAULT 0,
                    active_reward_sum REAL NOT NULL DEFAULT 0.0,
                    observed_count INTEGER NOT NULL DEFAULT 0,
                    observed_gain_sum REAL NOT NULL DEFAULT 0.0,
                    exec_hits INTEGER NOT NULL DEFAULT 0,
                    exec_trials INTEGER NOT NULL DEFAULT 0,
                    weight REAL NOT NULL DEFAULT 0.0,
                    z_score REAL NOT NULL DEFAULT 0.0,
                    last_used_round INTEGER NOT NULL DEFAULT 0,
                    consecutive_uses INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS cluster_stats (
                    cluster TEXT PRIMARY KEY,
                    slot TEXT NOT NULL,
                    label TEXT NOT NULL,
                    count INTEGER NOT NULL DEFAULT 0,
                    reward_sum REAL NOT NULL DEFAULT 0.0,
                    weight REAL NOT NULL DEFAULT 0.0
                );

                CREATE TABLE IF NOT EXISTS meta_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS external_imports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    filename TEXT NOT NULL DEFAULT '',
                    image_rel_path TEXT NOT NULL DEFAULT '',
                    rating INTEGER NOT NULL,
                    direct_explore INTEGER NOT NULL DEFAULT 0,
                    tag_elapsed_ms REAL NOT NULL DEFAULT 0.0,
                    detected_tags_json TEXT NOT NULL DEFAULT '[]',
                    applied_tags_json TEXT NOT NULL DEFAULT '[]',
                    ratings_json TEXT NOT NULL DEFAULT '[]',
                    summary_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS manual_generations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    progress_text TEXT NOT NULL DEFAULT '',
                    error_message TEXT NOT NULL DEFAULT '',
                    positive_prompt TEXT NOT NULL,
                    negative_prompt TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    steps INTEGER NOT NULL,
                    cfg REAL NOT NULL,
                    loras_json TEXT NOT NULL DEFAULT '[]',
                    image_rel_path TEXT NOT NULL DEFAULT '',
                    comfy_prompt_id TEXT NOT NULL DEFAULT '',
                    gen_elapsed_ms REAL NOT NULL DEFAULT 0.0,
                    tag_elapsed_ms REAL NOT NULL DEFAULT 0.0,
                    detected_tags_json TEXT NOT NULL DEFAULT '[]'
                );

                CREATE TABLE IF NOT EXISTS tag_pair_stats (
                    tag_a TEXT NOT NULL,
                    tag_b TEXT NOT NULL,
                    co_count INTEGER NOT NULL DEFAULT 0,
                    eff_count REAL NOT NULL DEFAULT 0.0,
                    rating_sum REAL NOT NULL DEFAULT 0.0,
                    synergy_sum REAL NOT NULL DEFAULT 0.0,
                    mean_rating REAL NOT NULL DEFAULT 0.0,
                    synergy REAL NOT NULL DEFAULT 0.0,
                    affinity REAL NOT NULL DEFAULT 0.0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (tag_a, tag_b)
                );

                CREATE INDEX IF NOT EXISTS idx_tag_pair_affinity
                    ON tag_pair_stats(affinity DESC);
                """
            )
            batch_cols = {
                row["name"]
                for row in self._conn.execute("PRAGMA table_info(batches)").fetchall()
            }
            if "prompt_mode" not in batch_cols:
                self._conn.execute(
                    "ALTER TABLE batches ADD COLUMN prompt_mode TEXT NOT NULL DEFAULT 'auto'"
                )
            tag_cols = {
                row["name"]
                for row in self._conn.execute("PRAGMA table_info(tag_stats)").fetchall()
            }
            newly_added_obs_eff = "observed_eff_count" not in tag_cols
            for col_name, col_def in (
                ("observed_eff_count", "REAL NOT NULL DEFAULT 0.0"),
                ("rating_count", "INTEGER NOT NULL DEFAULT 0"),
                ("rating_eff_count", "REAL NOT NULL DEFAULT 0.0"),
                ("rating_sum", "REAL NOT NULL DEFAULT 0.0"),
                ("mean_rating", "REAL NOT NULL DEFAULT 0.0"),
            ):
                if col_name not in tag_cols:
                    self._conn.execute(
                        f"ALTER TABLE tag_stats ADD COLUMN {col_name} {col_def}"
                    )
            if newly_added_obs_eff:
                self._conn.execute(
                    "UPDATE tag_stats SET observed_eff_count = CAST(observed_count AS REAL) "
                    "WHERE observed_eff_count = 0.0 AND observed_count > 0"
                )
            self._conn.execute(
                "INSERT OR IGNORE INTO meta_state(key, value) VALUES ('round_num', '0')"
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO meta_state(key, value) VALUES ('reward_ema', '0.0')"
            )

    def get_meta(self, key: str, default: str = "") -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM meta_state WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else default

    def set_meta(self, key: str, value: str, conn: sqlite3.Connection | None = None) -> None:
        if conn is not None:
            conn.execute(
                "INSERT OR REPLACE INTO meta_state(key, value) VALUES (?, ?)",
                (key, str(value)),
            )
            return
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO meta_state(key, value) VALUES (?, ?)",
                (key, str(value)),
            )

    def seed_initial_tags_if_empty(
        self,
        clusters: list[dict[str, str]],
        tags: list[dict[str, str]],
    ) -> bool:
        with self.transaction() as conn:
            cnt = conn.execute("SELECT COUNT(*) AS c FROM tag_stats").fetchone()["c"]
            if cnt > 0:
                return False
            now = utc_now_iso()
            for cl in clusters:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO cluster_stats(cluster, slot, label, count, reward_sum, weight)
                    VALUES (?, ?, ?, 0, 0.0, 0.0)
                    """,
                    (cl["cluster"], cl["slot"], cl["label"]),
                )
            for t in tags:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO tag_stats(
                        tag, slot, cluster, stage,
                        active_count, active_reward_sum,
                        observed_count, observed_gain_sum,
                        exec_hits, exec_trials,
                        weight, z_score, last_used_round, consecutive_uses, updated_at
                    ) VALUES (?, ?, ?, ?, 0, 0.0, 0, 0.0, 0, 0, 0.0, 0.0, 0, 0, ?)
                    """,
                    (t["tag"], t["slot"], t["cluster"], t.get("stage", "core"), now),
                )
            return True

    def create_batch(
        self,
        mode: str,
        progress_text: str = "",
        prompt_mode: str = "auto",
    ) -> int:
        now = utc_now_iso()
        with self.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO batches(created_at, updated_at, status, mode, prompt_mode, progress_text, error_message)
                VALUES (?, ?, 'generating', ?, ?, ?, '')
                """,
                (now, now, mode, prompt_mode, progress_text),
            )
            return int(cur.lastrowid)

    def update_batch(
        self,
        batch_id: int,
        status: str | None = None,
        progress_text: str | None = None,
        error_message: str | None = None,
    ) -> None:
        fields: list[str] = ["updated_at = ?"]
        params: list[Any] = [utc_now_iso()]
        if status is not None:
            fields.append("status = ?")
            params.append(status)
        if progress_text is not None:
            fields.append("progress_text = ?")
            params.append(progress_text)
        if error_message is not None:
            fields.append("error_message = ?")
            params.append(error_message)
        params.append(batch_id)
        with self.transaction() as conn:
            conn.execute(
                f"UPDATE batches SET {', '.join(fields)} WHERE id = ?",
                params,
            )

    def add_generation(
        self,
        batch_id: int,
        slot_index: int,
        seed: int,
        is_dropout_round: bool,
        positive_prompt: str,
        negative_prompt: str,
        sampled_tags: list[dict[str, Any]],
        detected_tags: list[dict[str, Any]],
        emergent_tags: list[dict[str, Any]],
        ratings: list[dict[str, Any]],
        image_rel_path: str,
        comfy_prompt_id: str,
        gen_elapsed_ms: float,
        tag_elapsed_ms: float,
    ) -> int:
        now = utc_now_iso()
        with self.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO generations(
                    batch_id, slot_index, seed, is_dropout_round,
                    positive_prompt, negative_prompt,
                    sampled_tags_json, detected_tags_json, emergent_tags_json, ratings_json,
                    image_rel_path, comfy_prompt_id, gen_elapsed_ms, tag_elapsed_ms,
                    user_rating, is_corrupted, weight_applied, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, 0, ?)
                """,
                (
                    batch_id,
                    slot_index,
                    seed,
                    1 if is_dropout_round else 0,
                    positive_prompt,
                    negative_prompt,
                    json.dumps(sampled_tags, ensure_ascii=False),
                    json.dumps(detected_tags, ensure_ascii=False),
                    json.dumps(emergent_tags, ensure_ascii=False),
                    json.dumps(ratings, ensure_ascii=False),
                    image_rel_path,
                    comfy_prompt_id,
                    float(gen_elapsed_ms),
                    float(tag_elapsed_ms),
                    now,
                ),
            )
            return int(cur.lastrowid)

    @staticmethod
    def _row_to_generation(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "batch_id": row["batch_id"],
            "slot_index": row["slot_index"],
            "seed": row["seed"],
            "is_dropout_round": bool(row["is_dropout_round"]),
            "positive_prompt": row["positive_prompt"],
            "negative_prompt": row["negative_prompt"],
            "sampled_tags": json.loads(row["sampled_tags_json"] or "[]"),
            "detected_tags": json.loads(row["detected_tags_json"] or "[]"),
            "emergent_tags": json.loads(row["emergent_tags_json"] or "[]"),
            "ratings": json.loads(row["ratings_json"] or "[]"),
            "image_rel_path": row["image_rel_path"],
            "comfy_prompt_id": row["comfy_prompt_id"],
            "gen_elapsed_ms": row["gen_elapsed_ms"],
            "tag_elapsed_ms": row["tag_elapsed_ms"],
            "user_rating": row["user_rating"],
            "is_corrupted": bool(row["is_corrupted"]),
            "weight_applied": bool(row["weight_applied"]),
            "created_at": row["created_at"],
        }

    def get_batch(self, batch_id: int) -> dict[str, Any] | None:
        with self._lock:
            b_row = self._conn.execute(
                "SELECT * FROM batches WHERE id = ?", (batch_id,)
            ).fetchone()
            if b_row is None:
                return None
            gen_rows = self._conn.execute(
                "SELECT * FROM generations WHERE batch_id = ? ORDER BY slot_index ASC, id ASC",
                (batch_id,),
            ).fetchall()
            return {
                "id": b_row["id"],
                "created_at": b_row["created_at"],
                "updated_at": b_row["updated_at"],
                "status": b_row["status"],
                "mode": b_row["mode"],
                "prompt_mode": b_row["prompt_mode"] if "prompt_mode" in b_row.keys() else "auto",
                "progress_text": b_row["progress_text"],
                "error_message": b_row["error_message"],
                "generations": [self._row_to_generation(r) for r in gen_rows],
            }

    def get_latest_batch(self) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM batches ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            return self.get_batch(int(row["id"]))

    def list_batches(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._lock:
            b_rows = self._conn.execute(
                "SELECT * FROM batches ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            results: list[dict[str, Any]] = []
            for b in b_rows:
                gens = self._conn.execute(
                    """
                    SELECT id, slot_index, seed, is_dropout_round, image_rel_path,
                           user_rating, is_corrupted, weight_applied, gen_elapsed_ms, tag_elapsed_ms
                    FROM generations
                    WHERE batch_id = ?
                    ORDER BY slot_index ASC, id ASC
                    """,
                    (b["id"],),
                ).fetchall()
                results.append(
                    {
                        "id": b["id"],
                        "created_at": b["created_at"],
                        "updated_at": b["updated_at"],
                        "status": b["status"],
                        "mode": b["mode"],
                        "prompt_mode": b["prompt_mode"] if "prompt_mode" in b.keys() else "auto",
                        "progress_text": b["progress_text"],
                        "error_message": b["error_message"],
                        "generations": [
                            {
                                "id": g["id"],
                                "slot_index": g["slot_index"],
                                "seed": g["seed"],
                                "is_dropout_round": bool(g["is_dropout_round"]),
                                "image_rel_path": g["image_rel_path"],
                                "user_rating": g["user_rating"],
                                "is_corrupted": bool(g["is_corrupted"]),
                                "weight_applied": bool(g["weight_applied"]),
                                "gen_elapsed_ms": g["gen_elapsed_ms"],
                                "tag_elapsed_ms": g["tag_elapsed_ms"],
                            }
                            for g in gens
                        ],
                    }
                )
            return results

    def has_active_batch(self) -> int | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM batches WHERE status = 'generating' ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return int(row["id"]) if row else None

    def get_all_tags(
        self,
        stage: str | None = None,
        slot: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if stage:
            clauses.append("stage = ?")
            params.append(stage)
        if slot:
            clauses.append("slot = ?")
            params.append(slot)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM tag_stats {where} ORDER BY z_score DESC, weight DESC, tag ASC"

        if conn is not None:
            rows = conn.execute(sql, params).fetchall()
        else:
            with self._lock:
                rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_tag(
        self, tag: str, conn: sqlite3.Connection | None = None
    ) -> dict[str, Any] | None:
        if conn is not None:
            row = conn.execute("SELECT * FROM tag_stats WHERE tag = ?", (tag,)).fetchone()
            return dict(row) if row else None
        with self._lock:
            row = self._conn.execute("SELECT * FROM tag_stats WHERE tag = ?", (tag,)).fetchone()
            return dict(row) if row else None

    def get_all_clusters(
        self, conn: sqlite3.Connection | None = None
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM cluster_stats ORDER BY weight DESC, cluster ASC"
        if conn is not None:
            rows = conn.execute(sql).fetchall()
        else:
            with self._lock:
                rows = self._conn.execute(sql).fetchall()
        return [dict(r) for r in rows]

    def ensure_clusters(
        self,
        clusters: list[dict[str, str]],
        conn: sqlite3.Connection | None = None,
    ) -> None:
        sql = (
            "INSERT OR IGNORE INTO cluster_stats(cluster, slot, label, count, reward_sum, weight) "
            "VALUES (?, ?, ?, 0, 0.0, 0.0)"
        )
        if conn is not None:
            for cl in clusters:
                conn.execute(sql, (cl["cluster"], cl["slot"], cl["label"]))
            return
        with self.transaction() as tx:
            for cl in clusters:
                tx.execute(sql, (cl["cluster"], cl["slot"], cl["label"]))

    def get_stage_slot_summary(
        self, conn: sqlite3.Connection | None = None
    ) -> dict[str, dict[str, int]]:
        sql = "SELECT slot, stage, COUNT(*) AS cnt FROM tag_stats GROUP BY slot, stage"
        if conn is not None:
            rows = conn.execute(sql).fetchall()
        else:
            with self._lock:
                rows = self._conn.execute(sql).fetchall()
        summary: dict[str, dict[str, int]] = {
            s: {"core": 0, "explore": 0, "staging": 0, "negative": 0, "ignored": 0, "total": 0}
            for s in SLOT_ORDER
        }
        for r in rows:
            sl = str(r["slot"])
            st = str(r["stage"])
            cnt = int(r["cnt"])
            bucket = summary.setdefault(
                sl,
                {"core": 0, "explore": 0, "staging": 0, "negative": 0, "ignored": 0, "total": 0},
            )
            bucket[st] = bucket.get(st, 0) + cnt
            bucket["total"] += cnt
        return summary

    def backup_database(self, backup_dir: Path, label: str = "backup") -> Path:
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dst_path = backup_dir / f"loop_{label}_{stamp}.db"
        with self._lock:
            dst_conn = sqlite3.connect(str(dst_path))
            try:
                self._conn.backup(dst_conn)
            finally:
                dst_conn.close()
        return dst_path

    def add_external_import(
        self,
        filename: str,
        image_rel_path: str,
        rating: int,
        direct_explore: bool,
        tag_elapsed_ms: float,
        detected_tags: list[dict[str, Any]],
        applied_tags: list[dict[str, Any]],
        ratings: list[dict[str, Any]],
        summary: dict[str, Any],
    ) -> int:
        now = utc_now_iso()
        with self.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO external_imports(
                    created_at, filename, image_rel_path, rating, direct_explore,
                    tag_elapsed_ms, detected_tags_json, applied_tags_json,
                    ratings_json, summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    filename,
                    image_rel_path,
                    int(rating),
                    1 if direct_explore else 0,
                    float(tag_elapsed_ms),
                    json.dumps(detected_tags, ensure_ascii=False),
                    json.dumps(applied_tags, ensure_ascii=False),
                    json.dumps(ratings, ensure_ascii=False),
                    json.dumps(summary, ensure_ascii=False),
                ),
            )
            return int(cur.lastrowid)

    def list_external_imports(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM external_imports ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "id": r["id"],
                    "created_at": r["created_at"],
                    "filename": r["filename"],
                    "image_rel_path": r["image_rel_path"],
                    "rating": r["rating"],
                    "direct_explore": bool(r["direct_explore"]),
                    "tag_elapsed_ms": r["tag_elapsed_ms"],
                    "detected_tags": json.loads(r["detected_tags_json"] or "[]"),
                    "applied_tags": json.loads(r["applied_tags_json"] or "[]"),
                    "ratings": json.loads(r["ratings_json"] or "[]"),
                    "summary": json.loads(r["summary_json"] or "{}"),
                }
            )
        return out

    def create_manual_generation(
        self,
        mode: str,
        positive_prompt: str,
        negative_prompt: str,
        seed: int,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        loras: list[dict[str, Any]],
        progress_text: str = "正在提交生图任务...",
    ) -> int:
        now = utc_now_iso()
        with self.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO manual_generations(
                    created_at, status, mode, progress_text, error_message,
                    positive_prompt, negative_prompt, seed, width, height, steps, cfg,
                    loras_json
                ) VALUES (?, 'generating', ?, ?, '', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    mode,
                    progress_text,
                    positive_prompt,
                    negative_prompt,
                    int(seed),
                    int(width),
                    int(height),
                    int(steps),
                    float(cfg),
                    json.dumps(loras, ensure_ascii=False),
                ),
            )
            return int(cur.lastrowid)

    def update_manual_generation(
        self,
        gen_id: int,
        status: str | None = None,
        progress_text: str | None = None,
        error_message: str | None = None,
        image_rel_path: str | None = None,
        comfy_prompt_id: str | None = None,
        gen_elapsed_ms: float | None = None,
        tag_elapsed_ms: float | None = None,
        detected_tags: list[dict[str, Any]] | None = None,
    ) -> None:
        fields: list[str] = []
        params: list[Any] = []
        if status is not None:
            fields.append("status = ?")
            params.append(status)
        if progress_text is not None:
            fields.append("progress_text = ?")
            params.append(progress_text)
        if error_message is not None:
            fields.append("error_message = ?")
            params.append(error_message)
        if image_rel_path is not None:
            fields.append("image_rel_path = ?")
            params.append(image_rel_path)
        if comfy_prompt_id is not None:
            fields.append("comfy_prompt_id = ?")
            params.append(comfy_prompt_id)
        if gen_elapsed_ms is not None:
            fields.append("gen_elapsed_ms = ?")
            params.append(float(gen_elapsed_ms))
        if tag_elapsed_ms is not None:
            fields.append("tag_elapsed_ms = ?")
            params.append(float(tag_elapsed_ms))
        if detected_tags is not None:
            fields.append("detected_tags_json = ?")
            params.append(json.dumps(detected_tags, ensure_ascii=False))
        if not fields:
            return
        params.append(int(gen_id))
        with self.transaction() as conn:
            conn.execute(
                f"UPDATE manual_generations SET {', '.join(fields)} WHERE id = ?",
                params,
            )

    @staticmethod
    def _row_to_manual_gen(r: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": r["id"],
            "created_at": r["created_at"],
            "status": r["status"],
            "mode": r["mode"],
            "progress_text": r["progress_text"],
            "error_message": r["error_message"],
            "positive_prompt": r["positive_prompt"],
            "negative_prompt": r["negative_prompt"],
            "seed": r["seed"],
            "width": r["width"],
            "height": r["height"],
            "steps": r["steps"],
            "cfg": r["cfg"],
            "loras": json.loads(r["loras_json"] or "[]"),
            "image_rel_path": r["image_rel_path"],
            "comfy_prompt_id": r["comfy_prompt_id"],
            "gen_elapsed_ms": r["gen_elapsed_ms"],
            "tag_elapsed_ms": r["tag_elapsed_ms"],
            "detected_tags": json.loads(r["detected_tags_json"] or "[]"),
        }

    def get_manual_generation(self, gen_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM manual_generations WHERE id = ?", (int(gen_id),)
            ).fetchone()
            return self._row_to_manual_gen(row) if row else None

    def list_manual_generations(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM manual_generations ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_manual_gen(r) for r in rows]

    def has_active_manual_generation(self) -> int | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM manual_generations WHERE status = 'generating' ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return int(row["id"]) if row else None

    def get_pair_count(self, conn: sqlite3.Connection | None = None) -> int:
        sql = "SELECT COUNT(*) AS c FROM tag_pair_stats"
        if conn is not None:
            row = conn.execute(sql).fetchone()
        else:
            with self._lock:
                row = self._conn.execute(sql).fetchone()
        return int(row["c"]) if row else 0

    def get_all_pairs(
        self,
        min_co_count: int = 1,
        conn: sqlite3.Connection | None = None,
    ) -> list[dict[str, Any]]:
        sql = (
            "SELECT * FROM tag_pair_stats WHERE co_count >= ? "
            "ORDER BY affinity DESC, co_count DESC, tag_a ASC, tag_b ASC"
        )
        params = (max(1, int(min_co_count)),)
        if conn is not None:
            rows = conn.execute(sql, params).fetchall()
        else:
            with self._lock:
                rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_rated_generations_chronological(
        self, conn: sqlite3.Connection | None = None
    ) -> list[dict[str, Any]]:
        sql = (
            "SELECT id, batch_id, user_rating, sampled_tags_json, detected_tags_json, emergent_tags_json "
            "FROM generations "
            "WHERE weight_applied = 1 AND is_corrupted = 0 AND user_rating IS NOT NULL "
            "ORDER BY id ASC"
        )
        if conn is not None:
            rows = conn.execute(sql).fetchall()
        else:
            with self._lock:
                rows = self._conn.execute(sql).fetchall()
        return [
            {
                "id": int(r["id"]),
                "batch_id": int(r["batch_id"]),
                "user_rating": int(r["user_rating"]),
                "sampled_tags": json.loads(r["sampled_tags_json"] or "[]"),
                "detected_tags": json.loads(r["detected_tags_json"] or "[]"),
                "emergent_tags": json.loads(r["emergent_tags_json"] or "[]"),
            }
            for r in rows
        ]

    def get_external_imports_chronological(
        self, conn: sqlite3.Connection | None = None
    ) -> list[dict[str, Any]]:
        sql = "SELECT id, rating, applied_tags_json FROM external_imports ORDER BY id ASC"
        if conn is not None:
            rows = conn.execute(sql).fetchall()
        else:
            with self._lock:
                rows = self._conn.execute(sql).fetchall()
        return [
            {
                "id": int(r["id"]),
                "rating": int(r["rating"]),
                "applied_tags": json.loads(r["applied_tags_json"] or "[]"),
            }
            for r in rows
        ]
