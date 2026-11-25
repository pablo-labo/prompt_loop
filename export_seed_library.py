#!/usr/bin/env python3
"""Export anonymous tag library, cluster stats, and pair synergy priors from local SQLite to seed_library.json.

Never exports personal batches, generated prompts, image paths, or external import records.
"""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB = BASE_DIR / "data" / "loop.db"
DEFAULT_OUT = BASE_DIR / "seed_library.json"


def _round_num(val: Any, ndigits: int = 6) -> float:
    return round(float(val or 0.0), ndigits)


def export_seed_library(db_path: Path, out_path: Path) -> dict[str, Any]:
    if not db_path.is_file():
        raise FileNotFoundError(f"Database not found: {db_path}")

    uri = f"file:{db_path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        meta_rows = {
            str(r["key"]): str(r["value"])
            for r in conn.execute("SELECT key, value FROM meta_state").fetchall()
        }
        meta = {
            "round_num": str(meta_rows.get("round_num", "0")),
            "reward_ema": str(meta_rows.get("reward_ema", "0.0")),
        }

        clusters: list[dict[str, Any]] = []
        for r in conn.execute(
            "SELECT cluster, slot, label, count, reward_sum, weight FROM cluster_stats ORDER BY slot, cluster"
        ).fetchall():
            clusters.append(
                {
                    "cluster": str(r["cluster"]),
                    "slot": str(r["slot"]),
                    "label": str(r["label"]),
                    "count": int(r["count"] or 0),
                    "reward_sum": _round_num(r["reward_sum"]),
                    "weight": _round_num(r["weight"]),
                }
            )

        tags: list[dict[str, Any]] = []
        valid_tags: set[str] = set()
        for r in conn.execute(
            "SELECT * FROM tag_stats ORDER BY stage, slot, tag"
        ).fetchall():
            t_name = str(r["tag"]).strip()
            if not t_name or t_name.startswith("wip_test_"):
                continue
            valid_tags.add(t_name)
            tags.append(
                {
                    "tag": t_name,
                    "slot": str(r["slot"]),
                    "cluster": str(r["cluster"]),
                    "stage": str(r["stage"]),
                    "active_count": int(r["active_count"] or 0),
                    "active_reward_sum": _round_num(r["active_reward_sum"]),
                    "observed_count": int(r["observed_count"] or 0),
                    "observed_eff_count": _round_num(r["observed_eff_count"]),
                    "observed_gain_sum": _round_num(r["observed_gain_sum"]),
                    "exec_hits": int(r["exec_hits"] or 0),
                    "exec_trials": int(r["exec_trials"] or 0),
                    "rating_count": int(r["rating_count"] or 0),
                    "rating_eff_count": _round_num(r["rating_eff_count"]),
                    "rating_sum": _round_num(r["rating_sum"]),
                    "mean_rating": _round_num(r["mean_rating"]),
                    "weight": _round_num(r["weight"]),
                    "z_score": _round_num(r["z_score"]),
                    "last_used_round": 0,
                    "consecutive_uses": 0,
                }
            )

        pairs: list[dict[str, Any]] = []
        for r in conn.execute(
            "SELECT * FROM tag_pair_stats ORDER BY co_count DESC, tag_a, tag_b"
        ).fetchall():
            ta = str(r["tag_a"]).strip()
            tb = str(r["tag_b"]).strip()
            if ta not in valid_tags or tb not in valid_tags:
                continue
            pairs.append(
                {
                    "tag_a": ta,
                    "tag_b": tb,
                    "co_count": int(r["co_count"] or 0),
                    "eff_count": _round_num(r["eff_count"]),
                    "rating_sum": _round_num(r["rating_sum"]),
                    "synergy_sum": _round_num(r["synergy_sum"]),
                    "mean_rating": _round_num(r["mean_rating"]),
                    "synergy": _round_num(r["synergy"]),
                    "affinity": _round_num(r["affinity"]),
                }
            )
    finally:
        conn.close()

    summary = {
        "tag_count": len(tags),
        "pair_count": len(pairs),
        "cluster_count": len(clusters),
    }

    def _fmt_lines(items: list[dict[str, Any]]) -> str:
        if not items:
            return "[]"
        inner = ",\n".join(
            "    " + json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            for item in items
        )
        return "[\n" + inner + "\n  ]"

    out_text = (
        "{\n"
        f'  "schema_version": 1,\n'
        f'  "summary": {json.dumps(summary, ensure_ascii=False)},\n'
        f'  "meta": {json.dumps(meta, ensure_ascii=False)},\n'
        f'  "clusters": {_fmt_lines(clusters)},\n'
        f'  "tags": {_fmt_lines(tags)},\n'
        f'  "pairs": {_fmt_lines(pairs)}\n'
        "}\n"
    )
    parsed = json.loads(out_text)
    out_path.write_text(out_text, encoding="utf-8")
    return parsed["summary"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export local tag library to seed_library.json")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Source SQLite loop.db path")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output seed_library.json path")
    args = parser.parse_args()
    summary = export_seed_library(args.db, args.out)
    size_kb = round(args.out.stat().st_size / 1024.0, 1)
    print(f"Exported {summary} -> {args.out} ({size_kb} KB)")


if __name__ == "__main__":
    main()
