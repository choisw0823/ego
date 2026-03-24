from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import sqlite3
from typing import Dict, Iterable, List, Optional, Tuple

from .srt_utils import parse_srt_file


SPEAKER_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_-]{1,30})\s*:\s*")
HOUR_FROM_CLIP_RE = re.compile(r"_(\d{2})\d{6}$")


@dataclass(slots=True)
class BuildPaths:
    egolife_dir: Path
    db_path: Path
    planner_memory_path: Path


class MemoryBuilder:
    def __init__(self, paths: BuildPaths):
        self.paths = paths
        self.fts_enabled = True

    def build(self, window_sec: int = 60, max_kappa_lines: int = 8) -> Dict[str, object]:
        self.paths.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.paths.planner_memory_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.paths.db_path) as conn:
            conn.row_factory = sqlite3.Row
            self._init_schema(conn)

            dense_count = self._ingest_source(
                conn,
                source_name="dense_caption",
                source_root=self.paths.egolife_dir / "DenseCaption",
            )
            transcript_count = self._ingest_source(
                conn,
                source_name="transcript",
                source_root=self.paths.egolife_dir / "Transcript",
            )

            if self.fts_enabled:
                conn.execute("INSERT INTO events_fts(rowid, text) SELECT event_id, text FROM events")

            descriptor_count = self._build_descriptors(
                conn,
                window_sec=window_sec,
                max_kappa_lines=max_kappa_lines,
            )

            if self.fts_enabled:
                conn.execute(
                    "INSERT INTO descriptors_fts(rowid, kappa_text) "
                    "SELECT descriptor_id, kappa_text FROM descriptors"
                )

            self._write_meta(conn, "fts_enabled", "1" if self.fts_enabled else "0")
            self._write_meta(conn, "window_sec", str(window_sec))

            planner_memory = self._build_planner_memory(conn)

            summary = {
                "db_path": str(self.paths.db_path),
                "planner_memory_path": str(self.paths.planner_memory_path),
                "fts_enabled": self.fts_enabled,
                "event_rows_dense": dense_count,
                "event_rows_transcript": transcript_count,
                "event_rows_total": dense_count + transcript_count,
                "descriptor_rows": descriptor_count,
                "actors": planner_memory.get("actors", []),
                "days": planner_memory.get("days", []),
            }

        self.paths.planner_memory_path.write_text(
            json.dumps(planner_memory, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            DROP TABLE IF EXISTS events;
            DROP TABLE IF EXISTS descriptors;
            DROP TABLE IF EXISTS meta;
            DROP TABLE IF EXISTS events_fts;
            DROP TABLE IF EXISTS descriptors_fts;

            CREATE TABLE events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                actor TEXT NOT NULL,
                day TEXT NOT NULL,
                clip_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                start_sec REAL NOT NULL,
                end_sec REAL NOT NULL,
                duration REAL NOT NULL,
                text TEXT NOT NULL,
                speaker TEXT,
                hour INTEGER
            );

            CREATE TABLE descriptors (
                descriptor_id INTEGER PRIMARY KEY AUTOINCREMENT,
                actor TEXT NOT NULL,
                day TEXT NOT NULL,
                clip_id TEXT NOT NULL,
                window_start REAL NOT NULL,
                window_end REAL NOT NULL,
                hour INTEGER,
                event_count INTEGER NOT NULL,
                primary_source TEXT NOT NULL,
                kappa_text TEXT NOT NULL
            );

            CREATE TABLE meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE INDEX idx_events_actor_day_clip ON events(actor, day, clip_id);
            CREATE INDEX idx_events_hour ON events(hour);
            CREATE INDEX idx_events_source ON events(source);
            CREATE INDEX idx_desc_actor_day_clip ON descriptors(actor, day, clip_id);
            CREATE INDEX idx_desc_hour ON descriptors(hour);
            """
        )

        try:
            conn.execute(
                "CREATE VIRTUAL TABLE events_fts USING fts5(text, content='events', content_rowid='event_id')"
            )
            conn.execute(
                "CREATE VIRTUAL TABLE descriptors_fts USING fts5(kappa_text, content='descriptors', content_rowid='descriptor_id')"
            )
            self.fts_enabled = True
        except sqlite3.OperationalError:
            self.fts_enabled = False

    def _write_meta(self, conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            (key, value),
        )

    def _ingest_source(
        self,
        conn: sqlite3.Connection,
        source_name: str,
        source_root: Path,
        batch_size: int = 5000,
    ) -> int:
        if not source_root.exists():
            return 0

        rows: List[Tuple] = []
        total = 0

        files = sorted(source_root.glob("**/*.srt"))
        for file_path in files:
            rel = file_path.relative_to(source_root)
            if len(rel.parts) < 3:
                continue

            actor = rel.parts[0]
            day = rel.parts[1]
            clip_id = file_path.stem
            hour = self._extract_hour(clip_id)

            entries = parse_srt_file(file_path)
            for e in entries:
                duration = max(0.0, e.end_sec - e.start_sec)
                speaker = self._extract_speaker(e.text)
                rows.append(
                    (
                        source_name,
                        actor,
                        day,
                        clip_id,
                        str(file_path),
                        e.start_sec,
                        e.end_sec,
                        duration,
                        e.text,
                        speaker,
                        hour,
                    )
                )
                total += 1

                if len(rows) >= batch_size:
                    self._insert_event_batch(conn, rows)
                    rows.clear()

        if rows:
            self._insert_event_batch(conn, rows)

        return total

    def _insert_event_batch(self, conn: sqlite3.Connection, rows: List[Tuple]) -> None:
        conn.executemany(
            """
            INSERT INTO events(
                source, actor, day, clip_id, file_path,
                start_sec, end_sec, duration, text, speaker, hour
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def _build_descriptors(
        self,
        conn: sqlite3.Connection,
        window_sec: int,
        max_kappa_lines: int,
        batch_size: int = 2000,
    ) -> int:
        clip_rows = conn.execute(
            "SELECT DISTINCT actor, day, clip_id, hour FROM events ORDER BY actor, day, clip_id"
        ).fetchall()

        to_insert: List[Tuple] = []
        total = 0

        for clip in clip_rows:
            actor = clip["actor"]
            day = clip["day"]
            clip_id = clip["clip_id"]
            hour = clip["hour"]

            events = conn.execute(
                """
                SELECT source, start_sec, end_sec, text
                FROM events
                WHERE actor = ? AND day = ? AND clip_id = ?
                ORDER BY start_sec, end_sec
                """,
                (actor, day, clip_id),
            ).fetchall()
            if not events:
                continue

            buckets: Dict[int, Dict[str, object]] = {}
            for ev in events:
                b = int(float(ev["start_sec"]) // window_sec)
                cur = buckets.setdefault(
                    b,
                    {
                        "start": float(ev["start_sec"]),
                        "end": float(ev["end_sec"]),
                        "dense": [],
                        "transcript": [],
                        "event_count": 0,
                    },
                )
                cur["start"] = min(float(cur["start"]), float(ev["start_sec"]))
                cur["end"] = max(float(cur["end"]), float(ev["end_sec"]))
                cur["event_count"] = int(cur["event_count"]) + 1
                text = str(ev["text"]).strip()
                if text:
                    if ev["source"] == "dense_caption":
                        cast_list = cur["dense"]
                    else:
                        cast_list = cur["transcript"]
                    cast_list.append(text)

            for b in sorted(buckets):
                item = buckets[b]
                dense_texts = self._unique_keep_order(item["dense"])
                trans_texts = self._unique_keep_order(item["transcript"])
                merged = (dense_texts + trans_texts)[:max_kappa_lines]
                if not merged:
                    continue

                kappa_text = " ; ".join(merged)
                primary_source = "dense_caption" if dense_texts else "transcript"
                to_insert.append(
                    (
                        actor,
                        day,
                        clip_id,
                        float(item["start"]),
                        float(item["end"]),
                        hour,
                        int(item["event_count"]),
                        primary_source,
                        kappa_text,
                    )
                )
                total += 1

                if len(to_insert) >= batch_size:
                    self._insert_descriptor_batch(conn, to_insert)
                    to_insert.clear()

        if to_insert:
            self._insert_descriptor_batch(conn, to_insert)

        return total

    def _insert_descriptor_batch(self, conn: sqlite3.Connection, rows: List[Tuple]) -> None:
        conn.executemany(
            """
            INSERT INTO descriptors(
                actor, day, clip_id, window_start, window_end,
                hour, event_count, primary_source, kappa_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def _build_planner_memory(self, conn: sqlite3.Connection) -> Dict[str, object]:
        actors = [r[0] for r in conn.execute("SELECT DISTINCT actor FROM events ORDER BY actor").fetchall()]
        days = [r[0] for r in conn.execute("SELECT DISTINCT day FROM events ORDER BY day").fetchall()]

        top_dense_actions = [
            {"text": r[0], "count": int(r[1])}
            for r in conn.execute(
                """
                SELECT text, COUNT(*) AS c
                FROM events
                WHERE source = 'dense_caption'
                GROUP BY text
                ORDER BY c DESC
                LIMIT 200
                """
            ).fetchall()
        ]

        actor_top_dense_actions: Dict[str, List[Dict[str, object]]] = {}
        for actor in actors:
            rows = conn.execute(
                """
                SELECT text, COUNT(*) AS c
                FROM events
                WHERE source = 'dense_caption' AND actor = ?
                GROUP BY text
                ORDER BY c DESC
                LIMIT 60
                """,
                (actor,),
            ).fetchall()
            actor_top_dense_actions[actor] = [
                {"text": r[0], "count": int(r[1])} for r in rows
            ]

        actor_hour_histogram: Dict[str, Dict[str, int]] = {}
        for actor in actors:
            rows = conn.execute(
                """
                SELECT hour, COUNT(*) AS c
                FROM events
                WHERE actor = ? AND hour IS NOT NULL
                GROUP BY hour
                ORDER BY hour
                """,
                (actor,),
            ).fetchall()
            actor_hour_histogram[actor] = {
                str(int(r[0])): int(r[1]) for r in rows if r[0] is not None
            }

        total_events = int(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])
        total_descriptors = int(conn.execute("SELECT COUNT(*) FROM descriptors").fetchone()[0])

        actor_name_map = {}
        for actor in actors:
            parts = actor.split("_", 1)
            if len(parts) == 2:
                actor_name_map[parts[1].lower()] = actor

        return {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "stats": {
                "total_events": total_events,
                "total_descriptors": total_descriptors,
            },
            "actors": actors,
            "days": days,
            "actor_name_map": actor_name_map,
            "top_dense_actions": top_dense_actions,
            "actor_top_dense_actions": actor_top_dense_actions,
            "actor_hour_histogram": actor_hour_histogram,
        }

    @staticmethod
    def _extract_speaker(text: str) -> Optional[str]:
        for part in text.split("|"):
            m = SPEAKER_RE.match(part.strip())
            if m:
                return m.group(1)
        return None

    @staticmethod
    def _extract_hour(clip_id: str) -> Optional[int]:
        m = HOUR_FROM_CLIP_RE.search(clip_id)
        if not m:
            return None
        try:
            h = int(m.group(1))
            if 0 <= h <= 23:
                return h
        except ValueError:
            return None
        return None

    @staticmethod
    def _unique_keep_order(items: Iterable[str]) -> List[str]:
        seen = set()
        out = []
        for x in items:
            y = str(x).strip()
            if not y or y in seen:
                continue
            seen.add(y)
            out.append(y)
        return out
