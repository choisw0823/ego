from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Dict, List, Sequence, Tuple



def _to_hms(sec: float) -> str:
    s = int(max(0, sec))
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    return f"{h:02d}:{m:02d}:{ss:02d}"


class EvidenceRetriever:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def retrieve(self, blueprint: Dict[str, object], top_k: int = 5) -> List[Dict[str, object]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            fts_enabled = self._fts_enabled(conn)
            desc_rows = self._query_descriptors(conn, blueprint, top_k, fts_enabled)
            out: List[Dict[str, object]] = []
            for row in desc_rows:
                events = self._fetch_supporting_events(conn, row)
                out.append(
                    {
                        "descriptor_id": int(row["descriptor_id"]),
                        "actor": row["actor"],
                        "day": row["day"],
                        "clip_id": row["clip_id"],
                        "window_start": float(row["window_start"]),
                        "window_end": float(row["window_end"]),
                        "window_range": f"{_to_hms(float(row['window_start']))}-{_to_hms(float(row['window_end']))}",
                        "hour": row["hour"],
                        "event_count": int(row["event_count"]),
                        "primary_source": row["primary_source"],
                        "kappa_text": row["kappa_text"],
                        "score": float(row["score"]),
                        "video_pointer": f"{row['clip_id']} {_to_hms(float(row['window_start']))}-{_to_hms(float(row['window_end']))}",
                        "events": events,
                    }
                )
            return out

    def _fts_enabled(self, conn: sqlite3.Connection) -> bool:
        row = conn.execute("SELECT value FROM meta WHERE key = 'fts_enabled'").fetchone()
        return bool(row and row[0] == "1")

    def _query_descriptors(
        self,
        conn: sqlite3.Connection,
        blueprint: Dict[str, object],
        top_k: int,
        fts_enabled: bool,
    ) -> List[sqlite3.Row]:
        actor = blueprint.get("actor")
        day = blueprint.get("day")
        hour_range = blueprint.get("hour_range")
        search_terms = blueprint.get("search_terms") or []

        filters = []
        params: List[object] = []

        if actor:
            filters.append("d.actor = ?")
            params.append(actor)
        if day:
            filters.append("d.day = ?")
            params.append(day)

        hour_filter_sql, hour_params = self._build_hour_filter(hour_range)
        if hour_filter_sql:
            filters.append(hour_filter_sql)
            params.extend(hour_params)

        where_clause = ""
        if filters:
            where_clause = " AND " + " AND ".join(filters)

        if search_terms:
            if fts_enabled:
                fts_query = self._to_fts_query(search_terms)
                if fts_query:
                    sql = (
                        "SELECT d.*, bm25(descriptors_fts) AS score "
                        "FROM descriptors_fts "
                        "JOIN descriptors d ON d.descriptor_id = descriptors_fts.rowid "
                        "WHERE descriptors_fts MATCH ?"
                        f"{where_clause} "
                        "ORDER BY score ASC "
                        "LIMIT ?"
                    )
                    rows = conn.execute(sql, [fts_query, *params, top_k]).fetchall()
                    if rows:
                        return rows

            like_parts = []
            like_params: List[object] = []
            for t in search_terms:
                t = str(t).strip()
                if not t:
                    continue
                like_parts.append("d.kappa_text LIKE ?")
                like_params.append(f"%{t}%")

            if like_parts:
                sql = (
                    "SELECT d.*, 999.0 AS score "
                    "FROM descriptors d "
                    "WHERE (" + " OR ".join(like_parts) + ")"
                    f"{where_clause} "
                    "ORDER BY d.event_count DESC "
                    "LIMIT ?"
                )
                rows = conn.execute(sql, [*like_params, *params, top_k]).fetchall()
                if rows:
                    return rows

        # Always fallback to filter-only retrieval if text match is empty.
        sql = (
            "SELECT d.*, 1000.0 AS score "
            "FROM descriptors d "
            "WHERE 1 = 1"
            f"{where_clause} "
            "ORDER BY d.event_count DESC "
            "LIMIT ?"
        )
        return conn.execute(sql, [*params, top_k]).fetchall()

    def _build_hour_filter(self, hour_range: object) -> Tuple[str, List[object]]:
        if not hour_range or not isinstance(hour_range, Sequence) or len(hour_range) != 2:
            return "", []

        start, end = hour_range
        try:
            s = int(start)
            e = int(end)
        except (TypeError, ValueError):
            return "", []

        s = max(0, min(23, s))
        e = max(0, min(24, e))

        if s < e:
            return "(d.hour >= ? AND d.hour < ?)", [s, e]
        if s > e:
            return "(d.hour >= ? OR d.hour < ?)", [s, e]
        return "", []

    def _to_fts_query(self, terms: Sequence[object]) -> str:
        escaped = []
        for t in terms:
            x = str(t).replace('"', "").strip()
            if not x:
                continue
            escaped.append(f'"{x}"')
        return " OR ".join(escaped) if escaped else ""

    def _fetch_supporting_events(self, conn: sqlite3.Connection, desc_row: sqlite3.Row) -> List[Dict[str, object]]:
        rows = conn.execute(
            """
            SELECT source, start_sec, end_sec, text
            FROM events
            WHERE actor = ?
              AND day = ?
              AND clip_id = ?
              AND start_sec <= ?
              AND end_sec >= ?
            ORDER BY start_sec ASC
            LIMIT 20
            """,
            (
                desc_row["actor"],
                desc_row["day"],
                desc_row["clip_id"],
                float(desc_row["window_end"]),
                float(desc_row["window_start"]),
            ),
        ).fetchall()

        out = []
        for r in rows:
            out.append(
                {
                    "source": r["source"],
                    "start": _to_hms(float(r["start_sec"])),
                    "end": _to_hms(float(r["end_sec"])),
                    "text": r["text"],
                }
            )
        return out
