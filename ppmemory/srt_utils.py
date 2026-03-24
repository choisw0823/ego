from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import List


TIMESTAMP_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})"
)


@dataclass(slots=True)
class SRTEntry:
    index: int
    start_sec: float
    end_sec: float
    text: str



def _to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0



def parse_srt_file(path: Path) -> List[SRTEntry]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if not raw:
        return []

    blocks = re.split(r"\n\s*\n", raw)
    out: List[SRTEntry] = []

    for block in blocks:
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue

        ts_idx = None
        ts_match = None
        for i, ln in enumerate(lines[:3]):
            m = TIMESTAMP_RE.search(ln)
            if m:
                ts_idx = i
                ts_match = m
                break

        if ts_idx is None or ts_match is None:
            continue

        start = _to_seconds(*ts_match.groups()[:4])
        end = _to_seconds(*ts_match.groups()[4:])

        text_lines = lines[ts_idx + 1 :]
        text = " | ".join(text_lines).strip()
        if not text:
            continue

        idx = len(out) + 1
        out.append(SRTEntry(index=idx, start_sec=start, end_sec=end, text=text))

    return out
