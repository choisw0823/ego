from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Dict, List, Optional, Tuple


DAY_RE = re.compile(r"(?:day\s*([1-7]))|(?:([1-7])일차)|(?:d([1-7]))", re.IGNORECASE)

TEMPORAL_HINTS = {
    "아침": (6, 11),
    "오전": (6, 12),
    "점심": (11, 14),
    "오후": (12, 18),
    "저녁": (17, 22),
    "밤": (21, 24),
    "새벽": (0, 6),
    "morning": (6, 12),
    "afternoon": (12, 18),
    "evening": (17, 22),
    "night": (21, 24),
}

ACTION_LEXICON = {
    "phone": ["手机", "폰", "휴대폰", "핸드폰", "스마트폰", "phone", "mobile"],
    "eat": ["吃", "밥", "식사", "먹", "meal", "lunch", "dinner"],
    "walk": ["走", "걷", "이동", "move", "route"],
    "talk": ["说话", "말", "대화", "talk", "speak"],
    "look": ["看", "보", "look", "watch"],
    "write": ["打字", "타이핑", "typing", "write"],
    "smile": ["笑", "웃", "smile"],
    "sit": ["坐", "앉", "sit"],
    "stand": ["站", "서", "stand"],
}

STOPWORDS = {
    "나", "내", "나는", "제가", "저", "좀", "그리고", "뭐", "뭐야", "뭐했지",
    "했어", "했지", "전에", "이후", "후", "어제", "오늘", "the", "a", "an", "is", "was", "were",
}

KOREAN_SUFFIXES = ["에서", "으로", "에게", "한테", "까지", "부터", "처럼", "보다"]
KOREAN_PARTICLES = set("가은는을를에와과도만로")


class SearchPlanner:
    def __init__(self, planner_memory_path: Path):
        self.planner_memory_path = planner_memory_path
        self.memory = json.loads(planner_memory_path.read_text(encoding="utf-8"))
        self.actors: List[str] = list(self.memory.get("actors", []))
        self.days: List[str] = list(self.memory.get("days", []))
        self.actor_aliases = self._build_actor_aliases()

    def plan(self, question: str, top_k: int = 5) -> Dict[str, object]:
        q = question.strip()
        q_lower = q.lower()

        actor = self._extract_actor(q_lower)
        day = self._extract_day(q_lower)
        hour_range = self._extract_time_range(q_lower)
        action_tags = self._extract_action_tags(q_lower)
        search_terms = self._extract_search_terms(q, action_tags)

        if any(tok in q_lower for tok in ["어제", "yesterday"]) and not day and self.days:
            day = sorted(self.days)[-1]

        if actor:
            short_name = actor.split("_", 1)[1] if "_" in actor else actor
            for name_term in [actor, short_name, short_name.lower()]:
                if name_term and name_term not in search_terms:
                    search_terms.insert(0, name_term)

        if day and day not in search_terms:
            search_terms.append(day)

        if not search_terms:
            if actor:
                actor_actions = self.memory.get("actor_top_dense_actions", {}).get(actor, [])
                search_terms = [x["text"] for x in actor_actions[:5] if isinstance(x, dict) and x.get("text")]
            else:
                top_actions = self.memory.get("top_dense_actions", [])
                search_terms = [x["text"] for x in top_actions[:5] if isinstance(x, dict) and x.get("text")]

        search_terms = self._uniq_keep_order(search_terms)[:12]

        return {
            "query": q,
            "actor": actor,
            "day": day,
            "hour_range": list(hour_range) if hour_range else None,
            "action_tags": action_tags,
            "search_terms": search_terms,
            "source_preference": ["dense_caption", "transcript"],
            "limit": max(1, top_k),
            "planner_notes": {
                "memory_on": True,
                "habit_token_used": False,
                "nightly_update_used": False,
            },
        }

    def _build_actor_aliases(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        name_map = self.memory.get("actor_name_map", {})
        for actor in self.actors:
            out[actor.lower()] = actor
            parts = actor.split("_", 1)
            if len(parts) == 2:
                out[parts[1].lower()] = actor
        for alias, actor in name_map.items():
            out[str(alias).lower()] = str(actor)
        return out

    def _extract_actor(self, q_lower: str) -> Optional[str]:
        for alias, actor in self.actor_aliases.items():
            if alias and alias in q_lower:
                return actor
        return None

    def _extract_day(self, q_lower: str) -> Optional[str]:
        m = DAY_RE.search(q_lower)
        if m:
            num = next((g for g in m.groups() if g), None)
            if num:
                return f"DAY{int(num)}"

        for d in self.days:
            if d.lower() in q_lower:
                return d
        return None

    def _extract_time_range(self, q_lower: str) -> Optional[Tuple[int, int]]:
        for hint, hr in TEMPORAL_HINTS.items():
            if hint in q_lower:
                return hr

        hm = re.search(r"(\d{1,2})\s*시", q_lower)
        if hm:
            h = int(hm.group(1))
            if 0 <= h <= 23:
                return (h, min(24, h + 1))

        pm = re.search(r"(\d{1,2})\s*pm", q_lower)
        if pm:
            h = int(pm.group(1))
            if 1 <= h <= 12:
                hh = 12 if h == 12 else h + 12
                return (hh, min(24, hh + 1))

        am = re.search(r"(\d{1,2})\s*am", q_lower)
        if am:
            h = int(am.group(1))
            if 1 <= h <= 12:
                hh = 0 if h == 12 else h
                return (hh, min(24, hh + 1))

        return None

    def _extract_action_tags(self, q_lower: str) -> List[str]:
        tags = []
        for tag, words in ACTION_LEXICON.items():
            if any(w.lower() in q_lower for w in words):
                tags.append(tag)
        return tags

    def _extract_search_terms(self, question: str, action_tags: List[str]) -> List[str]:
        terms: List[str] = []

        for tag in action_tags:
            terms.extend(ACTION_LEXICON.get(tag, []))

        raw_tokens = re.findall(r"[A-Za-z0-9_가-힣一-龥]{2,}", question)
        for tok in raw_tokens:
            norm = self._normalize_token(tok)
            low = norm.lower()
            if not norm or low in STOPWORDS:
                continue
            terms.append(norm)

        return terms

    def _normalize_token(self, token: str) -> str:
        t = token.strip().strip(".,?!:;\"'()[]{}")
        if not t:
            return ""

        for suf in KOREAN_SUFFIXES:
            if t.endswith(suf) and len(t) > len(suf) + 1:
                t = t[: -len(suf)]
                break

        if t and len(t) > 1 and t[-1] in KOREAN_PARTICLES:
            t = t[:-1]

        return t.strip()

    @staticmethod
    def _uniq_keep_order(items: List[str]) -> List[str]:
        seen = set()
        out = []
        for item in items:
            k = item.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(item)
        return out
