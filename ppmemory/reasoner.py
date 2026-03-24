from __future__ import annotations

from typing import Dict, List


class GroundedReasoner:
    """
    Evidence-only answer generator.
    Memory priors are intentionally ignored here (memory OFF) to reduce hallucination risk.
    """

    def answer(self, question: str, evidences: List[Dict[str, object]]) -> Dict[str, object]:
        if not evidences:
            return {
                "answer": "검색 조건에 맞는 근거 구간을 찾지 못했습니다. 질문을 더 구체화해 주세요.",
                "grounded": True,
                "memory_off": True,
                "evidence_count": 0,
                "supporting_facts": [],
                "evidence_pointers": [],
            }

        top = evidences[:3]

        supporting_facts: List[str] = []
        pointers: List[str] = []
        seen = set()

        for ev in top:
            pointers.append(f"{ev['actor']} {ev['day']} {ev['video_pointer']}")
            for item in ev.get("events", [])[:4]:
                txt = str(item.get("text", "")).strip()
                if not txt or txt in seen:
                    continue
                seen.add(txt)
                supporting_facts.append(
                    f"[{item.get('start')}-{item.get('end')}] {txt}"
                )
                if len(supporting_facts) >= 8:
                    break
            if len(supporting_facts) >= 8:
                break

        main_ev = top[0]
        answer = (
            "근거 구간 기준으로 보면, "
            f"{main_ev['actor']} {main_ev['day']}의 {main_ev['video_pointer']} 주변에서 "
            "질문과 연관된 행동이 확인됩니다. "
            "아래 근거 문장을 기반으로 해석하면, 질문 시점 직전/직후 행동 흐름은 supporting_facts와 같습니다."
        )

        return {
            "answer": answer,
            "grounded": True,
            "memory_off": True,
            "evidence_count": len(evidences),
            "supporting_facts": supporting_facts,
            "evidence_pointers": pointers,
        }
