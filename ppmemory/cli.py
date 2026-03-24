from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .offline_builder import BuildPaths, MemoryBuilder
from .planner import SearchPlanner
from .retriever import EvidenceRetriever
from .reasoner import GroundedReasoner


DEFAULT_EGOLIFE_DIR = Path("/workspace/EgoLifeCap")
DEFAULT_ARTIFACTS_DIR = Path("/workspace/ppagent/artifacts")



def cmd_build(args: argparse.Namespace) -> int:
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    builder = MemoryBuilder(
        BuildPaths(
            egolife_dir=Path(args.egolife_dir),
            db_path=artifacts_dir / "memory.db",
            planner_memory_path=artifacts_dir / "planner_memory.json",
        )
    )
    summary = builder.build(window_sec=args.window_sec, max_kappa_lines=args.max_kappa_lines)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0



def cmd_plan(args: argparse.Namespace) -> int:
    artifacts_dir = Path(args.artifacts_dir)
    planner = SearchPlanner(artifacts_dir / "planner_memory.json")
    blueprint = planner.plan(args.question, top_k=args.top_k)
    print(json.dumps(blueprint, ensure_ascii=False, indent=2))
    return 0



def cmd_ask(args: argparse.Namespace) -> int:
    artifacts_dir = Path(args.artifacts_dir)
    planner = SearchPlanner(artifacts_dir / "planner_memory.json")
    retriever = EvidenceRetriever(artifacts_dir / "memory.db")
    reasoner = GroundedReasoner()

    blueprint = planner.plan(args.question, top_k=args.top_k)
    evidences = retriever.retrieve(blueprint, top_k=args.top_k)
    answer = reasoner.answer(args.question, evidences)

    payload = {
        "question": args.question,
        "search_blueprint": blueprint,
        "retrieved": [
            {
                "rank": i + 1,
                "actor": ev["actor"],
                "day": ev["day"],
                "clip_id": ev["clip_id"],
                "window_range": ev["window_range"],
                "score": ev["score"],
                "video_pointer": ev["video_pointer"],
                "kappa_text": ev["kappa_text"],
                "events_preview": ev["events"][:5],
            }
            for i, ev in enumerate(evidences)
        ],
        "final": answer,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Personalized Parametric Process Memory Agent (MVP)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build external episodic DB and planner memory")
    p_build.add_argument("--egolife-dir", type=str, default=str(DEFAULT_EGOLIFE_DIR))
    p_build.add_argument("--artifacts-dir", type=str, default=str(DEFAULT_ARTIFACTS_DIR))
    p_build.add_argument("--window-sec", type=int, default=60)
    p_build.add_argument("--max-kappa-lines", type=int, default=8)
    p_build.set_defaults(func=cmd_build)

    p_plan = sub.add_parser("plan", help="Generate search blueprint JSON")
    p_plan.add_argument("--artifacts-dir", type=str, default=str(DEFAULT_ARTIFACTS_DIR))
    p_plan.add_argument("--question", type=str, required=True)
    p_plan.add_argument("--top-k", type=int, default=5)
    p_plan.set_defaults(func=cmd_plan)

    p_ask = sub.add_parser("ask", help="Run planner + retrieval + grounded answer")
    p_ask.add_argument("--artifacts-dir", type=str, default=str(DEFAULT_ARTIFACTS_DIR))
    p_ask.add_argument("--question", type=str, required=True)
    p_ask.add_argument("--top-k", type=int, default=5)
    p_ask.set_defaults(func=cmd_ask)

    return p



def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    rc = args.func(args)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
