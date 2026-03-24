# Personalized Parametric Process Memory (MVP)

This repository implements an end-to-end prototype of a personalized memory agent over EgoLifeCap.

## Implemented Scope (simplified)
- Implemented:
  - Offline episodic memory DB construction from `DenseCaption` and `Transcript`
  - Lightweight parametric planner memory (`planner_memory.json`)
  - Two-stage runtime (`planner -> retrieval`) plus grounded reasoning from evidence only
- Not implemented yet (intentionally skipped per request):
  - Habit tokens
  - Nightly/daytime split updates
  - LoRA training

## Folder layout
- Code: `/workspace/ppagent/ppmemory`
- Input data default: `/workspace/EgoLifeCap`
- Output artifacts default: `/workspace/ppagent/artifacts`

## Quickstart

```bash
cd /workspace/ppagent
python3 -m ppmemory.cli build \
  --egolife-dir /workspace/EgoLifeCap \
  --artifacts-dir /workspace/ppagent/artifacts
```

Ask a question:

```bash
python3 -m ppmemory.cli ask \
  --artifacts-dir /workspace/ppagent/artifacts \
  --question "나 어제 밥 먹고 양치하기 전에 뭐 했어?" \
  --top-k 5
```

Inspect planner blueprint only:

```bash
python3 -m ppmemory.cli plan \
  --artifacts-dir /workspace/ppagent/artifacts \
  --question "Jake가 DAY3 오후에 뭐했지?"
```

## Runtime design
1. Stage 1 (Memory ON): query -> search blueprint JSON
2. Stage 2 (Evidence retrieval): blueprint -> descriptor/event retrieval from external DB
3. Stage 3 (Memory OFF): answer generated from retrieved evidence only

## Artifact files
- `memory.db`: SQLite database (`events`, `descriptors`, optional FTS tables)
- `planner_memory.json`: actor/day priors and frequent action patterns

## Notes
- Retrieval works with FTS5 when available; otherwise falls back to SQL `LIKE` matching.
- This is text-grounded over captions/transcripts (no direct video tensor inference in this MVP).
