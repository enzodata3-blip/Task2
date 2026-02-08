# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

This repository (`model_b`) is one side of an A/B experiment comparing Claude model performance. It pairs with a sibling `model_a` directory under a shared experiment root. The actual task work happens here; the `.claude/hooks/` infrastructure captures session telemetry automatically.

## Hook Architecture

Three Python scripts run automatically via `.claude/settings.local.json`:

- **`capture_session_event.py`** — fires on `SessionStart` and `SessionEnd`; records git metadata (commit hash, branch) at session start and writes JSONL log entries.
- **`process_transcript.py`** — fires on `Stop` (incremental) and `SessionEnd` (final); reads the raw Claude transcript JSONL, deduplicates messages, extracts thinking blocks, aggregates token usage, analyzes tool calls, computes git diff metrics, and rebuilds the session log in chronological order.
- **`claude_code_capture_utils.py`** — shared utilities: detects `model_a`/`model_b` lane from the working path, locates the experiment root (parent containing both lanes), reads `manifest.json` for task/assignment metadata, and routes log files to `<experiment_root>/logs/<model_lane>/`.

## Log Output

All session data is written to `<experiment_root>/logs/model_b/`:
- `session_<id>.jsonl` — processed event log (session_start → messages → session_summary)
- `session_<id>_raw.jsonl` — copy of the raw Claude transcript

The `manifest.json` at the experiment root maps task IDs and model assignments.

## Key Behaviors

- Git diff metrics exclude `.claude/`, `__pycache__/`, `node_modules/`, `.mypy_cache/`, `.pytest_cache/`, `.DS_Store`, `.vscode/`, `.idea/`.
- Thinking blocks are extracted as separate `assistant_thinking` entries in the log but their tokens are **not** double-counted (already included in the parent assistant message's `output_tokens`).
- Log files are rewritten atomically on `SessionEnd` (write to `.tmp` then `os.replace`).
