"""把每次 PowerContext 调用固化为 evidence/events.jsonl，供人工逐条复核。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pc_client import PROJECT_ROOT, CallResult, prepared_items

EVIDENCE_DIR = PROJECT_ROOT / "evidence"
EVENTS_PATH = EVIDENCE_DIR / "events.jsonl"
FIELDS = (
    "ts",
    "step",
    "path",
    "query",
    "status",
    "hits",
    "items",
    "content_bytes",
    "latency_ms",
    "artifact_ids",
)


def reset_events() -> None:
    """每次生成周报前清空，保证 events.jsonl 与当前 out/weekly_report.json 一一对应。"""

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    EVENTS_PATH.write_text("", encoding="utf-8")


def _ref(artifact: dict[str, Any]) -> str:
    return f"{artifact['family']}/{artifact['artifact_id']}@{artifact['revision']}"


def _artifact_ids(step: str, body: dict[str, Any]) -> list[str]:
    if step == "list":
        return [_ref(item["artifact"]) for item in body.get("items", [])]
    if step == "search":
        return [_ref(hit["memory"]["artifact"]) for hit in body.get("hits", [])]
    if step in {"prepare", "prepare-fill"}:
        return [_ref(item["citation"]["artifact_ref"]) for item in prepared_items(body)]
    if step == "remember":
        return [_ref(record["artifact"]) for record in body.get("records", [])]
    return []


def log_event(step: str, result: CallResult, *, query: str | None = None) -> dict[str, Any]:
    body = result.body if isinstance(result.body, dict) else {}
    hits = len(body["hits"]) if isinstance(body.get("hits"), list) else None
    if step == "list":
        items = len(body.get("items", []))
    elif step in {"prepare", "prepare-fill"}:
        items = len(prepared_items(body))
    elif step == "remember":
        items = len(body.get("records", []))
    else:
        items = None
    event = {
        "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "step": step,
        "path": result.path,
        "query": query,
        "status": result.status,
        "hits": hits,
        "items": items,
        "content_bytes": body.get("content_bytes"),
        "latency_ms": result.latency_ms,
        "artifact_ids": _artifact_ids(step, body),
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    with EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


def read_events() -> list[dict[str, Any]]:
    if not EVENTS_PATH.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in EVENTS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events
