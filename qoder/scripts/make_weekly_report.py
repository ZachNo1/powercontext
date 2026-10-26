"""从 PowerContext 生成周报草稿 JSON（第 1 步产物，不含 HTML）。

三条链路：
  1. atomic-memory/list     -> 按 kind 拉全量 active，搭骨架
  2. atomic-memory/search   -> 只有给了 --topic 才调用，query 先压成 1-2 个关键词
  3. context/prepare        -> 每条正文与 citation 的唯一来源
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evidence import log_event, reset_events  # noqa: E402
from pc_client import (  # noqa: E402
    DEFAULT_PORT,
    PC_HOME,
    PROJECT_ROOT,
    CallResult,
    PowerContextClient,
    PowerContextError,
    ServerHandle,
    citation_of,
    compress_query,
    content_words,
    ensure_scope,
    prepared_items,
    start_server,
)

SECTIONS = (
    ("progress", "本周进展", "本周 完成"),
    ("decision", "关键决定", "决定 方案"),
    ("constraint", "风险待办", "显存 上限"),
    ("next-step", "下周计划", "下周 计划"),
)
SEED_FILE = PROJECT_ROOT / "seed" / "lab_memories.json"
REPORT_PATH = PROJECT_ROOT / "out" / "weekly_report.json"


def artifact_id(record_or_item: dict) -> str:
    source = record_or_item.get("artifact") or record_or_item["citation"]["artifact_ref"]
    return source["artifact_id"]


def generate(
    server: ServerHandle,
    *,
    topic: str | None = None,
    keywords: str | None = None,
    max_bytes: int = 3500,
) -> dict:
    """在已就绪的 Server 上跑完三条链路；events.jsonl 与返回值一一对应。"""

    reset_events()
    trace: list[dict] = []

    def record(step: str, result: CallResult, *, query: str | None = None) -> None:
        trace.append(log_event(step, result, query=query))

    seed = json.loads(SEED_FILE.read_text(encoding="utf-8"))
    client = PowerContextClient(server.base_url)
    scope_id, scope_reused = ensure_scope(
        client, seed["title"], seed["summary"], seed["idempotency_key"]
    )

    # 链路 1：全量 active，做骨架
    records, list_calls = client.list_memories(scope_id)
    for call in list_calls:
        record("list", call)

    seeded = False
    if not records:
        for memory in seed["memories"]:
            call = client.remember(scope_id, memory["kind"], memory["text"], memory.get("reason"))
            record("remember", call)
        records, list_calls = client.list_memories(scope_id)
        for call in list_calls:
            record("list", call)
        seeded = True

    grouped = {kind: [r for r in records if r["kind"] == kind] for kind, _, _ in SECTIONS}
    unknown = sorted({r["kind"] for r in records} - {kind for kind, _, _ in SECTIONS})

    # 链路 2：可选主题检索，query 只发 1-2 个关键词
    focus = None
    if topic:
        query = keywords or compress_query(topic, 2)
        if not query:
            raise SystemExit("无法从 --topic 提取关键词，请用 --keywords 显式指定 1-2 个词")
        call = client.search_memories(scope_id, query, limit=10)
        record("search", call, query=query)
        focus = {
            "topic": topic,
            "query": query,
            "hits": [
                {
                    "citation": f"{hit['memory']['artifact']['family']}/"
                    f"{hit['memory']['artifact']['artifact_id']}@"
                    f"{hit['memory']['artifact']['revision']}",
                    "kind": hit["memory"]["kind"],
                    "text": hit["memory"]["text"],
                    "score": hit.get("score"),
                }
                for hit in call.body.get("hits", [])
            ],
        }

    # 链路 3：prepare 提供正文与 citation；未被段级召回覆盖的条目逐条补一次
    prepare_calls = 0
    sections: list[dict] = []

    for kind, title, query in SECTIONS:
        members = grouped.get(kind, [])
        section_meta = {"status": "empty", "content_bytes": 0, "latency_ms": 0.0, "query": query}
        section_items: dict[str, dict] = {}
        if members:
            call = client.prepare_context(scope_id, query, max_bytes=max_bytes)
            prepare_calls += 1
            record("prepare", call, query=query)
            items = prepared_items(call.body)
            for item in items:
                section_items.setdefault(artifact_id(item), item)
            section_meta = {
                "status": call.body.get("status"),
                "content_bytes": call.body.get("content_bytes"),
                "latency_ms": call.latency_ms,
                "query": query,
                "items": len(items),
            }

        rendered: list[dict] = []
        for member in members:
            memory_id = artifact_id(member)
            item = section_items.get(memory_id)
            source = "section-prepare"
            source_query = query
            if item is None:
                source = "item-prepare"
                source_query = content_words(member["text"], 2) or query
                call = client.prepare_context(scope_id, source_query, max_bytes=max_bytes)
                prepare_calls += 1
                record("prepare-fill", call, query=source_query)
                item = next(
                    (c for c in prepared_items(call.body) if artifact_id(c) == memory_id),
                    None,
                )

            rendered.append(
                {
                    "citation": citation_of(item) if item else None,
                    "kind": member["kind"],
                    "text": item["content"] if item else None,
                    "truncated": item["truncated"] if item else None,
                    "source": source,
                    "source_query": source_query,
                }
            )

        sections.append({"kind": kind, "title": title, "prepare": section_meta, "items": rendered})

    missing = sum(1 for section in sections for item in section["items"] if item["text"] is None)
    latencies = [entry["latency_ms"] for entry in trace]
    return {
        "schema": "lab-weekly-report.draft.v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": {"scope_id": scope_id, "title": seed["title"], "summary": seed["summary"]},
        "powercontext": {
            "base_url": server.base_url,
            "port": server.port,
            "home": str(PC_HOME),
            "database": str(PC_HOME / "powercontext.db"),
            "server_reused": server.reused,
            "server_startup_ms": server.startup_ms,
        },
        "focus": focus,
        "sections": sections,
        "totals": {
            "active_memories": len(records),
            "scope_reused": scope_reused,
            "seeded": seeded,
            "sections": len(sections),
            "items": sum(len(section["items"]) for section in sections),
            "missing_bodies": missing,
            "prepare_calls": prepare_calls,
            "latency_ms": round(sum(latencies), 2),
            "unknown_kinds": unknown,
        },
        "trace": trace,
    }


def write_report(report: dict, out_path: str | Path = REPORT_PATH) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="生成实验室周报草稿 JSON")
    parser.add_argument("--topic", default=None, help="可选主题；给出时走 atomic-memory/search")
    parser.add_argument("--keywords", default=None, help="显式指定检索关键词（1-2 个），覆盖自动压缩")
    parser.add_argument("--max-bytes", type=int, default=3500)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--out", default=str(REPORT_PATH))
    parser.add_argument("--keep-server", action="store_true", help="结束后保留 PowerContext Server")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    server = start_server(port=args.port)
    try:
        report = generate(server, topic=args.topic, keywords=args.keywords, max_bytes=args.max_bytes)
        out_path = write_report(report, args.out)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\n[report] {out_path}", file=sys.stderr)
        return 0
    finally:
        if not args.keep_server:
            server.stop()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PowerContextError as error:
        print(f"[error] {error}", file=sys.stderr)
        raise SystemExit(1)
