"""一键启动：PowerContext Server -> 生成周报 -> FastAPI 页面。

用法：
    python app.py
    python app.py --web-port 8081 --topic ""
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evidence import read_events  # noqa: E402
from make_weekly_report import REPORT_PATH, generate, write_report  # noqa: E402
from pc_client import (  # noqa: E402
    DEFAULT_PORT,
    PC_HOME,
    SERVER_STATE,
    PowerContextError,
    port_in_use,
    start_server,
)

DEFAULT_TOPIC = "我们上周为什么放弃方案B，梯度检查点是怎么替代它的？"
KIND_ZH = {"decision": "决定", "constraint": "约束", "progress": "进展", "next-step": "下一步"}

app = FastAPI(title="实验室周报助手", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory=str(ROOT / "templates"))
templates.env.globals["kind_zh"] = KIND_ZH

STATE: dict[str, Any] = {}


def current_report() -> dict | None:
    report = STATE.get("report")
    if report is None and REPORT_PATH.exists():
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        STATE["report"] = report
    return report


def server_state() -> dict:
    if not SERVER_STATE.exists():
        return {}
    try:
        return json.loads(SERVER_STATE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def mark_reuse(report: dict) -> None:
    """同一 citation 出现在焦点与正文时打标，避免被误读成重复数据。"""

    counts = Counter(
        item["citation"]
        for section in report["sections"]
        for item in section["items"]
        if item["citation"]
    )
    focus = report.get("focus") or {}
    focus_hits = focus.get("hits", [])
    focus_citations = {hit["citation"] for hit in focus_hits}
    for hit in focus_hits:
        hit["reused"] = counts.get(hit["citation"], 0) > 0
    for section in report["sections"]:
        for item in section["items"]:
            total = counts.get(item["citation"], 0) + (1 if item["citation"] in focus_citations else 0)
            item["reused"] = total > 1


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    report = current_report()
    if report is not None:
        mark_reuse(report)
    return templates.TemplateResponse(request, "index.html", {"report": report})


@app.get("/evidence", response_class=HTMLResponse)
def evidence_page(request: Request):
    report = current_report()
    events = read_events()
    focus = (report or {}).get("focus") or {}
    items = [item for section in (report or {}).get("sections", []) for item in section["items"]]
    totals = (report or {}).get("totals", {})
    stats = {
        "calls": len(events),
        "memories": totals.get("active_memories", 0),
        "items": len(items),
        "items_with_citation": sum(1 for item in items if item["citation"]),
        "missing_bodies": totals.get("missing_bodies", 0),
        "prepare_calls": sum(1 for event in events if str(event["step"]).startswith("prepare")),
        "focus_hits": len(focus.get("hits", [])),
        "focus_query": focus.get("query"),
        "total_latency_ms": round(sum(event["latency_ms"] for event in events), 2),
        "prepare_latency_ms": round(
            sum(event["latency_ms"] for event in events if str(event["step"]).startswith("prepare")), 2
        ),
    }
    return templates.TemplateResponse(
        request,
        "evidence.html",
        {"report": report, "events": events, "stats": stats, "server": server_state(), "home": str(PC_HOME)},
    )


def run_web(host: str, port: int) -> None:
    """uvicorn 关停后会重放信号；提前接管，保证重放时仍能回到 finally 清理子进程。"""

    def _exit_cleanly(signum, frame):
        raise SystemExit(0)

    for sig in (signal.SIGINT, signal.SIGBREAK):
        signal.signal(sig, _exit_cleanly)
    uvicorn.run(app, host=host, port=port, log_level="warning")


def main() -> int:
    parser = argparse.ArgumentParser(description="启动实验室周报助手（PowerContext + FastAPI）")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="PowerContext Server 端口")
    parser.add_argument("--web-port", type=int, default=8080, help="周报页面端口")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="焦点问题；传空字符串则不调用 search")
    parser.add_argument("--keywords", default=None, help="显式指定检索关键词（1-2 个）")
    args = parser.parse_args()

    if port_in_use(args.web_port):
        raise SystemExit(f"端口 {args.web_port} 已被占用，请用 --web-port 指定其他端口")

    server = start_server(port=args.port)
    try:
        report = generate(server, topic=args.topic or None, keywords=args.keywords)
        out_path = write_report(report)
        STATE["report"] = report
        print(f"[pc]     {server.base_url}  reused={server.reused}  startup={server.startup_ms} ms")
        print(f"[report] {out_path}")
        print(f"[web]    http://127.0.0.1:{args.web_port}")
        run_web("127.0.0.1", args.web_port)
        return 0
    finally:
        server.stop()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PowerContextError as error:
        print(f"[error] {error}", file=sys.stderr)
        raise SystemExit(1)
