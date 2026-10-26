"""PowerContext 客户端与受管 Server 进程。

三条链路封装在 PowerContextClient 上：
  list_memories   -> /v1/atomic-memory/list
  search_memories -> /v1/atomic-memory/search
  prepare_context -> /v1/context/prepare
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PC_HOME = DATA_DIR / "pc-home"
SERVER_LOG = DATA_DIR / "pc-server.log"
SERVER_STATE = DATA_DIR / "server.json"
SCOPE_STATE = DATA_DIR / "scope.json"
DEFAULT_PORT = 8000

BEGIN_MARKER = "BEGIN_POWERCONTEXT_PREPARED_CONTEXT_V1"
END_MARKER = "END_POWERCONTEXT_PREPARED_CONTEXT_V1"

_ZH_STOPWORDS = frozenset(
    "的 了 和 与 及 或 是 在 把 被 将 对 为 从 到 中 上 下 里 这个 那个 这些 那些 "
    "我们 你们 他们 它 我 你 他 她 这 那 有 没有 要 会 能 可以 需要 进行 一个 一些 "
    "什么 怎么 如何 为什么 以及 还有 并且 但是 因为 所以 如果 然后 已经 正在 本周 上周 "
    "今天 明天 下周 the a an and or of to in on for with is are was were be this that".split()
)


class PowerContextError(RuntimeError):
    """PowerContext HTTP 调用或进程启动失败。"""

    def __init__(self, status: int, path: str, detail: str) -> None:
        self.status = status
        self.path = path
        self.detail = detail
        super().__init__(f"{path} -> HTTP {status}: {detail[:400]}")


@dataclass(frozen=True)
class CallResult:
    path: str
    latency_ms: float
    body: dict[str, Any]
    status: int = 200


def _request(
    method: str,
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 60.0,
) -> CallResult:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = Request(base_url.rstrip("/") + path, data=data, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise PowerContextError(error.code, path, detail) from error
    except URLError as error:
        raise PowerContextError(0, path, str(error.reason)) from error
    latency_ms = (time.perf_counter() - started) * 1000
    body = json.loads(raw) if raw.strip() else {}
    return CallResult(path=path, latency_ms=round(latency_ms, 2), body=body, status=response.status)


def health_ready(base_url: str, timeout: float = 2.0) -> bool:
    try:
        with urlopen(base_url.rstrip("/") + "/health/ready", timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def port_in_use(port: int) -> bool:
    """用 bind 探测占用：比 connect 更可靠（监听但 backlog 满的端口连不上却仍占用）。"""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return True
        return False


def powercontext_executable() -> str:
    override = os.environ.get("POWERCONTEXT_BIN")
    if override:
        return override
    found = shutil.which("powercontext")
    if found:
        return found
    name = "powercontext.exe" if os.name == "nt" else "powercontext"
    sibling = Path(sys.executable).with_name(name)
    if sibling.exists():
        return str(sibling)
    raise PowerContextError(0, "powercontext", "找不到 powercontext 命令，请先 pip install -r requirements.txt")


@dataclass
class ServerHandle:
    """已就绪的 PowerContext Server；process 为 None 表示复用了已在运行的实例。"""

    base_url: str
    port: int
    process: subprocess.Popen | None
    log_path: Path
    startup_ms: float = 0.0
    home: Path = PC_HOME

    @property
    def reused(self) -> bool:
        return self.process is None

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        _save_server_state(self, self.home, running=False)


def _save_server_state(handle: ServerHandle, home: Path, *, running: bool) -> None:
    """data/server.json 记录端口与启动/复用耗时，供 /evidence 页复核进程生命周期。"""

    previous: dict[str, Any] = {}
    if SERVER_STATE.exists():
        try:
            previous = json.loads(SERVER_STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
    state = {
        "port": handle.port,
        "base_url": handle.base_url,
        "pid": handle.process.pid if handle.process is not None else previous.get("pid"),
        "home": str(home),
        "reused": handle.reused,
        "startup_ms": handle.startup_ms,
        "running": running,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    SERVER_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


_FATAL_LOG_MARKERS = (
    "error while attempting to bind",
    "Address already in use",
    "[Errno 10048]",
)


def _new_fatal_log(offset: int) -> tuple[int, str | None]:
    """读取日志新增部分，命中致命标记时立刻返回，避免干等启动超时。"""

    if not SERVER_LOG.exists():
        return offset, None
    size = SERVER_LOG.stat().st_size
    if size <= offset:
        return offset, None
    with SERVER_LOG.open("rb") as handle:
        handle.seek(offset)
        chunk = handle.read().decode("utf-8", errors="replace")
    if any(marker in chunk for marker in _FATAL_LOG_MARKERS):
        return size, chunk[-1200:]
    return size, None


def start_server(
    port: int = DEFAULT_PORT,
    home: Path = PC_HOME,
    startup_timeout: float = 90.0,
    reuse: bool = True,
) -> ServerHandle:
    """启动（或复用）PowerContext Server，等到 /health/ready 返回 200 才返回。"""

    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    requested_base = f"http://127.0.0.1:{port}"
    if reuse:
        probe_started = time.perf_counter()
        if health_ready(requested_base):
            handle = ServerHandle(
                requested_base, port, None, SERVER_LOG, round((time.perf_counter() - probe_started) * 1000, 2)
            )
            _save_server_state(handle, home, running=True)
            return handle
        if SERVER_STATE.exists():
            saved_port = json.loads(SERVER_STATE.read_text(encoding="utf-8")).get("port")
            if isinstance(saved_port, int) and saved_port != port and health_ready(f"http://127.0.0.1:{saved_port}"):
                handle = ServerHandle(
                    f"http://127.0.0.1:{saved_port}",
                    saved_port,
                    None,
                    SERVER_LOG,
                    round((time.perf_counter() - probe_started) * 1000, 2),
                )
                _save_server_state(handle, home, running=True)
                return handle

    chosen = None
    for candidate in range(port, port + 20):
        if not port_in_use(candidate):
            chosen = candidate
            break
    if chosen is None:
        raise PowerContextError(0, "server", f"{port}-{port + 19} 端口都被占用，无法启动 PowerContext Server")

    env = os.environ.copy()
    env["POWERCONTEXT_HOME"] = str(home)
    # 默认 metrics 端口是 8001，HTTP 回退到 8001 时会自己撞自己。
    env["POWERCONTEXT_SERVER_METRICS_ENABLED"] = "false"
    log_offset = SERVER_LOG.stat().st_size if SERVER_LOG.exists() else 0
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    started = time.perf_counter()
    with SERVER_LOG.open("ab") as log:
        process = subprocess.Popen(
            [
                powercontext_executable(),
                "server",
                "run",
                "--host",
                "127.0.0.1",
                "--port",
                str(chosen),
                "--no-env-file",
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )

    base_url = f"http://127.0.0.1:{chosen}"
    deadline = time.monotonic() + startup_timeout
    exited = False
    while time.monotonic() < deadline:
        if health_ready(base_url):
            handle = ServerHandle(
                base_url,
                chosen,
                process,
                SERVER_LOG,
                round((time.perf_counter() - started) * 1000, 2),
                home=home,
            )
            _save_server_state(handle, home, running=True)
            return handle
        if process.poll() is not None:
            exited = True
            break
        log_offset, fatal = _new_fatal_log(log_offset)
        if fatal:
            process.kill()
            process.wait(timeout=10)
            raise PowerContextError(0, "server", f"Server 启动即失败：\n{fatal}")
        time.sleep(0.4)

    if process.poll() is None:
        process.kill()
        process.wait(timeout=10)
    tail = SERVER_LOG.read_text(encoding="utf-8", errors="replace")[-1500:]
    reason = "进程已退出" if exited else f"未在 {startup_timeout:.0f}s 内就绪"
    raise PowerContextError(0, "server", f"Server 启动失败（{reason}，端口 {chosen}）：\n{tail}")


class PowerContextClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def get_scope(self, scope_id: str) -> CallResult:
        return _request("GET", self.base_url, f"/v1/scopes/{scope_id}")

    def list_scopes(self, query: str | None = None) -> CallResult:
        path = "/v1/scopes"
        if query:
            from urllib.parse import quote

            path = f"{path}?query={quote(query)}"
        return _request("GET", self.base_url, path)

    def create_scope(self, title: str, summary: str, idempotency_key: str) -> CallResult:
        return _request(
            "POST",
            self.base_url,
            "/v1/scopes",
            {"title": title, "summary": summary, "idempotency_key": idempotency_key},
        )

    def remember(self, scope_id: str, kind: str, text: str, reason: str | None = None) -> CallResult:
        payload: dict[str, Any] = {"scope_id": scope_id, "kind": kind, "text": text}
        if reason:
            payload["reason"] = reason
        return _request("POST", self.base_url, "/v1/memory/remember", payload)

    def list_memories(
        self,
        scope_id: str,
        kind: str | None = None,
        states: Iterable[str] = ("active",),
        page_limit: int = 100,
    ) -> tuple[list[dict[str, Any]], list[CallResult]]:
        records: list[dict[str, Any]] = []
        calls: list[CallResult] = []
        cursor: str | None = None
        while True:
            payload: dict[str, Any] = {"scope_id": scope_id, "states": list(states), "limit": page_limit}
            if kind:
                payload["kind"] = kind
            if cursor:
                payload["cursor"] = cursor
            call = _request("POST", self.base_url, "/v1/atomic-memory/list", payload)
            calls.append(call)
            records.extend(call.body.get("items", []))
            cursor = call.body.get("next_cursor")
            if not cursor:
                return records, calls

    def search_memories(
        self, scope_id: str, query: str, limit: int = 10, mode: str = "text"
    ) -> CallResult:
        return _request(
            "POST",
            self.base_url,
            "/v1/atomic-memory/search",
            {"scope_id": scope_id, "query": query, "mode": mode, "limit": limit},
        )

    def prepare_context(self, scope_id: str, query: str, max_bytes: int = 3500) -> CallResult:
        return _request(
            "POST",
            self.base_url,
            "/v1/context/prepare",
            {"scope_id": scope_id, "query": query, "max_bytes": max_bytes},
        )


def ensure_scope(
    client: PowerContextClient,
    title: str,
    summary: str,
    idempotency_key: str,
) -> tuple[str, bool]:
    """复用 data/scope.json 里的 Scope；失效或冲突时按标题回查，再退回新建。"""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if SCOPE_STATE.exists():
        saved = json.loads(SCOPE_STATE.read_text(encoding="utf-8"))
        scope_id = saved.get("scope_id")
        if scope_id:
            try:
                client.get_scope(scope_id)
                return scope_id, True
            except PowerContextError as error:
                if error.status != 404:
                    raise

    try:
        created = client.create_scope(title, summary, idempotency_key)
    except PowerContextError as error:
        if error.status != 409:
            raise
        existing = client.list_scopes(query=title)
        matches = [item for item in existing.body.get("items", []) if item.get("title") == title]
        if not matches:
            raise
        created = CallResult("/v1/scopes", 0.0, matches[0])

    scope_id = created.body["scope_id"]
    SCOPE_STATE.write_text(
        json.dumps({"scope_id": scope_id, "title": title}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return scope_id, False


def _tokenize(text: str, chinese_only: bool = False) -> list[str]:
    try:
        import jieba

        jieba.setLogLevel(logging.WARNING)
        raw = list(jieba.cut(text, HMM=False))
    except Exception:
        raw = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", text)
    tokens: list[str] = []
    for token in raw:
        cleaned = token.strip().lower()
        cleaned = re.sub(r"^[^\w\u4e00-\u9fff]+|[^\w\u4e00-\u9fff]+$", "", cleaned)
        if len(cleaned) < 2 or cleaned in _ZH_STOPWORDS or cleaned.isdigit():
            continue
        if chinese_only and not re.fullmatch(r"[\u4e00-\u9fff]+", cleaned):
            continue
        tokens.append(cleaned)
    return tokens


def content_words(text: str, limit: int = 2) -> str:
    """按原句顺序取前 limit 个中文实词：单字、停用词、纯英文一律丢弃。"""

    words = list(dict.fromkeys(_tokenize(text, chinese_only=True)))
    return " ".join(words[:limit])


def compress_query(text: str, limit: int = 2) -> str:
    """把用户整句压成 1-2 个关键词。

    Server 在查询词数 >2 时要求命中 max(2, ceil(词数*25%)) 个词，整句通常会被拒成 0 条。
    """

    tokens = _tokenize(text)
    if not tokens:
        return ""
    counts = Counter(tokens)
    ranked = sorted(counts, key=lambda token: (counts[token], len(token)), reverse=True)
    return " ".join(ranked[:limit])


def prepared_items(body: dict[str, Any]) -> list[dict[str, Any]]:
    """从 PreparedContext.content 的标记区块里取出 items。"""

    content = body.get("content") or ""
    start = content.find(BEGIN_MARKER)
    end = content.find(END_MARKER)
    if start < 0 or end < 0 or end <= start:
        return []
    payload = json.loads(content[start + len(BEGIN_MARKER) : end].strip())
    return payload.get("items", [])


def citation_of(item: dict[str, Any]) -> str:
    ref = item["citation"]["artifact_ref"]
    return f"{ref['family']}/{ref['artifact_id']}@{ref['revision']}"
