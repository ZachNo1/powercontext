# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture the raw visible events of one Pi session branch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, Field, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import (
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
)
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PI_EVENT_SOURCE_NAME = "pi-event"
PI_EVENT_SOURCE_VERSION = "1"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"
PI_SESSION_CONNECTOR_VERSION = "1"
PI_SESSION_CHECKPOINT_SCHEMA = "powercontext.pi-session-history.v1"
PI_IDENTITY_PREFIX = "pi"

_SESSION_TYPE = "session"
_SESSION_VERSION = 3
_TEXT_BLOCK = "text"
_TOOL_CALL_BLOCK = "toolCall"


class PiSessionInput(BaseModel):
    """One Pi session archive plus the branch leaf to walk back from."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event on the selected branch, with its complete record."""

    session_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    parent_id: str | None = None
    event_type: str = Field(min_length=1)
    role: str = Field(min_length=1)
    content: str = Field(min_length=1)
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """A captured Pi event whose identity is the session plus event id."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None = None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Build the stable event identity from its two encoded components."""

    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-blank string")  # noqa: TRY003
    if not isinstance(event_id, str) or not event_id.strip():
        raise ValueError("event_id must be a non-blank string")  # noqa: TRY003
    identity = f"{PI_IDENTITY_PREFIX}:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError("pi event identity exceeds the maximum source id length")  # noqa: TRY003
    return identity


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Return every visible event from the root to the selected leaf."""

    records = _load_records(value.content)
    session_id = _validate_header(records[0])
    nodes = _build_nodes(records[1:])
    leaf_id = value.leaf_id if value.leaf_id is not None else (records[-1].get("id") if len(records) > 1 else None)
    if not isinstance(leaf_id, str) or leaf_id not in nodes:
        raise ValueError("selected leaf is not present in the session")  # noqa: TRY003

    chain: list[dict[str, Any]] = []
    current: str | None = leaf_id
    while current is not None:
        node = nodes[current]
        chain.append(node)
        current = node.get("parentId")
    chain.reverse()

    events: list[PiEventInput] = []
    for record in chain:
        visible = _visible_event(record)
        if visible is None:
            continue
        role, body = visible
        events.append(
            PiEventInput(
                session_id=session_id,
                event_id=record["id"],
                parent_id=record.get("parentId"),
                event_type=record.get("type"),
                role=role,
                content=f"{role}: {body}",
                record=record,
            )
        )
    return tuple(events)


class PiSessionConnector:
    """Import new branch events while keeping durable progress across runs."""

    name = PI_SESSION_CONNECTOR_NAME
    version = PI_SESSION_CONNECTOR_VERSION
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        raw = self._path.read_bytes()
        events = parse_pi_session(PiSessionInput(content=raw, leaf_id=self._leaf_id))
        imported = _load_checkpoint(session.checkpoint)

        digests = {event.event_id: _record_digest(event.record) for event in events}
        for event_id, digest in digests.items():
            previous = imported.get(event_id)
            if previous is not None and previous != digest:
                raise ValueError(f"already imported pi event {event_id!r} was rewritten")  # noqa: TRY003

        for event in events:
            if event.event_id in imported:
                continue
            outcome = await session.submit(event.event_id, PI_EVENT_SOURCE_NAME, event)
            if outcome.source_ref is not None:
                imported[event.event_id] = digests[event.event_id]

        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint={
                "schema": PI_SESSION_CHECKPOINT_SCHEMA,
                "events": imported,
            },
        )


def _load_records(content: bytes) -> list[dict[str, Any]]:
    if not isinstance(content, (bytes, bytearray)):
        raise ValueError("pi session content must be bytes")  # noqa: TRY003
    try:
        text = bytes(content).decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("pi session content must be valid UTF-8") from error
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("pi session content must be JSON Lines") from error
        if not isinstance(record, dict):
            raise ValueError("pi session records must be JSON objects")  # noqa: TRY003
        records.append(record)
    if not records:
        raise ValueError("pi session content must contain a header record")  # noqa: TRY003
    return records


def _validate_header(header: dict[str, Any]) -> str:
    if header.get("type") != _SESSION_TYPE:
        raise ValueError("pi session header must declare type=session")  # noqa: TRY003
    if header.get("version") != _SESSION_VERSION:
        raise ValueError("pi session header must declare version=3")  # noqa: TRY003
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("pi session header must declare a non-empty id")  # noqa: TRY003
    return session_id


def _build_nodes(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    for record in records:
        node_id = record.get("id")
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError("every pi session record must declare a non-empty id")  # noqa: TRY003
        if node_id in nodes:
            raise ValueError("pi session record ids must be unique")  # noqa: TRY003
        parent = record.get("parentId")
        if parent is not None and parent not in nodes:
            raise ValueError("pi session parentId must reference an earlier record")  # noqa: TRY003
        nodes[node_id] = record
    return nodes


def _visible_event(record: dict[str, Any]) -> tuple[str, str] | None:
    kind = record.get("type")
    if kind == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if not isinstance(role, str) or not role.strip():
            return None
        if role == "toolResult":
            body = _tool_result_body(message)
        elif role == "bashExecution":
            body = _bash_body(message)
        else:
            body = _blocks_body(message.get("content"))
        return (role, body) if body else None
    if kind in {"compaction", "branch_summary"}:
        summary = record.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            return None
        return kind, summary
    if kind == "custom_message":
        body = _blocks_body(record.get("content"))
        return ("custom", body) if body else None
    return None


def _blocks_body(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == _TEXT_BLOCK:
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        elif block_type == _TOOL_CALL_BLOCK:
            name = block.get("name")
            call_id = block.get("id")
            arguments = block.get("arguments")
            rendered = _compact_json(arguments) if arguments is not None else "{}"
            parts.append(f"[tool_call {name} {call_id}] {rendered}")
    return "\n".join(parts)


def _tool_result_body(message: dict[str, Any]) -> str:
    name = message.get("toolName")
    call_id = message.get("toolCallId")
    text = _blocks_body(message.get("content"))
    return f"[tool_result {name} {call_id}] {text}"


def _bash_body(message: dict[str, Any]) -> str:
    command = message.get("command")
    output = message.get("output")
    if isinstance(output, str) and output:
        return f"$ {command}\n{output}"
    return f"$ {command}"


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _record_digest(record: dict[str, JsonValue]) -> str:
    return hashlib.sha256(_compact_json(record).encode("utf-8")).hexdigest()


def _load_checkpoint(checkpoint: JsonValue | None) -> dict[str, str]:
    if checkpoint is None:
        return {}
    if not isinstance(checkpoint, dict):
        raise ValueError("pi session checkpoint must be an object")  # noqa: TRY003
    events = checkpoint.get("events", {})
    if not isinstance(events, dict):
        raise ValueError("pi session checkpoint events must be an object")  # noqa: TRY003
    restored: dict[str, str] = {}
    for event_id, digest in events.items():
        if not isinstance(event_id, str) or not isinstance(digest, str):
            raise ValueError("pi session checkpoint entries must be strings")  # noqa: TRY003
        restored[event_id] = digest
    return restored


class PiEventSourceAdapter:
    """Resolve and read one captured Pi event."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    """Expose one Pi event as canonical text evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_ADAPTER = PiEventSourceAdapter()
PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PI_EVENT_SOURCE_ADAPTER,
    version=PI_EVENT_SOURCE_VERSION,
    projections=(PiEventTextEvidenceProjection(),),
)


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_SESSION_CONNECTOR_NAME",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
