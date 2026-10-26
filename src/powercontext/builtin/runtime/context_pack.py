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

"""Prefix-reusing Context Pack transport for prepared context content."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from powercontext.errors import PowerContextError

CONTEXT_PACK_SCHEMA = "powercontext.context-pack.v1"
CONTEXT_PACK_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
DEFAULT_CONTEXT_PACK_CHUNK_BYTES = 512
"""Default chunk ceiling for the public ``ContextPack.from_content`` entry point."""

INCREMENTAL_CONTEXT_CHUNK_BYTES = 128
"""Chunk ceiling used by incremental prepare.

Prepared context keeps a short stable prologue before a body that changes on
every turn. Smaller chunks keep that prologue reusable instead of merging it
into one oversized first chunk that never matches again; the extra Base64
framing costs only a few bytes per chunk.
"""

_PACK_FIELDS = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_FIELDS = frozenset({"schema", "base_digest", "target_digest", "prefix_items", "appended"})
_DIGEST_PREFIX = "sha256:"


class ContextPackError(PowerContextError):
    """Raised when a Context Pack or delta violates the wire contract."""


@dataclass(frozen=True, slots=True)
class ContextPack:
    """A complete chunked context payload with its canonical digest."""

    items: tuple[bytes, ...]
    digest: str

    @classmethod
    def build(cls, items: tuple[bytes, ...] | list[bytes]) -> ContextPack:
        normalized = tuple(bytes(item) for item in items)
        return cls(items=normalized, digest=_pack_digest(normalized))

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = DEFAULT_CONTEXT_PACK_CHUNK_BYTES) -> ContextPack:
        """Split text into deterministic chunks that are valid UTF-8 and bounded."""

        return cls.build(_chunk_content(content, chunk_bytes))

    @classmethod
    def from_bytes(cls, raw: bytes) -> ContextPack:
        payload = _load_json(raw, "context pack")
        _require_exact_fields(payload, _PACK_FIELDS, "context pack")
        if payload["schema"] != CONTEXT_PACK_SCHEMA:
            raise ContextPackError("context pack schema is not supported")  # noqa: TRY003
        items = _decode_items(payload["items"], "context pack items")
        digest = payload["digest"]
        if not isinstance(digest, str):
            raise ContextPackError("context pack digest must be a string")  # noqa: TRY003
        if digest != _pack_digest(items):
            raise ContextPackError("context pack digest does not match its items")  # noqa: TRY003
        if payload["byte_count"] != len(bytes(raw)):
            raise ContextPackError("context pack byte_count does not match the payload")  # noqa: TRY003
        return cls(items=items, digest=digest)

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        return _encode_pack(self.items, self.digest)


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """A prefix reference into a base pack plus the target suffix."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        """Keep the longest shared prefix and carry the remaining target chunks."""

        prefix = 0
        for left, right in zip(base.items, target.items, strict=False):
            if left != right:
                break
            prefix += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix,
            appended=target.items[prefix:],
        )

    @classmethod
    def from_bytes(cls, raw: bytes) -> ContextPackDelta:
        payload = _load_json(raw, "context pack delta")
        _require_exact_fields(payload, _DELTA_FIELDS, "context pack delta")
        if payload["schema"] != CONTEXT_PACK_DELTA_SCHEMA:
            raise ContextPackError("context pack delta schema is not supported")  # noqa: TRY003
        base_digest = payload["base_digest"]
        target_digest = payload["target_digest"]
        if not isinstance(base_digest, str) or not isinstance(target_digest, str):
            raise ContextPackError("context pack delta digests must be strings")  # noqa: TRY003
        prefix_items = payload["prefix_items"]
        if type(prefix_items) is not int or prefix_items < 0:
            raise ContextPackError("context pack delta prefix_items must be a non-negative integer")  # noqa: TRY003
        appended = _decode_items(payload["appended"], "context pack delta appended")
        return cls(
            base_digest=base_digest,
            target_digest=target_digest,
            prefix_items=prefix_items,
            appended=appended,
        )

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        return _canonical_json(
            {
                "schema": CONTEXT_PACK_DELTA_SCHEMA,
                "base_digest": self.base_digest,
                "target_digest": self.target_digest,
                "prefix_items": self.prefix_items,
                "appended": [_encode_item(item) for item in self.appended],
            }
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Rebuild the target pack from a base pack and one delta."""

    if not isinstance(base, ContextPack):
        raise ContextPackError("base must be a ContextPack")  # noqa: TRY003
    if not isinstance(delta, ContextPackDelta):
        raise ContextPackError("delta must be a ContextPackDelta")  # noqa: TRY003
    if base.digest != delta.base_digest:
        raise ContextPackError("delta does not belong to the supplied base pack")  # noqa: TRY003
    prefix = delta.prefix_items
    if type(prefix) is not int or prefix < 0 or prefix > len(base.items):
        raise ContextPackError("delta prefix_items is out of range for the base pack")  # noqa: TRY003
    rebuilt = ContextPack.build(base.items[:prefix] + tuple(delta.appended))
    if rebuilt.digest != delta.target_digest:
        raise ContextPackError("rebuilt pack does not match the delta target digest")  # noqa: TRY003
    return rebuilt


def _chunk_content(content: str, chunk_bytes: int) -> tuple[bytes, ...]:
    if type(chunk_bytes) is not int or chunk_bytes < 1:
        raise ContextPackError("chunk_bytes must be a positive integer")  # noqa: TRY003
    if not isinstance(content, str):
        raise ContextPackError("content must be text")  # noqa: TRY003
    if not content:
        return ()

    chunks: list[bytes] = []
    current = bytearray()
    for character in content:
        encoded = character.encode("utf-8")
        if len(encoded) > chunk_bytes:
            raise ContextPackError("chunk_bytes cannot hold one complete character")  # noqa: TRY003
        if len(current) + len(encoded) > chunk_bytes:
            chunks.append(bytes(current))
            current = bytearray()
        current.extend(encoded)
    if current:
        chunks.append(bytes(current))
    return tuple(chunks)


def _pack_digest(items: tuple[bytes, ...]) -> str:
    canonical = _canonical_json(
        {
            "schema": CONTEXT_PACK_SCHEMA,
            "items": [_encode_item(item) for item in items],
        }
    )
    return f"{_DIGEST_PREFIX}{hashlib.sha256(canonical).hexdigest()}"


def _encode_pack(items: tuple[bytes, ...], digest: str) -> bytes:
    body = {
        "schema": CONTEXT_PACK_SCHEMA,
        "items": [_encode_item(item) for item in items],
        "digest": digest,
    }
    # byte_count is part of the payload it measures, so settle on the fixed point.
    byte_count = 0
    for _ in range(8):
        raw = _canonical_json({**body, "byte_count": byte_count})
        if len(raw) == byte_count:
            return raw
        byte_count = len(raw)
    raise ContextPackError("context pack byte_count did not converge")  # noqa: TRY003


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _load_json(raw: bytes, label: str) -> dict[str, Any]:
    if not isinstance(raw, (bytes, bytearray)):
        raise ContextPackError(f"{label} must be bytes")  # noqa: TRY003
    try:
        payload = json.loads(bytes(raw).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContextPackError(f"{label} must be valid UTF-8 JSON") from error  # noqa: TRY003
    if not isinstance(payload, dict):
        raise ContextPackError(f"{label} must be a JSON object")  # noqa: TRY003
    return payload


def _require_exact_fields(payload: dict[str, Any], expected: frozenset[str], label: str) -> None:
    if set(payload) != expected:
        raise ContextPackError(f"{label} must contain exactly {sorted(expected)}")  # noqa: TRY003


def _encode_item(item: bytes) -> str:
    return base64.b64encode(item).decode("ascii")


def _decode_items(values: Any, label: str) -> tuple[bytes, ...]:
    if not isinstance(values, list):
        raise ContextPackError(f"{label} must be a list")  # noqa: TRY003
    decoded: list[bytes] = []
    for value in values:
        if not isinstance(value, str):
            raise ContextPackError(f"{label} entries must be Base64 strings")  # noqa: TRY003
        try:
            decoded.append(base64.b64decode(value.encode("ascii"), validate=True))
        except (binascii.Error, UnicodeEncodeError) as error:
            raise ContextPackError(f"{label} entries must be valid Base64") from error  # noqa: TRY003
    return tuple(decoded)


__all__ = [
    "CONTEXT_PACK_DELTA_SCHEMA",
    "CONTEXT_PACK_SCHEMA",
    "DEFAULT_CONTEXT_PACK_CHUNK_BYTES",
    "INCREMENTAL_CONTEXT_CHUNK_BYTES",
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
