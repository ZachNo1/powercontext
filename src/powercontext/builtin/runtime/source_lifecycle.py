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

"""Propagate Source availability through a Scope's Memory dependency graph.

Every Scope owns one directed acyclic graph: registered Sources are the leaves and carry their
own ``available`` flag, while Memory and Topic Memory nodes derive their availability from their
parents. A node uses a single ``all``/``any`` policy over its parents, and the whole body of a
node is either usable or not - this module never rewrites content.

Deactivation is expressed through the artifacts' native state, never through a copy:

* Memory entries move between the ``active`` and ``inactive`` states already supported by the
  Memory service, so listing, search and prepared context pick the change up for free;
* Topic Memory artifacts stay immutable, and their visibility is reported to the Topic search
  filter as a set of unsupported artifact ids.

The graph is persisted per Scope, so a reopened Runtime answers the same questions a live one
does. A node's references are validated when it is registered, and an event is only recorded
after every native write it implies has landed, so a retry after an interrupted propagation
replays the same event without inventing revisions.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import delete, insert, select

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory.models import MemoryCitation
from powercontext.builtin.persistence.tables import (
    SOURCE_LIFECYCLE_EVENTS_TABLE,
    SOURCE_LIFECYCLE_NODES_TABLE,
)
from powercontext.builtin.runtime.models import GetMemoryEntryRequest
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import BuiltinRuntime

NodeKind: TypeAlias = Literal["source", "memory", "topic"]
DependencyPolicy: TypeAlias = Literal["all", "any"]
ResolutionPurpose: TypeAlias = Literal["generation", "audit"]
LifecycleReference: TypeAlias = SourceRef | MemoryCitation | ArtifactRef
_PURPOSES: frozenset[str] = frozenset({"generation", "audit"})
_POLICIES: frozenset[str] = frozenset({"all", "any"})
_REASON = "source lifecycle propagation"


class SourceLifecycleError(ValueError):
    """A lifecycle command that contradicts the registered dependency graph."""


class SourceLifecycleEvent(BaseModel):
    """One availability change for a registered Source.

    ``revision`` names the lifecycle version the caller believes it is editing, and ``sequence``
    orders events for the same node: replaying an identical event is a no-op, while a lower
    sequence - or the same sequence carrying different content - is a contradiction, not a retry.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    node_id: Annotated[str, Field(min_length=1)]
    reason: Annotated[str, Field(min_length=1)]
    revision: Annotated[int, Field(ge=1)]
    sequence: Annotated[int, Field(ge=1)]
    available: bool

    @field_validator("node_id", "reason")
    @classmethod
    def validate_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must contain non-whitespace content")  # noqa: TRY003
        return value


@dataclass(frozen=True)
class _LifecycleNode:
    """One registered node: its reference, its parents and its derived availability."""

    node_id: str
    kind: NodeKind
    reference: LifecycleReference
    parents: tuple[str, ...]
    policy: DependencyPolicy
    revision: int
    available: bool


@dataclass(frozen=True)
class _AppliedEvent:
    """The last event accepted for one Source, kept so replays can be told from contradictions."""

    sequence: int
    available: bool
    revision: int


@dataclass
class SourceLifecycleGraph:
    """One Scope's registered nodes, current availability and accepted event watermarks."""

    nodes: dict[str, _LifecycleNode] = field(default_factory=dict)
    events: dict[str, _AppliedEvent] = field(default_factory=dict)

    def recompute(self) -> None:
        """Re-derive every non-Source node from its parents in registration order.

        Registration rejects unknown parents, so parents always precede their children and one
        forward pass settles all/any for the whole graph, however deep it is.
        """

        for node_id in list(self.nodes):
            node = self.nodes[node_id]
            if node.kind == "source":
                continue
            support = tuple(self.nodes[parent].available for parent in node.parents)
            available = all(support) if node.policy == "all" else any(support)
            if available != node.available:
                self.nodes[node_id] = replace(node, available=available)

    def active_ids(self) -> list[str]:
        return sorted(node_id for node_id, node in self.nodes.items() if node.available)

    def unavailable_topics(self) -> frozenset[str]:
        """Artifact ids of Topic Memory nodes that currently have no support."""

        return frozenset(
            node.reference.artifact_id
            for node in self.nodes.values()
            if node.kind == "topic" and not node.available and isinstance(node.reference, ArtifactRef)
        )

    def memory_nodes(self) -> tuple[_LifecycleNode, ...]:
        return tuple(node for node in self.nodes.values() if node.kind == "memory")


class SourceLifecycleApplication:
    """Entry point reachable as ``runtime.lifecycle``."""

    def __init__(self, runtime: BuiltinRuntime) -> None:
        self._runtime = runtime

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycle:
        return ScopedSourceLifecycle(self._runtime, scope_id)


class ScopedSourceLifecycle:
    """Register evidence and move Source availability inside one Scope."""

    def __init__(self, runtime: BuiltinRuntime, scope_id: str) -> None:
        self._runtime = runtime
        self.scope_id = scope_id

    async def register_source(self, node_id: str, source_ref: SourceRef, *, revision: int) -> None:
        """Bind one captured Source to a node id at a lifecycle ``revision``."""

        _require_identifier(node_id)
        if type(revision) is not int or revision < 1:
            raise SourceLifecycleError("revision-must-be-positive-int")
        source = SourceRef.model_validate(source_ref)
        try:
            await self._runtime.records.for_scope(self.scope_id).get_source(
                source.source_type, source.source_id
            )
        except Exception as error:
            raise SourceLifecycleError("unknown-source-reference") from error
        node = _LifecycleNode(
            node_id=node_id,
            kind="source",
            reference=source,
            parents=(),
            policy="all",
            revision=revision,
            available=True,
        )
        await self._commit_registration(node)

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        *,
        parents: Sequence[str],
        policy: DependencyPolicy,
    ) -> None:
        """Bind one exact Memory entry to a node id and its supporting parents."""

        _require_identifier(node_id)
        reference = MemoryCitation.model_validate(citation)
        await self._runtime.memory.for_scope(self.scope_id).get(GetMemoryEntryRequest(citation=reference))
        node = _LifecycleNode(
            node_id=node_id,
            kind="memory",
            reference=reference,
            parents=_require_parents(node_id, parents, policy),
            policy=policy,
            revision=0,
            available=True,
        )
        registered = await self._commit_registration(node)
        if not registered.available:
            # A node registered under withdrawn evidence is unsupported from birth; the caller
            # must never see it as usable first and inactive a moment later.
            await self._write_memory_state(reference, active=False)

    async def register_topic(
        self,
        node_id: str,
        artifact_ref: ArtifactRef,
        *,
        parents: Sequence[str],
        policy: DependencyPolicy,
    ) -> None:
        """Bind one Topic Memory Revision to a node id and its supporting parents."""

        _require_identifier(node_id)
        reference = ArtifactRef.model_validate(artifact_ref)
        await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
            reference.family, reference.artifact_id, reference.revision
        )
        node = _LifecycleNode(
            node_id=node_id,
            kind="topic",
            reference=reference,
            parents=_require_parents(node_id, parents, policy),
            policy=policy,
            revision=0,
            available=True,
        )
        await self._commit_registration(node)

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        """Apply one Source event and return every currently available node id, sorted.

        Native writes happen before the event watermark is recorded: if one of them fails, the
        event stays unapplied and the caller's retry finishes the propagation. Because every
        native write skips an entry already in the target state, the retry adds no revisions.
        """

        graph = await self._graph()
        node = graph.nodes.get(event.node_id)
        if node is None or node.kind != "source":
            raise SourceLifecycleError("unknown-source-node")
        if event.revision != node.revision:
            raise SourceLifecycleError("revision-mismatch")
        previous = graph.events.get(event.node_id)
        if previous is not None:
            if event.sequence < previous.sequence:
                raise SourceLifecycleError("sequence-regression")
            if event.sequence == previous.sequence:
                if (event.available, event.revision) != (previous.available, previous.revision):
                    raise SourceLifecycleError("sequence-conflict")
                return graph.active_ids()
        graph.nodes[event.node_id] = replace(node, available=event.available)
        graph.recompute()
        for memory_node in graph.memory_nodes():
            await self._write_memory_state(
                memory_node.reference,
                active=memory_node.available,
                node_id=memory_node.node_id,
            )
        graph.events[event.node_id] = _AppliedEvent(
            sequence=event.sequence,
            available=event.available,
            revision=event.revision,
        )
        await self._persist(graph)
        return graph.active_ids()

    async def resolve(self, node_id: str, *, purpose: str) -> LifecycleReference:
        """Return a node's registered reference for generation or audit reads."""

        if purpose not in _PURPOSES:
            raise SourceLifecycleError("unknown-purpose")
        node = (await self._graph()).nodes.get(node_id)
        if node is None:
            raise SourceLifecycleError("unknown-node")
        if purpose == "generation" and not node.available:
            raise SourceLifecycleError("unavailable-node")
        return node.reference

    async def graph(self) -> SourceLifecycleGraph:
        return await self._graph()

    async def _graph(self) -> SourceLifecycleGraph:
        graphs = getattr(self._runtime, "_lifecycle_graphs", None)
        if graphs is None:
            graphs = {}
            self._runtime._lifecycle_graphs = graphs
        graph = graphs.get(self.scope_id)
        if graph is None:
            graph = await self._read(self.scope_id)
            graphs[self.scope_id] = graph
        return graph

    async def _read(self, scope_id: str) -> SourceLifecycleGraph:
        graph = SourceLifecycleGraph()
        async with self._runtime._provider.database.transaction() as connection:
            node_rows = (
                await connection.execute(
                    select(SOURCE_LIFECYCLE_NODES_TABLE)
                    .where(SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id)
                    .order_by(SOURCE_LIFECYCLE_NODES_TABLE.c.ordinal)
                )
            ).mappings()
            for row in node_rows:
                graph.nodes[row["node_id"]] = _LifecycleNode(
                    node_id=row["node_id"],
                    kind=row["kind"],
                    reference=_load_reference(row["kind"], row["reference"]),
                    parents=tuple(json.loads(bytes(row["parents"]))),
                    policy=row["policy"],
                    revision=row["revision"],
                    available=row["available"],
                )
            event_rows = (
                await connection.execute(
                    select(SOURCE_LIFECYCLE_EVENTS_TABLE).where(SOURCE_LIFECYCLE_EVENTS_TABLE.c.scope_id == scope_id)
                )
            ).mappings()
            for row in event_rows:
                graph.events[row["node_id"]] = _AppliedEvent(
                    sequence=row["sequence"],
                    available=row["available"],
                    revision=row["revision"],
                )
        return graph

    async def _persist(self, graph: SourceLifecycleGraph) -> None:
        """Rewrite this Scope's graph rows in one transaction.

        The graph is small (tens of nodes) and rewritten as a whole, so a partially written
        propagation can never be observed: either the new availability and its event watermark both
        land, or neither does and the caller's retry starts from the previous state.
        """

        nodes = [
            {
                "scope_id": self.scope_id,
                "node_id": node.node_id,
                "kind": node.kind,
                "ordinal": ordinal,
                "reference": _dump_reference(node.reference),
                "parents": json.dumps(list(node.parents)).encode("utf-8"),
                "policy": node.policy,
                "revision": node.revision,
                "available": node.available,
            }
            for ordinal, node in enumerate(graph.nodes.values())
        ]
        events = [
            {
                "scope_id": self.scope_id,
                "node_id": node_id,
                "sequence": applied.sequence,
                "available": applied.available,
                "revision": applied.revision,
            }
            for node_id, applied in graph.events.items()
        ]
        async with self._runtime._provider.database.transaction() as connection:
            await connection.execute(
                delete(SOURCE_LIFECYCLE_NODES_TABLE).where(SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == self.scope_id)
            )
            await connection.execute(
                delete(SOURCE_LIFECYCLE_EVENTS_TABLE).where(SOURCE_LIFECYCLE_EVENTS_TABLE.c.scope_id == self.scope_id)
            )
            if nodes:
                await connection.execute(insert(SOURCE_LIFECYCLE_NODES_TABLE), nodes)
            if events:
                await connection.execute(insert(SOURCE_LIFECYCLE_EVENTS_TABLE), events)

    async def _commit_registration(self, node: _LifecycleNode) -> _LifecycleNode:
        graph = await self._graph()
        existing = graph.nodes.get(node.node_id)
        if existing is not None:
            if _same_registration(existing, node):
                return existing
            raise SourceLifecycleError("node-conflict")
        for parent in node.parents:
            if parent == node.node_id:
                raise SourceLifecycleError("self-dependency")
            if parent not in graph.nodes:
                raise SourceLifecycleError("unknown-parent")
        graph.nodes[node.node_id] = node
        graph.recompute()
        await self._persist(graph)
        return graph.nodes[node.node_id]

    async def _write_memory_state(
        self,
        reference: LifecycleReference,
        *,
        active: bool,
        node_id: str = "",
    ) -> None:
        """Move one Memory entry to ``active``/``inactive`` without touching its content.

        The Memory service already returns the current revision unchanged when the entry is in the
        target state, so this is both the propagation write and the idempotent retry path: an
        interrupted ``apply`` leaves nothing behind but the writes that already landed, and the
        retry skips them because their state matches.
        """

        if not isinstance(reference, MemoryCitation):
            raise SourceLifecycleError("memory-reference-expected")
        async with self._runtime._context(self.scope_id) as context, self._runtime._locked(self.scope_id):
            service = context.artifacts.memory
            memory, entries = await service.head_entries(context.artifacts.memory_artifact_id)
            entry = next((item for item in entries if item.entry_id == reference.entry_id), None)
            if entry is None:
                raise SourceLifecycleError(f"memory-entry-missing:{node_id or reference.entry_id}")
            if active:
                await service.reactivate(memory, entries=(entry,), reason=_REASON)
            else:
                await service.forget(memory, entries=(entry,), reason=_REASON)


def _dump_reference(reference: LifecycleReference) -> bytes:
    return json.dumps(reference.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode("utf-8")


def _load_reference(kind: str, payload: object) -> LifecycleReference:
    data = json.loads(bytes(payload))
    if kind == "source":
        return SourceRef.model_validate(data)
    if kind == "memory":
        return MemoryCitation.model_validate(data)
    return ArtifactRef.model_validate(data)


async def unavailable_topic_artifacts(runtime: BuiltinRuntime, scope_id: str) -> frozenset[str]:
    """Topic Memory artifact ids the Scope's lifecycle graph currently reports as unsupported."""

    graphs = getattr(runtime, "_lifecycle_graphs", None)
    cached = None if graphs is None else graphs.get(scope_id)
    if cached is not None:
        return cached.unavailable_topics()
    return (await ScopedSourceLifecycle(runtime, scope_id)._read(scope_id)).unavailable_topics()


def _require_identifier(node_id: str) -> None:
    if not isinstance(node_id, str) or not node_id.strip():
        raise SourceLifecycleError("node-id-must-be-non-blank")


def _require_parents(
    node_id: str,
    parents: Sequence[str],
    policy: DependencyPolicy,
) -> tuple[str, ...]:
    if policy not in _POLICIES:
        raise SourceLifecycleError("unknown-policy")
    resolved = tuple(parents)
    if not resolved:
        raise SourceLifecycleError("parent-required")
    if any(not isinstance(parent, str) or not parent.strip() for parent in resolved):
        raise SourceLifecycleError("parent-must-be-non-blank")
    if len(set(resolved)) != len(resolved):
        raise SourceLifecycleError("duplicate-parent")
    if node_id in resolved:
        raise SourceLifecycleError("self-dependency")
    return resolved


def _same_registration(left: _LifecycleNode, right: _LifecycleNode) -> bool:
    return (
        left.kind == right.kind
        and left.reference == right.reference
        and left.parents == right.parents
        and left.policy == right.policy
        and left.revision == right.revision
    )

