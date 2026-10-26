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

"""Prepare final, bounded context from approved Artifact evidence."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import chain
from math import log
from typing import TypeVar

from powercontext.artifacts import ArtifactAddress, ArtifactRef
from powercontext.builtin.artifacts.experience import Experience, ExperienceSearchHit, render_experience
from powercontext.builtin.artifacts.memory.models import MemoryCitation, MemoryHit
from powercontext.builtin.artifacts.profile.models import Profile
from powercontext.builtin.artifacts.topic_memory import TopicMemory, TopicMemorySearchHit
from powercontext.builtin.runtime.errors import PreparedContextInvariantError
from powercontext.builtin.runtime.models import PrepareContextRequest, PreparedContext
from powercontext.builtin.runtime.prepared_code import (
    CodeEvidenceRef,
    PreparedCodeCandidate,
    assemble_code,
)
from powercontext.builtin.runtime.prepared_text import (
    TRUST_POLICY,
    ContextTextItem,
    fit_context_text_item,
    render_context_text,
)
from powercontext.builtin.runtime.recall_sufficiency import RecallBudgetView

_MIN_TRUNCATED_CONTENT_BYTES = 64
_ELLIPSIS = "…"
_BEGIN_MARKER = "BEGIN_POWERCONTEXT_PREPARED_CONTEXT_V1"
_END_MARKER = "END_POWERCONTEXT_PREPARED_CONTEXT_V1"
_Item = TypeVar("_Item")
# Evidence selection keeps whole sentences, so a fragment is capped well below the request budget;
# only a single over-long sentence is ever broken further, and only at whitespace.
_MAX_FRAGMENT_BYTES = 480
_FRAGMENT_TOKEN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?。！？])[ \t\r\f\v]+|\n+")
_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class _PreparedContextEntry:
    origin: PreparedContextOrigin
    kind: str
    citation: dict[str, object]
    content: str
    truncated: bool


@dataclass(frozen=True)
class _EntryFit:
    """One fit attempt on the non-assembly path: the entry that fitted, or why none did.

    The two drop reasons are disjoint and mirror the assembly path's
    :class:`~powercontext.builtin.runtime.prepared_text.FitOutcome`. An entry is dropped
    *below the minimum truncated content* when a shorter rendering exists and fits but is too
    short to be an honest delivery; it is dropped *with no fitting truncation* when no
    shortened rendering fits at all.
    """

    entry: _PreparedContextEntry | None = None
    dropped_below_min_bytes: bool = False
    dropped_no_fitting_truncation: bool = False


@dataclass(frozen=True)
class PreparedContextOmissions:
    """Aggregate counts of the items the byte budget omitted in one build.

    Every count is a per-call aggregate, never per-entry attribution and never a verdict about
    an entry: an entry losing to the budget is not a negative result about that entry.

    ``dropped_items`` is the sum of its two sub-counts, so "the budget could not fit this
    item" can be told apart from "this item was too short to truncate into the remaining
    space". A single merged counter would misreport the second cause as the first.
    """

    truncated_items: int = 0
    dropped_items: int = 0
    dropped_below_min_bytes: int = 0
    dropped_no_fitting_truncation: int = 0


@dataclass(frozen=True)
class PreparedContextBuild:
    """Final public context and the exact origins selected to produce it.

    ``omissions`` is always filled by the Builder. The RFC 1560 recall trace is deliberately
    **not** a field here: ``ScopedContextApplication._prepare`` returns ``build.context`` and
    discards the rest, so such a field would have no production observer. The trace is
    delivered through the Runtime's optional ``RecallEffortSink`` instead.
    """

    context: PreparedContext
    origins: tuple[PreparedContextOrigin, ...]
    omissions: PreparedContextOmissions = PreparedContextOmissions()
    code_origins: tuple[CodeEvidenceRef, ...] = ()
    fragments: tuple[MemoryFragment, ...] = ()


@dataclass(frozen=True)
class MemoryEntryAddress:
    """Identify one exact Memory entry version across Scope boundaries."""

    memory: ArtifactAddress
    entry_id: str
    entry_version_id: str


PreparedContextOrigin = MemoryCitation | ArtifactRef | MemoryEntryAddress | ArtifactAddress


@dataclass(frozen=True)
class MemoryFragment:
    """One literal slice of a Memory entry, addressed back to the exact version it came from.

    ``text`` is always ``entry.text[start:end]`` for the addressed entry version: evidence is
    quoted, never rewritten, so a caller can re-read the source and find the same characters at
    the same half-open character coordinates.
    """

    address: MemoryEntryAddress
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class PreparedMemoryCandidates:
    """Memory candidates read from one Scope."""

    scope_id: str
    memory_ref: ArtifactRef | None = None
    hits: tuple[MemoryHit, ...] = ()


@dataclass(frozen=True)
class PreparedExperienceCandidates:
    """Experience candidates read from one Scope."""

    scope_id: str
    hits: tuple[ExperienceSearchHit, ...] = ()


@dataclass(frozen=True)
class PreparedProfileCandidate:
    """The latest committed Profile snapshot read from one Scope."""

    scope_id: str
    profile: Profile


class PreparedContextBuilder:
    """Select and render final context without I/O, persistence, or reranking."""

    memory_candidate_limit = 16
    topic_memory_candidate_limit = 8
    experience_candidate_limit = 8
    candidate_limit = memory_candidate_limit
    entry_limit = 8
    topic_memory_entry_limit = 8
    experience_entry_limit = 2
    max_entry_content_bytes = 2000
    _text_entry_limit: int | None = None

    def empty(self) -> PreparedContext:
        return PreparedContext(status="empty", content=None, content_bytes=0)

    def build(
        self,
        *,
        request: PrepareContextRequest,
        scope_id: str | None = None,
        memory_ref: ArtifactRef | None = None,
        hits: Sequence[MemoryHit] = (),
        topic_memory_hits: Sequence[TopicMemorySearchHit] = (),
        experience_hits: Sequence[ExperienceSearchHit] = (),
    ) -> PreparedContext:
        return self.build_result(
            request=request,
            scope_id=scope_id,
            memory_ref=memory_ref,
            hits=hits,
            topic_memory_hits=topic_memory_hits,
            experience_hits=experience_hits,
        ).context

    def build_result(
        self,
        *,
        request: PrepareContextRequest,
        scope_id: str | None = None,
        memory_ref: ArtifactRef | None = None,
        hits: Sequence[MemoryHit] = (),
        topic_memory_hits: Sequence[TopicMemorySearchHit] = (),
        experience_hits: Sequence[ExperienceSearchHit] = (),
    ) -> PreparedContextBuild:
        return self.build_scopes_result(
            request=request,
            current_scope_id=scope_id,
            memory_candidates=(
                PreparedMemoryCandidates(scope_id=scope_id or "", memory_ref=memory_ref, hits=tuple(hits)),
            ),
            experience_candidates=(PreparedExperienceCandidates(scope_id=scope_id or "", hits=tuple(experience_hits)),),
            topic_memory_hits=topic_memory_hits,
        )

    def build_scopes_result(
        self,
        *,
        request: PrepareContextRequest,
        current_scope_id: str | None,
        memory_candidates: Sequence[PreparedMemoryCandidates] = (),
        topic_memory_hits: Sequence[TopicMemorySearchHit] = (),
        experience_candidates: Sequence[PreparedExperienceCandidates] = (),
        profile_candidates: Sequence[PreparedProfileCandidate] = (),
        code_candidates: Sequence[PreparedCodeCandidate] = (),
    ) -> PreparedContextBuild:
        if request.include_code:

            def historical(history_request: PrepareContextRequest, entries: int) -> PreparedContextBuild:
                builder = PreparedContextBuilder()
                builder.entry_limit = entries
                builder._text_entry_limit = entries if entries < self.entry_limit else None
                return builder.build_scopes_result(
                    request=history_request.model_copy(update={"include_code": False}),
                    current_scope_id=current_scope_id,
                    memory_candidates=memory_candidates,
                    topic_memory_hits=topic_memory_hits,
                    experience_candidates=experience_candidates,
                    profile_candidates=profile_candidates,
                )

            return assemble_code(request, code_candidates, self.entry_limit, historical)

        if sum(len(candidates.hits) for candidates in memory_candidates) > self.memory_candidate_limit:
            raise PreparedContextInvariantError("memory-candidate-limit")
        if len(topic_memory_hits) > self.topic_memory_candidate_limit:
            raise PreparedContextInvariantError("topic-memory-candidate-limit")
        if sum(len(candidates.hits) for candidates in experience_candidates) > self.experience_candidate_limit:
            raise PreparedContextInvariantError("experience-candidate-limit")

        if request.evidence_selection and request.assembly is None:
            return self._build_evidence_result(
                request=request,
                current_scope_id=current_scope_id,
                memory_candidates=memory_candidates,
            )

        content, origins, omissions = self._select_entries(
            request=request,
            current_scope_id=current_scope_id,
            memory_candidates=memory_candidates,
            experience_candidates=experience_candidates,
            profile_candidates=profile_candidates,
            topic_memory_hits=topic_memory_hits,
        )
        if content is None:
            return PreparedContextBuild(context=self.empty(), origins=(), omissions=omissions)
        content_bytes = len(content.encode("utf-8"))
        # The assembly path is bounded by construction (every fitted item is measured against
        # `max_bytes`), so the ceiling check belongs to the non-assembly renderer only.
        if request.assembly is None and content_bytes > request.max_bytes:
            raise PreparedContextInvariantError("output-budget")
        return PreparedContextBuild(
            context=PreparedContext(status="ready", content=content, content_bytes=content_bytes),
            origins=origins,
            omissions=omissions,
        )

    def probe_budget(
        self,
        *,
        request: PrepareContextRequest,
        current_scope_id: str | None,
        memory_candidates: Sequence[PreparedMemoryCandidates] = (),
        topic_memory_hits: Sequence[TopicMemorySearchHit] = (),
        experience_candidates: Sequence[PreparedExperienceCandidates] = (),
        profile_candidates: Sequence[PreparedProfileCandidate] = (),
    ) -> RecallBudgetView:
        """Report what the byte budget does to one candidate set, without delivering it.

        This is the RFC 1560 *budget probe*: one pass of this Builder's own pure selection and
        rendering code, keeping only the counters and discarding the rendered output. It lets
        the gate tell budget-limited thinness from recall-limited thinness.

        Pure and side-effect free: no I/O, no persistence, nothing reported to any sink, and
        the same inputs as :meth:`build_scopes_result` are guaranteed to yield the same
        counters because both go through :meth:`_select_entries`.
        """

        if request.evidence_selection and request.assembly is None:
            fragments = self._select_fragment_entries(
                request=request,
                current_scope_id=current_scope_id,
                memory_candidates=memory_candidates,
            )
            fragment_bytes = 0 if not fragments else len(_render_fragments(fragments).encode("utf-8"))
            return RecallBudgetView(
                max_bytes=request.max_bytes,
                delivered_items=len(fragments),
                truncated_items=0,
                dropped_items=0,
                unused_bytes=max(0, request.max_bytes - fragment_bytes),
            )

        content, origins, omissions = self._select_entries(
            request=request,
            current_scope_id=current_scope_id,
            memory_candidates=memory_candidates,
            experience_candidates=experience_candidates,
            profile_candidates=profile_candidates,
            topic_memory_hits=topic_memory_hits,
        )
        content_bytes = 0 if content is None else len(content.encode("utf-8"))
        return RecallBudgetView(
            max_bytes=request.max_bytes,
            delivered_items=len(origins),
            truncated_items=omissions.truncated_items,
            dropped_items=omissions.dropped_items,
            unused_bytes=max(0, request.max_bytes - content_bytes),
        )

    def _build_evidence_result(
        self,
        *,
        request: PrepareContextRequest,
        current_scope_id: str | None,
        memory_candidates: Sequence[PreparedMemoryCandidates],
    ) -> PreparedContextBuild:
        """Deliver query-driven literal fragments instead of whole entries.

        Only Memory candidates take part: the selection mode exists to spend a byte budget on the
        relevant part of recalled Memory text, so Code, Topic Memory and Experience keep their
        ordinary packing path (and the ordinary path keeps its behaviour untouched).
        """

        fragments = self._select_fragment_entries(
            request=request,
            current_scope_id=current_scope_id,
            memory_candidates=memory_candidates,
        )
        if not fragments:
            return PreparedContextBuild(context=self.empty(), origins=())
        content = _render_fragments(fragments)
        content_bytes = len(content.encode("utf-8"))
        if content_bytes > request.max_bytes:
            raise PreparedContextInvariantError("output-budget")
        return PreparedContextBuild(
            context=PreparedContext(status="ready", content=content, content_bytes=content_bytes),
            origins=tuple(fragment.address for fragment in fragments),
            fragments=fragments,
        )

    def _select_fragment_entries(
        self,
        *,
        request: PrepareContextRequest,
        current_scope_id: str | None,
        memory_candidates: Sequence[PreparedMemoryCandidates],
    ) -> tuple[MemoryFragment, ...]:
        """Choose literal Memory slices for ``request.query`` inside ``request.max_bytes``.

        Every sentence of a candidate is a possible fragment. Selection runs in three passes so a
        tight budget is spent on evidence rather than on whatever the retriever happened to rank
        first:

        1. query coverage — repeatedly take the fragment that adds the most *uncovered* query
           weight, so a rare query term late in a document still wins a slot;
        2. query relevance — spend the remainder on further fragments that mention the query;
        3. adjacency — add the sentences next to an already chosen one, because a quoted evidence
           passage is often split across a sentence boundary.

        The budget is measured on the finished envelope (trust policy, markers, JSON and
        citations included), never on the fragment text alone.
        """

        candidates = _fragment_candidates(current_scope_id, memory_candidates)
        if not candidates:
            return ()
        query_tokens = _fragment_tokens(request.query)
        if not query_tokens:
            return ()
        weights = _fragment_weights(query_tokens, candidates)
        scores = tuple(_fragment_score(candidate.tokens, weights) for candidate in candidates)
        ranked = sorted(range(len(candidates)), key=lambda index: (-scores[index], index))
        max_bytes = request.max_bytes
        chosen: set[int] = set()
        selected: list[int] = []

        uncovered = dict(weights)
        while uncovered:
            best = -1
            best_gain = 0.0
            for index in ranked:
                if index in chosen:
                    continue
                gain = _fragment_score(candidates[index].tokens, uncovered)
                if gain <= best_gain:
                    continue
                if not _fragments_fit(candidates, (*selected, index), max_bytes):
                    # Too large for what is left now, so it can never fit later either.
                    chosen.add(index)
                    continue
                best = index
                best_gain = gain
            if best < 0:
                break
            selected.append(best)
            chosen.add(best)
            uncovered = {
                token: weight for token, weight in uncovered.items() if token not in candidates[best].tokens
            }

        for index in ranked:
            if index in chosen or scores[index] <= 0.0:
                continue
            if _fragments_fit(candidates, (*selected, index), max_bytes):
                selected.append(index)
                chosen.add(index)

        groups = _fragment_groups(candidates)
        for index in tuple(selected):
            siblings = groups[_address_key(candidates[index].address)]
            position = siblings.index(index)
            for offset in (-1, 1):
                neighbour = position + offset
                if not 0 <= neighbour < len(siblings):
                    continue
                adjacent = siblings[neighbour]
                if adjacent in chosen:
                    continue
                if _fragments_fit(candidates, (*selected, adjacent), max_bytes):
                    selected.append(adjacent)
                    chosen.add(adjacent)

        if not selected:
            for index in range(len(candidates)):
                if _fragments_fit(candidates, (*selected, index), max_bytes):
                    selected.append(index)
                    chosen.add(index)

        # Merging contiguous spans releases bytes the earlier passes could not use,
        # so spend the remainder on further query-relevant sentences.
        for index in ranked:
            if index in chosen:
                continue
            if _fragments_fit(candidates, (*selected, index), max_bytes):
                selected.append(index)
                chosen.add(index)

        return _merge_selected(candidates, selected)

    def _select_entries(
        self,
        *,
        request: PrepareContextRequest,
        current_scope_id: str | None,
        memory_candidates: Sequence[PreparedMemoryCandidates],
        topic_memory_hits: Sequence[TopicMemorySearchHit],
        experience_candidates: Sequence[PreparedExperienceCandidates],
        profile_candidates: Sequence[PreparedProfileCandidate],
    ) -> tuple[str | None, tuple[PreparedContextOrigin, ...], PreparedContextOmissions]:
        """Run one complete pure selection pass and render it.

        Returns ``(content, origins, omissions)``; ``content`` is ``None`` exactly when nothing
        was selected, which is the empty-result case for both paths. This is the single
        selection path shared by :meth:`build_scopes_result` and :meth:`probe_budget`, so the
        probe's counters always describe the selection the build would actually perform.
        """

        if request.assembly is not None:
            return self._select_text(
                request,
                current_scope_id=current_scope_id,
                memory_candidates=memory_candidates,
                experience_candidates=experience_candidates,
                profile_candidates=profile_candidates,
                topic_memory_hits=topic_memory_hits,
            )

        memory_entries = _interleave_groups(
            tuple(
                self._memory_entries(
                    candidates.memory_ref,
                    candidates.hits,
                    scope_id=None if candidates.scope_id == current_scope_id else candidates.scope_id or None,
                )
                for candidates in memory_candidates
            )
        )
        topic_memory_entries = self._topic_memory_entries(topic_memory_hits)
        experience_entries = _interleave_groups(
            tuple(
                self._experience_entries(
                    candidates.hits,
                    scope_id=None if candidates.scope_id == current_scope_id else candidates.scope_id or None,
                )
                for candidates in experience_candidates
            )
        )[: self.experience_entry_limit]
        entries, omissions = self._fit_entries(request, memory_entries, topic_memory_entries, experience_entries)
        if not entries:
            return None, (), omissions
        return _render(entries), tuple(entry.origin for entry in entries), omissions

    def _select_text(
        self,
        request: PrepareContextRequest,
        *,
        current_scope_id: str | None,
        memory_candidates: Sequence[PreparedMemoryCandidates],
        experience_candidates: Sequence[PreparedExperienceCandidates],
        profile_candidates: Sequence[PreparedProfileCandidate],
        topic_memory_hits: Sequence[TopicMemorySearchHit],
    ) -> tuple[str | None, tuple[PreparedContextOrigin, ...], PreparedContextOmissions]:
        assembly = request.assembly
        if assembly is None:
            raise PreparedContextInvariantError("text-assembly-missing")
        included: list[ContextTextItem] = []
        origins: list[PreparedContextOrigin] = []
        truncated_items = 0
        dropped_below_min_bytes = 0
        dropped_no_fitting_truncation = 0
        for section in assembly.sections:
            # Unfilled earlier sections leave their capacity available to later ones.
            limit = (
                section.limit
                if self._text_entry_limit is None
                else min(section.limit, self._text_entry_limit - len(included))
            )
            if section.family == "profile":
                entries = self._profile_entries(profile_candidates)
            elif section.family == "topic-memory":
                entries = self._topic_memory_entries(topic_memory_hits, scope_id=current_scope_id)
            else:
                groups = (
                    tuple(
                        tuple(
                            self._memory_entries(group.memory_ref, (hit,), scope_id=group.scope_id)
                            for hit in group.hits
                        )
                        for group in memory_candidates
                    )
                    if section.family == "memory"
                    else tuple(
                        tuple(self._experience_entries((hit,), scope_id=group.scope_id) for hit in group.hits)
                        for group in experience_candidates
                    )
                )
                entries = tuple(chain.from_iterable(_interleave_groups(groups)))
            seen: set[tuple[str, str, str, int, str | None, str | None]] = set()
            rank = 0
            selected_count = 0
            for entry in entries:
                if selected_count >= limit:
                    break
                item = _text_item(entry)
                artifact = item.artifact
                identity = (
                    artifact.scope_id,
                    artifact.artifact.family,
                    artifact.artifact.artifact_id,
                    artifact.artifact.revision,
                    item.entry_id,
                    item.entry_version_id,
                )
                if identity in seen or not item.content.strip():
                    continue
                seen.add(identity)
                rank += 1
                outcome = fit_context_text_item(included, replace(item, recall_rank=rank), assembly, request.max_bytes)
                if outcome.item is None:
                    dropped_below_min_bytes += int(outcome.dropped_below_min_bytes)
                    dropped_no_fitting_truncation += int(outcome.dropped_no_fitting_truncation)
                    continue
                fitted = outcome.item
                included.append(fitted)
                origins.append(entry.origin)
                selected_count += 1
                truncated_items += int(fitted.truncated)
        omissions = PreparedContextOmissions(
            truncated_items=truncated_items,
            dropped_items=dropped_below_min_bytes + dropped_no_fitting_truncation,
            dropped_below_min_bytes=dropped_below_min_bytes,
            dropped_no_fitting_truncation=dropped_no_fitting_truncation,
        )
        if not included:
            return None, (), omissions
        content = render_context_text(included, assembly)
        return content, tuple(origins), omissions

    def _profile_entries(self, candidates: Sequence[PreparedProfileCandidate]) -> tuple[_PreparedContextEntry, ...]:
        entries = []
        for candidate in candidates:
            origin = ArtifactAddress(scope_id=candidate.scope_id, artifact=candidate.profile.as_ref())
            entries.append(
                _PreparedContextEntry(
                    origin=origin,
                    kind="profile",
                    citation={"artifact": origin.model_dump(mode="json")},
                    content=candidate.profile.content.content,
                    truncated=False,
                )
            )
        return tuple(entries)

    def _memory_entries(
        self,
        memory_ref: ArtifactRef | None,
        hits: Sequence[MemoryHit],
        *,
        scope_id: str | None = None,
    ) -> tuple[_PreparedContextEntry, ...]:
        if hits and memory_ref is None:
            raise PreparedContextInvariantError("memory-ref-missing")
        memory_entries: list[_PreparedContextEntry] = []
        seen: set[tuple[str, str]] = set()
        for hit in hits:
            if hit.memory_ref != memory_ref:
                raise PreparedContextInvariantError("memory-ref-mismatch")

            citation_key = (hit.entry_id, hit.entry_version_id)
            if citation_key in seen:
                continue
            seen.add(citation_key)
            if not hit.entry_id.strip() or not hit.entry_version_id.strip() or not hit.text.strip():
                continue
            citation = MemoryCitation(
                memory_ref=hit.memory_ref,
                entry_id=hit.entry_id,
                entry_version_id=hit.entry_version_id,
            )
            origin: PreparedContextOrigin = citation
            rendered_citation = citation.model_dump(mode="json")
            if scope_id is not None:
                memory = ArtifactAddress(scope_id=scope_id, artifact=hit.memory_ref)
                origin = MemoryEntryAddress(
                    memory=memory,
                    entry_id=hit.entry_id,
                    entry_version_id=hit.entry_version_id,
                )
                rendered_citation = {
                    "memory": memory.model_dump(mode="json"),
                    "entry_id": hit.entry_id,
                    "entry_version_id": hit.entry_version_id,
                }
            memory_entries.append(
                _PreparedContextEntry(
                    origin=origin,
                    kind="memory",
                    citation=rendered_citation,
                    content=hit.text,
                    truncated=False,
                )
            )
        return tuple(memory_entries)

    def _topic_memory_entries(
        self,
        hits: Sequence[TopicMemorySearchHit],
        *,
        scope_id: str | None = None,
    ) -> tuple[_PreparedContextEntry, ...]:
        topic_entries: list[_PreparedContextEntry] = []
        seen_topics: set[tuple[str, int]] = set()
        for hit in hits:
            if hit.artifact_ref.family != TopicMemory.family:
                raise PreparedContextInvariantError("topic-memory-family-mismatch")
            identity = (hit.artifact_ref.artifact_id, hit.artifact_ref.revision)
            if identity in seen_topics:
                continue
            seen_topics.add(identity)
            if len(topic_entries) >= self.topic_memory_entry_limit:
                break
            content = {"title": hit.title, "summary": hit.summary}
            if hit.snippet is not None:
                content["snippet"] = hit.snippet
            origin: PreparedContextOrigin = hit.artifact_ref
            citation = {"artifact_ref": hit.artifact_ref.model_dump(mode="json")}
            body = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
            if scope_id is not None:
                origin = ArtifactAddress(scope_id=scope_id, artifact=hit.artifact_ref)
                citation = {"artifact": origin.model_dump(mode="json")}
                body = "\n\n".join(f"{label.capitalize()}: {value}" for label, value in content.items())
            topic_entries.append(
                _PreparedContextEntry(
                    origin=origin,
                    kind="topic-memory",
                    citation=citation,
                    content=body,
                    truncated=False,
                )
            )
        return tuple(topic_entries)

    def _experience_entries(
        self,
        hits: Sequence[ExperienceSearchHit],
        *,
        scope_id: str | None = None,
    ) -> tuple[_PreparedContextEntry, ...]:
        experience_entries: list[_PreparedContextEntry] = []
        seen_experiences: set[tuple[str, int]] = set()
        for hit in hits:
            if hit.artifact_ref.family != Experience.family:
                raise PreparedContextInvariantError("experience-family-mismatch")
            identity = (hit.artifact_ref.artifact_id, hit.artifact_ref.revision)
            if identity in seen_experiences:
                continue
            seen_experiences.add(identity)
            origin: PreparedContextOrigin = hit.artifact_ref
            rendered_citation: dict[str, object] = {"artifact_ref": hit.artifact_ref.model_dump(mode="json")}
            if scope_id is not None:
                origin = ArtifactAddress(scope_id=scope_id, artifact=hit.artifact_ref)
                rendered_citation = {"artifact": origin.model_dump(mode="json")}
            experience_entries.append(
                _PreparedContextEntry(
                    origin=origin,
                    kind="experience",
                    citation=rendered_citation,
                    content=render_experience(hit.content),
                    truncated=False,
                )
            )
        return tuple(experience_entries)

    def _fit_entries(
        self,
        request: PrepareContextRequest,
        memory_entries: Sequence[_PreparedContextEntry],
        topic_memory_entries: Sequence[_PreparedContextEntry],
        experience_entries: Sequence[_PreparedContextEntry],
    ) -> tuple[tuple[_PreparedContextEntry, ...], PreparedContextOmissions]:
        entries: list[_PreparedContextEntry] = []
        truncated_items = 0
        dropped_below_min_bytes = 0
        dropped_no_fitting_truncation = 0
        for candidate in _interleave(memory_entries, topic_memory_entries, experience_entries):
            if len(entries) >= self.entry_limit:
                break
            fit = self._fit_entry(
                entries,
                origin=candidate.origin,
                kind=candidate.kind,
                citation=candidate.citation,
                text=candidate.content,
                max_bytes=request.max_bytes,
            )
            if fit.entry is None:
                dropped_below_min_bytes += int(fit.dropped_below_min_bytes)
                dropped_no_fitting_truncation += int(fit.dropped_no_fitting_truncation)
                continue
            entries.append(fit.entry)
            if fit.entry.truncated:
                truncated_items += 1
        return tuple(entries), PreparedContextOmissions(
            truncated_items=truncated_items,
            dropped_items=dropped_below_min_bytes + dropped_no_fitting_truncation,
            dropped_below_min_bytes=dropped_below_min_bytes,
            dropped_no_fitting_truncation=dropped_no_fitting_truncation,
        )

    def _fit_entry(
        self,
        entries: Sequence[_PreparedContextEntry],
        *,
        origin: PreparedContextOrigin,
        kind: str,
        citation: dict[str, object],
        text: str,
        max_bytes: int,
    ) -> _EntryFit:
        """Fit one entry into the remaining budget, reporting why it was dropped if it was.

        The two ``None`` paths of the original code are kept apart: ``below-min-bytes`` means a
        shorter rendering exists and fits but is too short to deliver honestly, while
        ``no-fitting-truncation`` means no shortened rendering fitted at all.
        """

        source_bytes = len(text.encode("utf-8"))
        entry_budget = min(source_bytes, self.max_entry_content_bytes)
        candidate = _PreparedContextEntry(
            origin=origin,
            kind=kind,
            citation=citation,
            content=text if source_bytes <= entry_budget else _truncate_utf8(text, entry_budget),
            truncated=source_bytes > entry_budget,
        )
        if _rendered_bytes((*entries, candidate)) <= max_bytes:
            return _EntryFit(entry=candidate)
        if source_bytes < _MIN_TRUNCATED_CONTENT_BYTES:
            return _EntryFit(dropped_below_min_bytes=True)

        lower = _MIN_TRUNCATED_CONTENT_BYTES
        upper = min(entry_budget, source_bytes - 1)
        best: _PreparedContextEntry | None = None
        while lower <= upper:
            byte_budget = (lower + upper) // 2
            candidate = _PreparedContextEntry(
                origin=origin,
                kind=kind,
                citation=citation,
                content=_truncate_utf8(text, byte_budget),
                truncated=True,
            )
            if len(candidate.content.encode("utf-8")) < _MIN_TRUNCATED_CONTENT_BYTES:
                lower = byte_budget + 1
                continue
            if _rendered_bytes((*entries, candidate)) <= max_bytes:
                best = candidate
                lower = byte_budget + 1
            else:
                upper = byte_budget - 1
        if best is None:
            return _EntryFit(dropped_no_fitting_truncation=True)
        return _EntryFit(entry=best)


def _interleave(
    memory: Sequence[_PreparedContextEntry],
    topics: Sequence[_PreparedContextEntry],
    experiences: Sequence[_PreparedContextEntry],
) -> tuple[_PreparedContextEntry, ...]:
    """Preserve each family's rank while alternating Memory, Topic, Experience."""

    ordered: list[_PreparedContextEntry] = []
    for index in range(max(len(memory), len(topics), len(experiences))):
        if index < len(memory):
            ordered.append(memory[index])
        if index < len(topics):
            ordered.append(topics[index])
        if index < len(experiences):
            ordered.append(experiences[index])
    return tuple(ordered)


def _interleave_groups(groups: Sequence[Sequence[_Item]]) -> tuple[_Item, ...]:
    ordered: list[_Item] = []
    for index in range(max((len(group) for group in groups), default=0)):
        ordered.extend(group[index] for group in groups if index < len(group))
    return tuple(ordered)


def _render(entries: Sequence[_PreparedContextEntry]) -> str:
    envelope = {
        "trust": "untrusted_history",
        "items": [_render_entry(entry) for entry in entries],
    }
    encoded = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
    return "\n\n".join((TRUST_POLICY, f"{_BEGIN_MARKER}\n{encoded}\n{_END_MARKER}"))


@dataclass(frozen=True)
class _FragmentCandidate:
    """One quotable sentence of a candidate entry, with its exact character span."""

    address: MemoryEntryAddress
    start: int
    end: int
    text: str
    tokens: frozenset[str]
    entry_text: str = ""


def _fragment_tokens(text: str) -> frozenset[str]:
    """Lower-case word and number tokens, plus CJK characters as single-token evidence."""

    return frozenset(_FRAGMENT_TOKEN.findall(text.lower()))


def _fragment_weights(query_tokens: frozenset[str], candidates: Sequence[_FragmentCandidate]) -> dict[str, float]:
    """Inverse document frequency of each query token over the candidate sentences.

    A term that appears in one short evidence sentence out of forty candidates is a far stronger
    signal than one that appears in all of them, so rarity is what earns a slot in the budget.
    """

    frequency: dict[str, int] = {}
    for candidate in candidates:
        for token in query_tokens & candidate.tokens:
            frequency[token] = frequency.get(token, 0) + 1
    total = len(candidates)
    return {token: log(1.0 + total / (1.0 + count)) for token, count in frequency.items()}


def _fragment_score(tokens: frozenset[str], weights: dict[str, float]) -> float:
    return sum(weight for token, weight in weights.items() if token in tokens)


def _address_key(address: MemoryEntryAddress) -> tuple[str, str, str]:
    return (address.memory.scope_id, address.entry_id, address.entry_version_id)


def _fragment_groups(candidates: Sequence[_FragmentCandidate]) -> dict[tuple[str, str, str], tuple[int, ...]]:
    grouped: dict[tuple[str, str, str], list[int]] = {}
    for index, candidate in enumerate(candidates):
        grouped.setdefault(_address_key(candidate.address), []).append(index)
    return {key: tuple(indexes) for key, indexes in grouped.items()}


def _fragment_candidates(
    current_scope_id: str | None,
    memory_candidates: Sequence[PreparedMemoryCandidates],
) -> tuple[_FragmentCandidate, ...]:
    """Cut every candidate entry into sentence-sized spans that stay inside one entry version."""

    candidates: list[_FragmentCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for group in memory_candidates:
        scope_id = group.scope_id or current_scope_id or ""
        for hit in group.hits:
            key = (scope_id, hit.entry_id, hit.entry_version_id)
            if key in seen or not hit.text.strip():
                continue
            seen.add(key)
            address = MemoryEntryAddress(
                memory=ArtifactAddress(scope_id=scope_id, artifact=hit.memory_ref),
                entry_id=hit.entry_id,
                entry_version_id=hit.entry_version_id,
            )
            for start, end in _entry_spans(hit.text):
                for span_start, span_end in _split_long_span(hit.text, start, end):
                    text = hit.text[span_start:span_end]
                    candidates.append(
                        _FragmentCandidate(
                            address=address,
                            start=span_start,
                            end=span_end,
                            text=text,
                            tokens=_fragment_tokens(text),
                            entry_text=hit.text,
                        )
                    )
    return tuple(candidates)


def _entry_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Split text at sentence boundaries into gap-free half-open character spans.

    Each span ends where the next one starts, so the whitespace between two sentences belongs to
    the earlier span. Selecting consecutive sentences therefore covers the original text with no
    one-character holes, which is what "the merged intervals cover the evidence intervals" needs.
    """

    starts = [0]
    for match in _SENTENCE_BREAK.finditer(text):
        starts.append(match.end())
    spans: list[tuple[int, int]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        if index == 0:
            start += len(text[start:end]) - len(text[start:end].lstrip())
        if start < end and text[start:end].strip():
            spans.append((start, end))
    return tuple(spans)


def _split_long_span(text: str, start: int, end: int) -> tuple[tuple[int, int], ...]:
    """Break one over-long span at whitespace so a single fragment can always fit a budget."""

    if len(text[start:end].encode("utf-8")) <= _MAX_FRAGMENT_BYTES:
        return ((start, end),)
    starts: list[int] = []
    chunk_start = -1
    for match in _WORD.finditer(text, start, end):
        word_start, word_end = match.start(), match.end()
        if chunk_start < 0:
            chunk_start = word_start
        elif len(text[chunk_start:word_end].encode("utf-8")) > _MAX_FRAGMENT_BYTES:
            starts.append(chunk_start)
            chunk_start = word_start
    if chunk_start >= 0:
        starts.append(chunk_start)
    return tuple(
        (span_start, starts[index + 1] if index + 1 < len(starts) else end)
        for index, span_start in enumerate(starts)
    )


def _fragment(candidate: _FragmentCandidate) -> MemoryFragment:
    return MemoryFragment(
        address=candidate.address,
        start=candidate.start,
        end=candidate.end,
        text=candidate.text,
    )


def _fragment_item(fragment: MemoryFragment) -> dict[str, object]:
    return {
        "citation": {
            "scope_id": fragment.address.memory.scope_id,
            "memory_ref": fragment.address.memory.artifact.model_dump(mode="json"),
            "entry_id": fragment.address.entry_id,
            "entry_version_id": fragment.address.entry_version_id,
        },
        "content": fragment.text,
        "start": fragment.start,
        "end": fragment.end,
    }


def _render_fragments(fragments: Sequence[MemoryFragment]) -> str:
    envelope = {
        "trust": "untrusted_history",
        "items": [_fragment_item(fragment) for fragment in fragments],
    }
    encoded = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
    return "\n\n".join((TRUST_POLICY, f"{_BEGIN_MARKER}\n{encoded}\n{_END_MARKER}"))


def _merge_selected(
    candidates: Sequence[_FragmentCandidate],
    indexes: Sequence[int],
) -> tuple[MemoryFragment, ...]:
    """Coalesce contiguous selected spans so one citation carries the whole passage.

    Every emitted item repeats its full citation, and a citation costs roughly as
    much as a short sentence. Splitting one passage across several fragments spends
    the byte budget on identity instead of on evidence, so touching spans of the
    same entry are merged back into a single literal slice.
    """

    grouped: dict[tuple[str, str, str], list[int]] = {}
    order: list[tuple[str, str, str]] = []
    for index in indexes:
        key = _address_key(candidates[index].address)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(index)

    fragments: list[MemoryFragment] = []
    for key in order:
        spans = sorted(grouped[key], key=lambda index: candidates[index].start)
        first = candidates[spans[0]]
        start, end = first.start, first.end
        for index in spans[1:]:
            candidate = candidates[index]
            if candidate.start <= end:
                end = max(end, candidate.end)
                continue
            fragments.append(
                MemoryFragment(address=first.address, start=start, end=end, text=first.entry_text[start:end])
            )
            start, end = candidate.start, candidate.end
        fragments.append(
            MemoryFragment(address=first.address, start=start, end=end, text=first.entry_text[start:end])
        )
    return tuple(fragments)


def _fragments_fit(
    candidates: Sequence[_FragmentCandidate],
    indexes: Sequence[int],
    max_bytes: int,
) -> bool:
    rendered = _render_fragments(_merge_selected(candidates, indexes))
    return len(rendered.encode("utf-8")) <= max_bytes


def _text_item(entry: _PreparedContextEntry) -> ContextTextItem:
    origin = entry.origin
    if isinstance(origin, MemoryEntryAddress):
        return ContextTextItem(
            artifact=origin.memory,
            content=entry.content,
            recall_rank=0,
            entry_id=origin.entry_id,
            entry_version_id=origin.entry_version_id,
        )
    if isinstance(origin, ArtifactAddress):
        return ContextTextItem(artifact=origin, content=entry.content, recall_rank=0)
    raise PreparedContextInvariantError("text-scope-missing")


def _render_entry(entry: _PreparedContextEntry) -> dict[str, object]:
    rendered: dict[str, object] = {
        "citation": entry.citation,
        "content": entry.content,
        "truncated": entry.truncated,
    }
    if entry.kind != "memory":
        rendered["kind"] = entry.kind
    return rendered


def _rendered_bytes(entries: Sequence[_PreparedContextEntry]) -> int:
    return len(_render(entries).encode("utf-8"))


def _truncate_utf8(text: str, byte_budget: int) -> str:
    prefix_budget = byte_budget - len(_ELLIPSIS.encode("utf-8"))
    encoded_prefix = text.encode("utf-8")[:prefix_budget]
    return f"{encoded_prefix.decode('utf-8', errors='ignore')}{_ELLIPSIS}"
