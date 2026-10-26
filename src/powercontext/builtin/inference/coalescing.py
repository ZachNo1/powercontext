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

"""Share in-flight embedding work across calls and cache successful vectors."""

from __future__ import annotations

import asyncio
from collections import OrderedDict

from powercontext.builtin.artifacts.memory import EmbeddingProfile
from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel

DEFAULT_MAX_BATCH_SIZE = 64
DEFAULT_COALESCE_MS = 5.0
DEFAULT_CACHE_SIZE = 1024

_BatchOutcome = tuple[tuple[EmbeddingVector, ...], InferenceUsage]


class CoalescingEmbeddingModel:
    """Wrap an embedding model with request coalescing and a bounded LRU cache.

    Every call first reuses cached vectors, then joins work that is already in
    flight for the same text, and only creates new provider batches for what is
    still missing. Each batch runs in its own task so that cancelling one waiter
    never cancels work other callers depend on.
    """

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        coalesce_ms: float = DEFAULT_COALESCE_MS,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")  # noqa: TRY003
        if coalesce_ms < 0:
            raise ValueError("coalesce_ms must not be negative")  # noqa: TRY003
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")  # noqa: TRY003
        self._model = model
        self.profile: EmbeddingProfile = model.profile
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = coalesce_ms / 1000.0
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._inflight: dict[str, tuple[asyncio.Task[_BatchOutcome], int]] = {}

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Return one vector per input position, reusing cache and shared work."""

        ordered = tuple(texts)
        if not ordered:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        unique = list(dict.fromkeys(ordered))
        cached: dict[str, EmbeddingVector] = {}
        waiting: dict[str, tuple[asyncio.Task[_BatchOutcome], int]] = {}
        missing: list[str] = []

        for text in unique:
            hit = self._cached(text)
            if hit is not None:
                cached[text] = hit
                continue
            pending = self._inflight.get(text)
            if pending is not None:
                waiting[text] = pending
                continue
            missing.append(text)

        for start in range(0, len(missing), self._max_batch_size):
            chunk = missing[start : start + self._max_batch_size]
            task: asyncio.Task[_BatchOutcome] = asyncio.create_task(self._run_batch(chunk))
            task.add_done_callback(_consume_outcome)
            for index, text in enumerate(chunk):
                self._inflight[text] = (task, index)
                waiting[text] = (task, index)

        batches = {id(task): task for task, _ in waiting.values()}
        if batches:
            outcomes = await asyncio.gather(
                *(asyncio.shield(task) for task in batches.values()),
                return_exceptions=True,
            )
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome

        vectors: list[EmbeddingVector] = []
        for text in ordered:
            if text in cached:
                vectors.append(cached[text])
                continue
            task, index = waiting[text]
            vectors.append(task.result()[0][index])

        return EmbeddingResult(vectors=tuple(vectors), usage=self._usage_for(batches.values()))

    async def _run_batch(self, chunk: list[str]) -> _BatchOutcome:
        try:
            if self._coalesce_seconds > 0:
                await asyncio.sleep(self._coalesce_seconds)
            result = await self._model.embed(tuple(chunk))
            if len(result.vectors) != len(chunk):
                raise ValueError("embedding provider returned the wrong number of vectors")  # noqa: TRY003
            outcome: _BatchOutcome = (tuple(result.vectors), result.usage)
        except BaseException:
            for text in chunk:
                self._inflight.pop(text, None)
            raise
        for text, vector in zip(chunk, outcome[0], strict=True):
            self._remember(text, vector)
            self._inflight.pop(text, None)
        return outcome

    def _usage_for(self, tasks: object) -> InferenceUsage:
        requests = 0
        input_tokens: int | None = None
        output_tokens: int | None = None
        for task in tasks:  # type: ignore[union-attr]
            if task.cancelled() or task.exception() is not None:
                continue
            usage = task.result()[1]
            requests += usage.requests
            input_tokens = _add(input_tokens, usage.input_tokens)
            output_tokens = _add(output_tokens, usage.output_tokens)
        return InferenceUsage(requests=requests, input_tokens=input_tokens, output_tokens=output_tokens)

    def _cached(self, text: str) -> EmbeddingVector | None:
        if self._cache_size <= 0:
            return None
        vector = self._cache.get(text)
        if vector is None:
            return None
        self._cache.move_to_end(text)
        return vector

    def _remember(self, text: str, vector: EmbeddingVector) -> None:
        if self._cache_size <= 0:
            return
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)


def _add(left: int | None, right: int | None) -> int | None:
    if left is None and right is None:
        return None
    return (left or 0) + (right or 0)


def _consume_outcome(task: asyncio.Task[_BatchOutcome]) -> None:
    """Mark a batch outcome as observed so abandoned work stays quiet."""

    if not task.cancelled():
        task.exception()


__all__ = [
    "DEFAULT_CACHE_SIZE",
    "DEFAULT_COALESCE_MS",
    "DEFAULT_MAX_BATCH_SIZE",
    "CoalescingEmbeddingModel",
]
