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

"""HTTP cross-encoder reranking over a coarse Memory candidate pool."""

from __future__ import annotations

import math
from typing import Any

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import (
    MemoryRerankDecision,
    _validate_bounds,
)
from powercontext.builtin.inference import InferenceUsage

DEFAULT_HTTP_RERANK_POLICY_ID = "http-cross-encoder-v1"
_SCORE_FIELDS = frozenset({"rank", "score"})
_RESPONSE_FIELDS = frozenset({"scores"})


class HTTPMemoryReranker:
    """Score candidates with one HTTP service, then select globally.

    The service enforces a batch ceiling, so the candidate pool is split into
    bounded requests. Every candidate keeps its original one-based global rank
    across batches; the final selection is taken over the whole pool rather
    than per batch.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = DEFAULT_HTTP_RERANK_POLICY_ID,
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")  # noqa: TRY003
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Return original global ranks ordered by descending service score."""

        _validate_bounds(len(candidates), limit)
        ranked = tuple(enumerate(candidates, start=1))
        scores: dict[int, float] = {}
        requests = 0

        for start in range(0, len(ranked), self._batch_size):
            batch = ranked[start : start + self._batch_size]
            expected = {rank for rank, _ in batch}
            payload = {
                "query": query,
                "candidates": [{"rank": rank, "text": hit.text} for rank, hit in batch],
            }
            response = await self._client.post(self._endpoint, json=payload)
            requests += 1
            response.raise_for_status()
            scores.update(_parse_scores(response, expected))

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return MemoryRerankDecision(
            selected_ranks=tuple(rank for rank, _ in ordered[:limit]),
            usage=InferenceUsage(requests=requests),
        )


def _parse_scores(response: httpx.Response, expected: set[int]) -> dict[int, float]:
    """Validate one batch response and return its rank to score mapping."""

    try:
        payload: Any = response.json()
    except ValueError as error:
        raise ValueError("reranker response must be valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != _RESPONSE_FIELDS:
        raise ValueError("reranker response must contain exactly the scores field")  # noqa: TRY003
    raw_scores = payload["scores"]
    if not isinstance(raw_scores, list):
        raise ValueError("reranker scores must be a list")  # noqa: TRY003

    parsed: dict[int, float] = {}
    for item in raw_scores:
        if not isinstance(item, dict) or set(item) != _SCORE_FIELDS:
            raise ValueError("each reranker score must contain exactly rank and score")  # noqa: TRY003
        rank = item["rank"]
        if type(rank) is not int or rank not in expected:
            raise ValueError("reranker score rank must belong to the current batch")  # noqa: TRY003
        if rank in parsed:
            raise ValueError("reranker score ranks must be unique within a batch")  # noqa: TRY003
        score = item["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("reranker score must be a number")  # noqa: TRY003
        numeric = float(score)
        if not math.isfinite(numeric):
            raise ValueError("reranker score must be finite")  # noqa: TRY003
        parsed[rank] = numeric

    if set(parsed) != expected:
        raise ValueError("reranker scores must cover every rank in the batch exactly once")  # noqa: TRY003
    return parsed


__all__ = ["DEFAULT_HTTP_RERANK_POLICY_ID", "HTTPMemoryReranker"]
