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

"""Page-level PDF capture so retrieval can cite an exact physical page."""

from __future__ import annotations

import hashlib
import io
import re
from urllib.parse import quote, unquote

from pydantic import BaseModel, Field
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PDF_PAGE_SOURCE_NAME = "pdf-page"
PDF_PAGE_SOURCE_VERSION = "1"
PDF_IDENTITY_PREFIX = "pdf"
PDF_IDENTITY_PARTS = 4

_SEARCH_CANDIDATE_FLOOR = 60
_SEARCH_CANDIDATE_MULTIPLIER = 4
_SEARCH_CANDIDATE_CEILING = 160
_SEARCH_RANK_OFFSET = 12
_SEARCH_PROBE_CEILING = 16
_KEYWORD_LIMIT = 4
_PHRASE_LIMIT = 3
_ALIAS_LIMIT = 6

_STOPWORDS = frozenset(
    """
    a an and are as at be been being both by can could did do does for from had has have how
    if in into is it its may might must not of on or our shall should so than that the their
    them then there these they this those to was we were what when where which who whom why
    will with would you your
    """.split()
)
_WORD_PATTERN = re.compile(r"[a-z]+|\d+")

# Analyst wording rarely appears verbatim in a filing: a question about EBITDA or
# capex is answered by a statement of income or a cash-flow statement. These are
# generic finance-document concepts, never document-, question- or answer-specific.
_FINANCIAL_ALIASES = (
    (
        r"\b(?:litigation|lawsuits?|legal|court|settlement|proceedings)\b",
        ("legal proceedings", "litigation", "commitments and contingencies"),
    ),
    (
        r"\b(?:ebitda|profitability|operating margin|profit margin|earnings)\b",
        ("consolidated statements of income", "operating income", "depreciation and amortization"),
    ),
    (
        r"\b(?:capital expenditures?|capex|investments? in property)\b",
        ("capital expenditures", "purchases of property plant and equipment", "cash flows from investing activities"),
    ),
    (
        r"\b(?:cash flow|cash flows|operating cash)\b",
        ("consolidated statements of cash flows", "cash provided by operating activities"),
    ),
    (
        r"\b(?:geograph\w*|regional|regions|international|countries|country)\b",
        ("geographic information", "geographical areas", "segment information"),
    ),
    (r"\b(?:revenue|sales|turnover)\b", ("net sales", "revenues", "consolidated statements of income")),
    (r"\b(?:net income|net earnings|profit|income statement)\b", ("net income", "consolidated statements of income")),
    (r"\b(?:segments?|business units?)\b", ("segment information", "reportable segments")),
    (r"\b(?:income tax|taxes|tax rate)\b", ("income taxes", "effective tax rate")),
    (
        r"\b(?:debt|borrowings|credit facilities|interest expense)\b",
        ("long term debt", "notes payable", "interest expense"),
    ),
)


class PDFDocumentInput(BaseModel):
    """One complete PDF document identified by a caller-stable document id."""

    document_id: str = Field(min_length=1)
    content: bytes


class PDFPageInput(BaseModel):
    """One physical PDF page and the extractable text it carries."""

    document_id: str = Field(min_length=1)
    document_sha256: str = Field(min_length=64, max_length=64)
    page_number: int = Field(ge=1)
    text: str


class PDFPageSource(Source):
    """A captured PDF page whose identity covers document, digest and page."""

    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Build the stable page identity from document, digest and physical page."""

    _require_document_id(document_id)
    if not isinstance(document_sha256, str) or len(document_sha256) != 64:
        raise ValueError("document_sha256 must be a 64-character hex digest")  # noqa: TRY003
    if type(page_number) is not int or page_number < 1:
        raise ValueError("page_number must be a positive integer")  # noqa: TRY003
    identity = f"{PDF_IDENTITY_PREFIX}:{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError("pdf page identity exceeds the maximum source id length")  # noqa: TRY003
    return identity


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Parse every physical page of one PDF, failing atomically on bad input."""

    document_id = _require_document_id(value.document_id)
    content = value.content
    if not isinstance(content, (bytes, bytearray)) or not content:
        raise ValueError("pdf content must be non-empty bytes")  # noqa: TRY003
    raw = bytes(content)
    digest = hashlib.sha256(raw).hexdigest()

    try:
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("pdf requires a password that was not supplied")  # noqa: TRY003
        pages = tuple(
            PDFPageInput(
                document_id=document_id,
                document_sha256=digest,
                page_number=index,
                text=_page_text(page),
            )
            for index, page in enumerate(reader.pages, start=1)
        )
    except ValueError:
        raise
    except (PdfReadError, OSError, KeyError, TypeError, AttributeError) as error:
        raise ValueError("pdf content could not be parsed completely") from error
    if not pages:
        raise ValueError("pdf content must contain at least one page")  # noqa: TRY003
    return pages


def _query_terms(text: str) -> tuple[str, ...]:
    seen: list[str] = []
    for token in _WORD_PATTERN.findall(text.lower()):
        if token in _STOPWORDS or len(token) < 2:
            continue
        if token not in seen:
            seen.append(token)
    return tuple(seen)


def _search_probes(question: str) -> list[tuple[str, str, float]]:
    """Build bounded lexical probes from the question alone.

    Nothing here is keyed to a document, a question id or an answer. Each probe
    carries a weight because a section-style phrase such as ``legal proceedings``
    discriminates far better than a single common word.
    """

    terms = _query_terms(question)
    probes: list[tuple[str, str, float]] = []
    seen: set[str] = set()

    def add(kind: str, query: str, weight: float) -> None:
        normalized = query.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            probes.append((kind, query, weight))

    add("original", question, 3.2)
    if terms:
        add("compact", " ".join(terms[:8]), 2.4)

    # The discriminating term is often late in the question, so keep the longest
    # content words rather than an arbitrary prefix of the original wording.
    priority = [term for _, term in sorted(enumerate(terms), key=lambda item: (-len(item[1]), item[0]))]
    priority = priority[:_KEYWORD_LIMIT]

    bigrams = [
        (len(left) + len(right), f"{left} {right}")
        for left, right in zip(terms, terms[1:])
        if len(left) > 2 and len(right) > 2
    ]
    for _, phrase in sorted(bigrams, reverse=True)[:_PHRASE_LIMIT]:
        add("phrase", phrase, 1.7)

    lowered = question.lower()
    for pattern, expansions in _FINANCIAL_ALIASES:
        if re.search(pattern, lowered):
            for expansion in expansions:
                add("alias", expansion, 1.9)
        if sum(1 for kind, _, _ in probes if kind == "alias") >= _ALIAS_LIMIT:
            break

    for term in priority:
        add("keyword", term, 0.65)
    return probes[:_SEARCH_PROBE_CEILING]


async def search_pdf_pages(memory, question: str, limit: int = 10) -> tuple[object, ...]:
    """Return up to ``limit`` citations that each point at a distinct page.

    Every probe votes with a weighted reciprocal-rank score. Fusion happens on
    Memory identity, and page de-duplication uses the complete
    ``(document, digest, page)`` key so two revisions of one document never
    collapse into a single citation.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-blank string")  # noqa: TRY003
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")  # noqa: TRY003

    from powercontext.builtin.artifacts.memory.models import MemoryCitation
    from powercontext.builtin.runtime import GetMemoryEntryRequest, SearchMemoryRequest

    per_probe = min(
        max(_SEARCH_CANDIDATE_FLOOR, limit * _SEARCH_CANDIDATE_MULTIPLIER),
        _SEARCH_CANDIDATE_CEILING,
    )

    scores: dict[tuple[str, str, str], float] = {}
    first_seen: dict[tuple[str, str, str], int] = {}
    citations: dict[tuple[str, str, str], object] = {}

    for _, query, weight in _search_probes(question):
        result = await memory.search(SearchMemoryRequest(query=query, mode="fts", limit=per_probe))
        for rank, hit in enumerate(result.hits):
            key = (str(hit.memory_ref), hit.entry_id, hit.entry_version_id)
            if key not in citations:
                citations[key] = MemoryCitation(
                    memory_ref=hit.memory_ref,
                    entry_id=hit.entry_id,
                    entry_version_id=hit.entry_version_id,
                )
                first_seen[key] = len(first_seen)
            scores[key] = scores.get(key, 0.0) + weight / (_SEARCH_RANK_OFFSET + rank)

    ordered = sorted(scores, key=lambda key: (-scores[key], first_seen[key]))
    output: list[object] = []
    pages: set[tuple[str, str, int]] = set()
    for key in ordered:
        entry = (await memory.get(GetMemoryEntryRequest(citation=citations[key]))).entry
        page = None
        for reference in entry.sources:
            page = _source_page_key(reference.source_id)
            if page is not None:
                break
        if page is None or page in pages:
            continue
        pages.add(page)
        output.append(citations[key])
        if len(output) >= limit:
            break
    return tuple(output)


def _page_text(page: object) -> str:
    text = page.extract_text()
    return text if isinstance(text, str) else ""


def _require_document_id(document_id: str) -> str:
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-blank string")  # noqa: TRY003
    return document_id


def _source_page_key(source_id: str) -> tuple[str, str, int] | None:
    parts = source_id.split(":")
    if len(parts) != PDF_IDENTITY_PARTS or parts[0] != PDF_IDENTITY_PREFIX:
        return None
    try:
        page_number = int(parts[3])
    except ValueError:
        return None
    return (unquote(parts[1]), parts[2], page_number)


def _location_of(source_id: str) -> tuple[str, int] | None:
    """Document and page only; kept for diagnostics that ignore revisions."""

    key = _source_page_key(source_id)
    return None if key is None else (key[0], key[2])


class PDFPageSourceAdapter:
    """Resolve and read one captured PDF page."""

    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        source_id = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    """Expose one page as text evidence that still locates the physical page."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.source_id,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_ADAPTER = PDFPageSourceAdapter()
PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(
    PDF_PAGE_SOURCE_ADAPTER,
    version=PDF_PAGE_SOURCE_VERSION,
    projections=(PDFPageTextEvidenceProjection(),),
)


__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDF_PAGE_SOURCE_NAME",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
