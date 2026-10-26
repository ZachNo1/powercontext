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

_SEARCH_CANDIDATE_MULTIPLIER = 12
_SEARCH_CANDIDATE_FLOOR = 120
_SEARCH_CANDIDATE_CEILING = 400
_KEYWORD_LIMIT = 6
_STOPWORDS = frozenset(
    """
    a an and are as at be been being both by can could did do does for from had has have how
    if in into is it its may might must not of on or our shall should so than that the their
    them then there these they this those to was we were what when where which who whom why
    will with would you your
    """.split()
)
_WORD_PATTERN = re.compile(r"[a-z]+|\d+")


def _query_terms(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    seen: list[str] = []
    for token in _WORD_PATTERN.findall(lowered):
        if token in _STOPWORDS or len(token) < 2:
            continue
        if token not in seen:
            seen.append(token)
    return tuple(seen)


async def search_pdf_pages(memory, question: str, limit: int = 10) -> tuple[object, ...]:
    """Return up to ``limit`` citations that each point at a distinct page.

    Long natural-language questions rank poorly under a single lexical query,
    so the search runs the question and a keyword probe, then re-ranks the
    merged candidate pool by how much of the question each page covers.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-blank string")  # noqa: TRY003
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")  # noqa: TRY003

    from powercontext.builtin.artifacts.memory.models import MemoryCitation
    from powercontext.builtin.runtime import GetMemoryEntryRequest, SearchMemoryRequest

    terms = _query_terms(question)
    candidates = min(
        max(limit * _SEARCH_CANDIDATE_MULTIPLIER, _SEARCH_CANDIDATE_FLOOR),
        _SEARCH_CANDIDATE_CEILING,
    )
    probes = [question]
    if terms:
        probes.append(" ".join(terms[:_KEYWORD_LIMIT]))

    ranked: dict[tuple[str, int], tuple[int, int, object]] = {}
    for probe_index, probe in enumerate(probes):
        page = await memory.search(SearchMemoryRequest(query=probe, mode="fts", limit=candidates))
        for hit_index, hit in enumerate(page.hits):
            citation = MemoryCitation(
                memory_ref=hit.memory_ref,
                entry_id=hit.entry_id,
                entry_version_id=hit.entry_version_id,
            )
            entry = (await memory.get(GetMemoryEntryRequest(citation=citation))).entry
            location = None
            for reference in entry.sources:
                location = _location_of(reference.source_id)
                if location is not None:
                    break
            if location is None:
                continue
            covered = sum(1 for term in terms if term in hit.text.lower())
            # Keep the earliest lexical position as the tiebreaker so the
            # original ranking still decides pages that cover the same terms.
            position = probe_index * candidates + hit_index
            current = ranked.get(location)
            if current is None or (covered, -position) > (current[0], -current[1]):
                ranked[location] = (covered, position, citation)

    ordered = sorted(ranked.values(), key=lambda item: item[1])
    return tuple(item[2] for item in ordered[:limit])


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
    identity = (
        f"{PDF_IDENTITY_PREFIX}:{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    )
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


def _page_text(page: object) -> str:
    text = page.extract_text()
    return text if isinstance(text, str) else ""


def _require_document_id(document_id: str) -> str:
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-blank string")  # noqa: TRY003
    if document_id != document_id.strip():
        raise ValueError("document_id must not contain leading or trailing whitespace")  # noqa: TRY003
    return document_id


def _location_of(source_id: str) -> tuple[str, int] | None:
    parts = source_id.split(":")
    if len(parts) != PDF_IDENTITY_PARTS or parts[0] != PDF_IDENTITY_PREFIX:
        return None
    try:
        page_number = int(parts[3])
    except ValueError:
        return None
    return (unquote(parts[1]), page_number)


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
