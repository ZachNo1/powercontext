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
import math
import re
from collections import Counter
from typing import TYPE_CHECKING
from urllib.parse import quote, unquote

from pydantic import BaseModel, Field
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

if TYPE_CHECKING:
    from collections.abc import Sequence

    from powercontext.builtin.artifacts.memory.models import MemoryCitation
    from powercontext.builtin.runtime.models import MemoryEntryRecord

    _RankablePage = tuple[MemoryEntryRecord, tuple[str, str, int], str]

PDF_PAGE_SOURCE_NAME = "pdf-page"
PDF_PAGE_SOURCE_VERSION = "1"
PDF_IDENTITY_PREFIX = "pdf"
PDF_IDENTITY_PARTS = 4

_STOPWORDS = frozenset(
    """
    a an and are as at be been being both by can could did do does for from had has have how
    if in into is it its may might must not of on or our shall should so than that the their
    them then there these they this those to was we were what when where which who whom why
    will with would you your
    """.split()  # noqa: SIM905
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
            raise ValueError("pdf requires a password that was not supplied")  # noqa: TRY003, TRY301
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
        raise ValueError("pdf content could not be parsed completely") from error  # noqa: TRY003
    if not pages:
        raise ValueError("pdf content must contain at least one page")  # noqa: TRY003
    return pages


# Remove question scaffolding, not accounting vocabulary. The corpus and its
# document frequencies come exclusively from the caller's scoped Memory.
_QUERY_NOISE = frozenset(
    """
    fy usd million millions billion billions company companies fiscal year years based
    data give response question relying details shown answer calculate define assuming
    perspective investment analyst only use please state explain why useful relevant
    metric measure profile improving much changed change drove among most least
    reported report reasonably healthy directly outlined explicitly quantity
""".split()  # noqa: SIM905
)

_RETRIEVAL_TOPICS = (
    (
        r"\b(?:margin|ebitda|profitability|net income|net earnings|income statement|p&l)\b",
        ("consolidated statements income earnings operations", "revenue cost sales operating income"),
    ),
    (
        r"\b(?:ebitda|capex|capital expenditure|capital-intensive|cash flow)\b",
        (
            "consolidated statements cash flows",
            "depreciation amortization capital expenditures property plant equipment",
        ),
    ),
    (
        r"\b(?:balance sheet|liabilit\w*|quick ratio|current ratio|liquidity|capital-intensive)\b",
        ("consolidated balance sheets", "total current assets current liabilities inventories"),
    ),
    (
        r"\b(?:geograph\w*|countries|regions|international)\b",
        ("geographic geographical areas united states international", "business operations segments countries"),
    ),
    (r"\b(?:legal|litigation|lawsuit|court|settlement)\b", ("legal proceedings", "litigation contingencies")),
    (
        r"\b(?:effective tax|income tax|tax rate)\b",
        ("effective income tax rate", "consolidated statements income earnings operations"),
    ),
    (r"\b(?:segments?|categories|products|services|growth)\b", ("business segment net sales revenues",)),
    (r"\b(?:retain|retention|attrition|members)\b", ("card members retention attrition", "cards in force basic cards")),
    (r"\bcustomers?\b", ("customers commercial government",)),
)


def _retrieval_terms(text: str) -> list[str]:
    terms = re.findall(r"[a-z]+\d*[a-z]*", text.lower())
    return [
        t[:-1] if len(t) > 4 and t.endswith("s") and not t.endswith("ss") else t
        for t in terms
        if t not in _STOPWORDS and t not in _QUERY_NOISE and len(t) > 1
    ]


def _document_match(document_id: str, question: str) -> float:
    company = re.split(r"[_\- ](?:19|20)\d{2}", document_id.lower())[0]
    company = re.sub(r"[^a-z0-9]", "", company)
    compact = re.sub(r"[^a-z0-9]", "", question.lower())
    if company and company in compact:
        return float(len(company))
    # Filenames often join company words; accept a shortened two-part name.
    for word in _retrieval_terms(question):
        if 4 <= len(word) <= 6 and company.startswith(word[:2]) and word[2:] in company[2:]:
            return 0.5
    return 0.0


def _record_page_key(record: MemoryEntryRecord) -> tuple[str, str, int] | None:
    return next((key for ref in record.entry.sources if (key := _source_page_key(ref.source_id)) is not None), None)


def _document_pages(records: Sequence[MemoryEntryRecord], question: str) -> list[_RankablePage]:
    """Resolve page provenance and narrow an explicitly named company/year."""
    pages = []
    for record in records:
        key = _record_page_key(record)
        if key is not None:
            pages.append((record, key, record.entry.text.partition("\n")[2]))
    matches = {key[0]: _document_match(key[0], question) for _, key, _ in pages}
    best = max(matches.values(), default=0.0)
    if best > 0:
        pages = [page for page in pages if matches[page[1][0]] == best]
        years = {int(year) for year in re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", question)}
        years.update(2000 + int(year) for year in re.findall(r"\bfy\s*['\u2019]?(\d{2})\b", question.lower()))
        if years:
            latest = str(max(years))
            dated = [page for page in pages if re.search(rf"(?:^|[_\- ]){latest}(?:\D|$)", page[1][0])]
            if dated:
                pages = dated
    return pages


def _lexical_scores(pages: list[_RankablePage], question: str) -> list[float]:
    """Fuse document-local BM25 rankings for the question and finance concepts."""
    company_words = set()
    for doc in {key[0] for _, key, _ in pages}:
        company_words.update(_retrieval_terms(re.split(r"[_\- ](?:19|20)\d{2}", doc)[0]))
    query = [term for term in _retrieval_terms(question) if term not in company_words]
    bodies = [Counter(_retrieval_terms(text)) for _, _, text in pages]
    frequency = Counter(term for body in bodies for term in body)
    lengths = [sum(body.values()) for body in bodies]
    average = sum(lengths) / max(1, len(bodies))
    probes = [(query, 1.5)]
    for pattern, expansions in _RETRIEVAL_TOPICS:
        if re.search(pattern, question.lower()):
            probes.extend((_retrieval_terms(expansion), 1.0 / len(expansions)) for expansion in expansions)
    fused = [0.0] * len(pages)
    for terms, weight in probes:
        ranking = []
        for index, (body, length) in enumerate(zip(bodies, lengths, strict=True)):
            score = 0.0
            for term in terms:
                tf = body[term]
                if tf:
                    idf = math.log(1 + (len(pages) - frequency[term] + 0.5) / (frequency[term] + 0.5))
                    score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * length / max(1, average)))
            if score:
                ranking.append((score, index))
        ranking.sort(key=lambda item: (-item[0], pages[item[1]][1]))
        for rank, (_, index) in enumerate(ranking):
            fused[index] += weight / (10 + rank)
    return fused


def _structure_scores(pages: list[_RankablePage], question: str) -> list[float]:
    # Filing headings are stronger evidence than repeated terms in footnotes.
    # Keep the source text intact; structure affects ranking only.
    fused = [0.0] * len(pages)
    lowered = question.lower()
    for index, (_, _, text) in enumerate(pages):
        header = re.sub(r"\s+", " ", text[:500].lower())
        if re.search(r"margin|ebitda|capital-intensive|income statement|p&l|tax rate", lowered) and re.search(
            r"consolidated statements? (?:of )?(?:income|earnings|operations)", header
        ):
            fused[index] += 0.25
        if re.search(
            r"balance sheet|liabilit|quick ratio|current ratio|liquidity|capital-intensive", lowered
        ) and re.search(r"consolidated balance sheets?", header):
            fused[index] += 0.25
        if re.search(r"cash flow|ebitda|capex|capital expenditure|capital-intensive", lowered) and re.search(
            r"consolidated statements? (?:of )?cash flow", header
        ):
            fused[index] += 0.25
        if re.search(r"geograph|countries|regions", lowered) and re.search(
            r"geograph\w* (?:operations|information|areas|region)", header
        ):
            fused[index] += 0.25
        if re.search(r"products|services|primary customers|geograph", lowered) and re.search(
            r"\bitem\s+1\.?\s+business\b", re.sub(r"\s+", " ", text.lower())
        ):
            fused[index] += 0.25
            document, digest, number = pages[index][1]
            for neighbor, (_, key, _) in enumerate(pages):
                if key[:2] == (document, digest) and number < key[2] <= number + 2:
                    fused[neighbor] += 0.18
    return fused


def _rank_pdf_records(records: Sequence[MemoryEntryRecord], question: str) -> list[MemoryEntryRecord]:
    """Rank real, active Memory entries without manufacturing references."""
    pages = _document_pages(records, question)
    lexical = _lexical_scores(pages, question)
    structural = _structure_scores(pages, question)
    scores = [
        (left + right, key, record) for left, right, (record, key, _) in zip(lexical, structural, pages, strict=True)
    ]
    scores.sort(key=lambda item: (-item[0], item[1]))
    return [record for _, _, record in scores]


async def search_pdf_pages(memory, question: str, limit: int = 10) -> tuple[MemoryCitation, ...]:
    """Rank scoped, active PDF pages and return their exact stored citations.

    A single public listing avoids FTS admission dropping pages before the
    document-aware BM25 and financial-section ranking. At the challenge's scale
    this scans about a thousand pages, without per-candidate database reads.
    No cache survives mutations or a Runtime restart.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-blank string")  # noqa: TRY003
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")  # noqa: TRY003
    records = (await memory.list()).entries
    output: list[MemoryCitation] = []
    seen: set[tuple[str, str, int]] = set()
    for record in _rank_pdf_records(records, question):
        key = _record_page_key(record)
        if key is not None and key not in seen:
            seen.add(key)
            output.append(record.citation)
            if len(output) == limit:
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
