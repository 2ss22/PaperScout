"""Validated records shared by PaperScout's tools, agents and UI."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


SourceKind = Literal['pdf', 'abstract', 'table', 'vision']


class TableCell(Record):
    row: int = Field(ge=0, le=200)
    column: int = Field(ge=0, le=100)
    text: str
    row_span: int = Field(default=1, ge=1, le=200)
    column_span: int = Field(default=1, ge=1, le=100)
    header: bool = False


class VisualElement(Record):
    kind: Literal['figure', 'table']
    label: str
    description: str
    visible_text: str
    cells: list[TableCell] = Field(default_factory=list, max_length=600)
    limitations: str = ''


class VisualReading(Record):
    readable: bool
    elements: list[VisualElement] = Field(max_length=8)
    limitations: str = ''


class VisualAsset(Record):
    id: str
    paper_id: str
    page: int
    kind: Literal['page', 'table'] = 'page'
    image_path: str
    image_sha256: str
    pdf_sha256: str
    bbox: list[float] = Field(default_factory=list)
    caption: str = ''
    reading: VisualReading | None = None
    error: str = ''


class Paper(Record):
    id: str  # Versioned arXiv ID, e.g. 2412.06769v2
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    published: str = ""
    reason: str = ""
    search_source: str = 'arxiv_api'
    read_status: Literal["not_read", "full_text", "abstract_only", "visual_only"] = "not_read"
    read_error: str = ""
    assets: list[VisualAsset] = Field(default_factory=list)
    visual_pages_available: int = 0
    visual_limitations: list[str] = Field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://arxiv.org/abs/{self.id}"

    @property
    def pdf_url(self) -> str:
        return f"https://arxiv.org/pdf/{self.id}"


class TextChunk(Record):
    id: str
    paper_id: str
    page: int | None  # Physical PDF page, 1-based; None means abstract.
    text: str
    source: SourceKind = "pdf"
    asset_id: str = ''
    cells: list[TableCell] = Field(default_factory=list)


class Evidence(Record):
    id: str
    paper_id: str
    chunk_id: str
    page: int | None
    quote: str
    source: SourceKind
    asset_id: str = ''


class Claim(Record):
    id: str
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    kind: Literal["problem", "method", "dataset", "metric", "conclusion"] = "conclusion"
    status: Literal["supported", "partial", "insufficient", "unreviewed"] = "unreviewed"
    reason: str = ""


class Gap(Record):
    question: str
    experiment: str


class Usage(Record):
    calls: int = 0
    vision_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls_with_usage: int = 0


class Event(Record):
    stage: str
    message: str
    data: object | None = None


class Trace(Record):
    stage: str
    message: str
    elapsed_seconds: float
    details: dict = Field(default_factory=dict)


class RunResult(Record):
    schema_version: int = 2
    run_id: str
    created_at: str
    question: str
    provider: str
    model: str
    status: Literal["running", "complete", "partial", "failed"] = "running"
    queries: list[str] = Field(default_factory=list)
    candidates: list[Paper] = Field(default_factory=list)
    papers: list[Paper] = Field(default_factory=list)
    chunks: list[TextChunk] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    # Freeze each pre-review draft for paired, same-output evaluation.
    drafts: list[list[Claim]] = Field(default_factory=list)
    trace: list[Trace] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    elapsed_seconds: float = 0
    supplemental_rounds: int = 0
    stop_reason: str = ""
    report: str = ""


# LLM output contracts: IDs and quotations are checked against tool-owned data.
class Plan(Record):
    queries: list[str] = Field(min_length=1, max_length=2)


class Selection(Record):
    paper_id: str
    reason: str


class Selections(Record):
    papers: list[Selection] = Field(max_length=3)


class ExtractedFinding(Record):
    kind: Literal["problem", "method", "dataset", "metric", "conclusion"]
    text: str
    chunk_id: str
    quote: str = Field(min_length=15, max_length=1500)


class Extraction(Record):
    findings: list[ExtractedFinding] = Field(max_length=8)


class ExtractionResult(Extraction):
    # Executor diagnostics, never requested from the model as evidence.
    warnings: list[str] = Field(default_factory=list)


class QuoteSpan(Record):
    id: str
    chunk_id: str
    start: int
    end: int
    quote: str


class AnchoredFinding(Record):
    kind: Literal['problem', 'method', 'dataset', 'metric', 'conclusion']
    text: str
    span_id: str


class AnchoredExtraction(Record):
    findings: list[AnchoredFinding] = Field(max_length=8)


class AnchoredExtractionResult(AnchoredExtraction):
    warnings: list[str] = Field(default_factory=list)


class DraftClaim(Record):
    text: str
    evidence_ids: list[str] = Field(min_length=1, max_length=8)
    kind: Literal["problem", "method", "dataset", "metric", "conclusion"]


class Draft(Record):
    claims: list[DraftClaim] = Field(max_length=20)
    gaps: list[Gap] = Field(max_length=3)


class Verdict(Record):
    claim_id: str
    status: Literal["supported", "partial", "insufficient"]
    reason: str


class Review(Record):
    verdicts: list[Verdict]
    action: Literal["finish", "search"]
    reason: str
    query: str = ""


class VisualReview(Record):
    verdicts: list[Verdict]
