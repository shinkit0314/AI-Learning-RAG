"""Deterministic contracts and metrics for audited PDF retrieval evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from pathlib import Path

from ai_learning_rag.document_index import SearchResult, VectorIndexManifest

EVALUATION_SCHEMA_VERSION = "pdf-retrieval-evaluation-v1"


class EvaluationCaseType(StrEnum):
    """Supported question roles inside one retrieval evaluation dataset."""

    ANSWERABLE = "answerable"
    UNANSWERABLE = "unanswerable"
    AMBIGUOUS = "ambiguous"


class ReviewStatus(StrEnum):
    """Human-review state for one proposed evaluation case."""

    DRAFT = "draft"
    APPROVED = "approved"


class CorpusCompatibilityError(ValueError):
    """The evaluation gold set does not belong to the active chunk corpus."""


@dataclass(frozen=True, order=True, slots=True)
class ChunkIdentity:
    """Stable evaluation identity for one accepted PDF chunk."""

    document_id: str
    chunk_fingerprint: str

    def __post_init__(self) -> None:
        _require_text(self.document_id, label="document_id")
        _require_sha256(self.chunk_fingerprint, label="chunk_fingerprint")

    def to_payload(self) -> dict[str, str]:
        """Return the canonical JSON representation used by corpus hashing."""

        return {
            "document_id": self.document_id,
            "chunk_fingerprint": self.chunk_fingerprint.lower(),
        }


@dataclass(frozen=True, slots=True)
class CorpusSample:
    """One immutable source file represented by a corpus snapshot."""

    sample_id: str
    path: str
    document_id: str
    file_sha256: str

    def __post_init__(self) -> None:
        for label, value in (
            ("sample_id", self.sample_id),
            ("path", self.path),
            ("document_id", self.document_id),
        ):
            _require_text(value, label=label)
        _require_sha256(self.file_sha256, label="file_sha256")

    def to_payload(self) -> dict[str, str]:
        """Return stable source metadata for reports and compatibility checks."""

        return {
            "sample_id": self.sample_id,
            "path": self.path,
            "document_id": self.document_id,
            "file_sha256": self.file_sha256.lower(),
        }


@dataclass(frozen=True, slots=True)
class CorpusSnapshot:
    """Compact identity of the exact PDF files and accepted chunk universe."""

    corpus_id: str
    samples: tuple[CorpusSample, ...]
    chunking_profile: str
    gate_profile: str
    chunk_count: int
    chunk_set_fingerprint: str

    def __post_init__(self) -> None:
        for label, value in (
            ("corpus_id", self.corpus_id),
            ("chunking_profile", self.chunking_profile),
            ("gate_profile", self.gate_profile),
        ):
            _require_text(value, label=label)
        if not self.samples:
            raise ValueError("corpus snapshot must contain at least one sample")
        sample_ids = [sample.sample_id for sample in self.samples]
        document_ids = [sample.document_id for sample in self.samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("corpus sample ids must be unique")
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("corpus document ids must be unique")
        if tuple(sorted(sample_ids)) != tuple(sample_ids):
            raise ValueError("corpus samples must be sorted by sample_id")
        if self.chunk_count <= 0:
            raise ValueError("chunk_count must be greater than zero")
        _require_sha256(self.chunk_set_fingerprint, label="chunk_set_fingerprint")

    def to_payload(self) -> dict[str, object]:
        """Return the versioned compatibility contract."""

        return {
            "corpus_id": self.corpus_id,
            "samples": [sample.to_payload() for sample in self.samples],
            "chunking_profile": self.chunking_profile,
            "gate_profile": self.gate_profile,
            "chunk_count": self.chunk_count,
            "chunk_set_fingerprint": self.chunk_set_fingerprint.lower(),
        }


@dataclass(frozen=True, slots=True)
class CorpusInventory:
    """Exact accepted chunks used to derive and validate a corpus snapshot."""

    corpus_id: str
    samples: tuple[CorpusSample, ...]
    chunking_profile: str
    gate_profile: str
    chunks: tuple[ChunkIdentity, ...]

    def __post_init__(self) -> None:
        if not self.chunks:
            raise ValueError("corpus inventory must contain at least one chunk")
        if tuple(sorted(set(self.chunks))) != self.chunks:
            raise ValueError("corpus chunks must be sorted and unique")
        sample_documents = {sample.document_id for sample in self.samples}
        if any(chunk.document_id not in sample_documents for chunk in self.chunks):
            raise ValueError("every corpus chunk must belong to a declared sample")
        _ = self.snapshot  # Trigger all compact snapshot invariants.

    @classmethod
    def create(
        cls,
        *,
        corpus_id: str,
        samples: Sequence[CorpusSample],
        chunking_profile: str,
        gate_profile: str,
        chunks: Sequence[ChunkIdentity],
    ) -> CorpusInventory:
        """Canonicalize source and chunk order before computing compatibility."""

        return cls(
            corpus_id=corpus_id,
            samples=tuple(sorted(samples, key=lambda sample: sample.sample_id)),
            chunking_profile=chunking_profile,
            gate_profile=gate_profile,
            chunks=tuple(sorted(chunks)),
        )

    @property
    def snapshot(self) -> CorpusSnapshot:
        """Return the compact dataset-facing identity for this inventory."""

        payload = {
            "corpus_id": self.corpus_id,
            "samples": [sample.to_payload() for sample in self.samples],
            "chunking_profile": self.chunking_profile,
            "gate_profile": self.gate_profile,
            "chunks": [chunk.to_payload() for chunk in self.chunks],
        }
        return CorpusSnapshot(
            corpus_id=self.corpus_id,
            samples=self.samples,
            chunking_profile=self.chunking_profile,
            gate_profile=self.gate_profile,
            chunk_count=len(self.chunks),
            chunk_set_fingerprint=_canonical_sha256(payload),
        )

    def to_payload(self) -> dict[str, object]:
        """Return exact chunk membership plus its compact compatibility snapshot."""

        return {
            "snapshot": self.snapshot.to_payload(),
            "chunks": [chunk.to_payload() for chunk in self.chunks],
        }


@dataclass(frozen=True, slots=True)
class RelevantChunkReference:
    """Human-approved relevance judgment plus non-identity audit context."""

    identity: ChunkIdentity
    sample_id: str
    page_start: int
    page_end: int
    content_type: str
    section_path: tuple[str, ...]
    source_block_ids: tuple[str, ...]
    content: str

    def __post_init__(self) -> None:
        for label, value in (
            ("sample_id", self.sample_id),
            ("content_type", self.content_type),
            ("content", self.content),
        ):
            _require_text(value, label=label)
        if self.page_start < 1 or self.page_end < self.page_start:
            raise ValueError("relevant chunk pages must form a positive ordered range")
        if any(not value.strip() for value in self.section_path):
            raise ValueError("section_path entries must not be blank")
        if not self.source_block_ids or any(not value.strip() for value in self.source_block_ids):
            raise ValueError("source_block_ids must contain non-blank values")
        if len(self.source_block_ids) != len(set(self.source_block_ids)):
            raise ValueError("source_block_ids must be unique")

    def to_payload(self) -> dict[str, object]:
        """Return both machine identity and human-review context."""

        return {
            **self.identity.to_payload(),
            "sample_id": self.sample_id,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "content_type": self.content_type,
            "section_path": list(self.section_path),
            "source_block_ids": list(self.source_block_ids),
            "content": self.content,
        }


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationCase:
    """One question and its human-reviewed chunk relevance judgments."""

    query_id: str
    query: str
    case_type: EvaluationCaseType
    review_status: ReviewStatus
    relevant_chunks: tuple[RelevantChunkReference, ...]
    notes: str = ""

    def __post_init__(self) -> None:
        _require_text(self.query_id, label="query_id")
        _require_text(self.query, label="query")
        identities = [reference.identity for reference in self.relevant_chunks]
        if len(identities) != len(set(identities)):
            raise ValueError("relevant chunk identities must be unique within a question")
        if self.case_type is EvaluationCaseType.ANSWERABLE and not self.relevant_chunks:
            raise ValueError("answerable questions require at least one relevant chunk")
        if self.case_type is not EvaluationCaseType.ANSWERABLE and self.relevant_chunks:
            raise ValueError("diagnostic questions must have an empty relevant chunk set")
        if self.notes and not self.notes.strip():
            raise ValueError("notes must not contain only whitespace")

    def to_payload(self) -> dict[str, object]:
        """Return the auditable annotation payload."""

        return {
            "query_id": self.query_id,
            "query": self.query,
            "case_type": self.case_type.value,
            "review_status": self.review_status.value,
            "relevant_chunks": [reference.to_payload() for reference in self.relevant_chunks],
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationDataset:
    """Versioned questions, judgments, metric settings, and corpus compatibility."""

    dataset_id: str
    dataset_version: str
    corpus: CorpusSnapshot
    recall_ks: tuple[int, ...]
    mrr_k: int
    cases: tuple[RetrievalEvaluationCase, ...]
    schema_version: str = EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EVALUATION_SCHEMA_VERSION:
            raise ValueError("unsupported retrieval evaluation schema")
        _require_text(self.dataset_id, label="dataset_id")
        _require_text(self.dataset_version, label="dataset_version")
        if not self.recall_ks or any(value <= 0 for value in self.recall_ks):
            raise ValueError("recall_ks must contain positive values")
        if tuple(sorted(set(self.recall_ks))) != self.recall_ks:
            raise ValueError("recall_ks must be sorted and unique")
        if self.mrr_k <= 0 or self.mrr_k not in self.recall_ks:
            raise ValueError("mrr_k must be one of the configured recall_ks")
        if not self.cases:
            raise ValueError("evaluation dataset must contain at least one case")
        query_ids = [case.query_id for case in self.cases]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("query ids must be unique")

    @property
    def max_k(self) -> int:
        """Return the single retrieval depth needed for every configured metric."""

        return max(*self.recall_ks, self.mrr_k)

    def to_payload(self) -> dict[str, object]:
        """Return the complete versioned dataset representation."""

        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "corpus": self.corpus.to_payload(),
            "metrics": {"recall_ks": list(self.recall_ks), "mrr_k": self.mrr_k},
            "cases": [case.to_payload() for case in self.cases],
        }


@dataclass(frozen=True, slots=True)
class EvaluatedHit:
    """One ranked retrieval result with its gold relevance decision."""

    rank: int
    result: SearchResult
    relevant: bool

    @property
    def identity(self) -> ChunkIdentity:
        """Return the stable chunk key used for relevance matching."""

        return ChunkIdentity(
            self.result.record.document_id,
            self.result.record.chunk_fingerprint,
        )


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    """Per-question ranking, missing gold, and optional formal metrics."""

    case: RetrievalEvaluationCase
    hits: tuple[EvaluatedHit, ...]
    missing_relevant: tuple[ChunkIdentity, ...]
    recall_at_k: tuple[tuple[int, float], ...]
    reciprocal_rank: float | None
    first_relevant_rank: int | None


@dataclass(frozen=True, slots=True)
class AggregateRetrievalMetrics:
    """Macro-averaged metrics over approved answerable questions."""

    scored_query_count: int
    diagnostic_query_count: int
    recall_at_k: tuple[tuple[int, float], ...]
    mrr_k: int
    mrr: float


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationReport:
    """Auditable real or fixture evaluation bound to an exact index manifest."""

    dataset: RetrievalEvaluationDataset
    corpus: CorpusSnapshot
    index_manifest: VectorIndexManifest
    queries: tuple[QueryEvaluation, ...]
    aggregate: AggregateRetrievalMetrics

    def to_payload(self) -> dict[str, object]:
        """Return machine-readable aggregate and per-question evidence."""

        return {
            "schema_version": "pdf-retrieval-evaluation-report-v1",
            "dataset": {
                "dataset_id": self.dataset.dataset_id,
                "dataset_version": self.dataset.dataset_version,
                "schema_version": self.dataset.schema_version,
            },
            "corpus": self.corpus.to_payload(),
            "index": {
                "index_id": self.index_manifest.index_id,
                "status": self.index_manifest.status.value,
                "identity": self.index_manifest.identity_payload,
                "audit_metadata": dict(self.index_manifest.audit_metadata),
            },
            "metrics": {
                "scored_query_count": self.aggregate.scored_query_count,
                "diagnostic_query_count": self.aggregate.diagnostic_query_count,
                "recall_at_k": {str(key): value for key, value in self.aggregate.recall_at_k},
                "mrr_k": self.aggregate.mrr_k,
                "mrr": self.aggregate.mrr,
            },
            "queries": [_query_payload(query) for query in self.queries],
        }

    def to_json(self) -> str:
        """Render deterministic human-readable JSON."""

        return json.dumps(self.to_payload(), ensure_ascii=False, indent=2)

    def to_markdown(self) -> str:
        """Render aggregate scores and complete ranked evidence for audit."""

        recall_summary = ", ".join(
            f"Recall@{key} `{value:.6f}`" for key, value in self.aggregate.recall_at_k
        )
        lines = [
            "# PDF retrieval evaluation",
            "",
            f"- Dataset: `{self.dataset.dataset_id}` `{self.dataset.dataset_version}`",
            f"- Corpus: `{self.corpus.corpus_id}`",
            f"- Chunk set: `{self.corpus.chunk_set_fingerprint}`",
            f"- Index: `{self.index_manifest.index_id}`",
            f"- Scored questions: {self.aggregate.scored_query_count}",
            f"- Diagnostic questions: {self.aggregate.diagnostic_query_count}",
            f"- {recall_summary}",
            f"- MRR@{self.aggregate.mrr_k}: `{self.aggregate.mrr:.6f}`",
            "",
        ]
        for query in self.queries:
            lines.extend(_query_markdown(query, self.dataset.mrr_k))
        return "\n".join(lines)


def evaluate_retrieval(
    *,
    dataset: RetrievalEvaluationDataset,
    inventory: CorpusInventory,
    results_by_query: Mapping[str, Sequence[SearchResult]],
    index_manifest: VectorIndexManifest,
) -> RetrievalEvaluationReport:
    """Compare ranked chunks with approved gold after fail-closed compatibility checks."""

    actual_snapshot = inventory.snapshot
    if actual_snapshot != dataset.corpus:
        raise CorpusCompatibilityError(
            "evaluation corpus does not match the dataset snapshot: "
            f"expected {dataset.corpus.chunk_set_fingerprint}, "
            f"observed {actual_snapshot.chunk_set_fingerprint}"
        )
    unapproved = [
        case.query_id for case in dataset.cases if case.review_status is not ReviewStatus.APPROVED
    ]
    if unapproved:
        raise ValueError("formal evaluation requires approved cases: " + ", ".join(unapproved))
    expected_query_ids = {case.query_id for case in dataset.cases}
    observed_query_ids = set(results_by_query)
    if observed_query_ids != expected_query_ids:
        missing = sorted(expected_query_ids - observed_query_ids)
        extra = sorted(observed_query_ids - expected_query_ids)
        raise ValueError(
            f"retrieval result query ids do not match; missing={missing}, extra={extra}"
        )
    if any(
        result.record.index_id != index_manifest.index_id
        for results in results_by_query.values()
        for result in results
    ):
        raise ValueError("retrieval result index_id does not match the report manifest")

    available_chunks = set(inventory.chunks)
    evaluations = tuple(
        _evaluate_case(
            case,
            results_by_query[case.query_id],
            dataset,
            available_chunks,
        )
        for case in dataset.cases
    )
    scored = [
        evaluation
        for evaluation in evaluations
        if evaluation.case.case_type is EvaluationCaseType.ANSWERABLE
    ]
    if not scored:
        raise ValueError("evaluation dataset requires at least one answerable case")
    aggregate = AggregateRetrievalMetrics(
        scored_query_count=len(scored),
        diagnostic_query_count=len(evaluations) - len(scored),
        recall_at_k=tuple(
            (
                key,
                sum(dict(evaluation.recall_at_k)[key] for evaluation in scored) / len(scored),
            )
            for key in dataset.recall_ks
        ),
        mrr_k=dataset.mrr_k,
        mrr=sum(evaluation.reciprocal_rank or 0.0 for evaluation in scored) / len(scored),
    )
    return RetrievalEvaluationReport(
        dataset=dataset,
        corpus=actual_snapshot,
        index_manifest=index_manifest,
        queries=evaluations,
        aggregate=aggregate,
    )


def load_retrieval_evaluation_dataset(path: Path) -> RetrievalEvaluationDataset:
    """Load and validate a human-reviewed JSON evaluation dataset."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    payload = _require_mapping(raw, label="dataset")
    corpus_payload = _require_mapping(payload.get("corpus"), label="corpus")
    metrics_payload = _require_mapping(payload.get("metrics"), label="metrics")
    samples_payload = _require_sequence(corpus_payload.get("samples"), label="corpus.samples")
    cases_payload = _require_sequence(payload.get("cases"), label="cases")
    corpus = CorpusSnapshot(
        corpus_id=str(corpus_payload.get("corpus_id", "")),
        samples=tuple(_load_sample(item) for item in samples_payload),
        chunking_profile=str(corpus_payload.get("chunking_profile", "")),
        gate_profile=str(corpus_payload.get("gate_profile", "")),
        chunk_count=_require_int(corpus_payload.get("chunk_count"), label="chunk_count"),
        chunk_set_fingerprint=str(corpus_payload.get("chunk_set_fingerprint", "")),
    )
    return RetrievalEvaluationDataset(
        schema_version=str(payload.get("schema_version", "")),
        dataset_id=str(payload.get("dataset_id", "")),
        dataset_version=str(payload.get("dataset_version", "")),
        corpus=corpus,
        recall_ks=tuple(
            _require_int(value, label="recall_k")
            for value in _require_sequence(metrics_payload.get("recall_ks"), label="recall_ks")
        ),
        mrr_k=_require_int(metrics_payload.get("mrr_k"), label="mrr_k"),
        cases=tuple(_load_case(item) for item in cases_payload),
    )


def _evaluate_case(
    case: RetrievalEvaluationCase,
    results: Sequence[SearchResult],
    dataset: RetrievalEvaluationDataset,
    available_chunks: set[ChunkIdentity],
) -> QueryEvaluation:
    if len(results) > dataset.max_k:
        raise ValueError(f"query {case.query_id} returned more than Top-{dataset.max_k}")
    gold = {reference.identity for reference in case.relevant_chunks}
    missing_from_corpus = sorted(gold - available_chunks)
    if missing_from_corpus:
        raise CorpusCompatibilityError(
            f"query {case.query_id} references chunks absent from the compatible corpus"
        )
    hits: list[EvaluatedHit] = []
    observed: set[ChunkIdentity] = set()
    for rank, result in enumerate(results, start=1):
        if not isfinite(result.score):
            raise ValueError(f"query {case.query_id} contains a non-finite score")
        identity = ChunkIdentity(
            result.record.document_id,
            result.record.chunk_fingerprint,
        )
        if identity not in available_chunks:
            raise CorpusCompatibilityError(
                f"query {case.query_id} returned a chunk outside the compatible corpus"
            )
        if identity in observed:
            raise ValueError(f"query {case.query_id} returned a duplicate chunk identity")
        observed.add(identity)
        hits.append(EvaluatedHit(rank=rank, result=result, relevant=identity in gold))

    if case.case_type is not EvaluationCaseType.ANSWERABLE:
        return QueryEvaluation(case, tuple(hits), (), (), None, None)

    relevant_ranks = [hit.rank for hit in hits if hit.relevant]
    first_rank = relevant_ranks[0] if relevant_ranks else None
    recall = tuple(
        (
            key,
            sum(hit.relevant for hit in hits if hit.rank <= key) / len(gold),
        )
        for key in dataset.recall_ks
    )
    reciprocal_rank = 0.0 if first_rank is None or first_rank > dataset.mrr_k else 1.0 / first_rank
    returned_relevant = {hit.identity for hit in hits if hit.relevant}
    return QueryEvaluation(
        case=case,
        hits=tuple(hits),
        missing_relevant=tuple(sorted(gold - returned_relevant)),
        recall_at_k=recall,
        reciprocal_rank=reciprocal_rank,
        first_relevant_rank=first_rank,
    )


def _query_payload(evaluation: QueryEvaluation) -> dict[str, object]:
    metrics: dict[str, object] | None = None
    if evaluation.reciprocal_rank is not None:
        metrics = {
            "recall_at_k": {str(key): value for key, value in evaluation.recall_at_k},
            "reciprocal_rank": evaluation.reciprocal_rank,
            "first_relevant_rank": evaluation.first_relevant_rank,
        }
    return {
        "query_id": evaluation.case.query_id,
        "query": evaluation.case.query,
        "case_type": evaluation.case.case_type.value,
        "metrics": metrics,
        "relevant_chunks": [
            reference.to_payload() for reference in evaluation.case.relevant_chunks
        ],
        "missing_relevant": [identity.to_payload() for identity in evaluation.missing_relevant],
        "hits": [_hit_payload(hit) for hit in evaluation.hits],
    }


def _hit_payload(hit: EvaluatedHit) -> dict[str, object]:
    record = hit.result.record
    return {
        "rank": hit.rank,
        "score": hit.result.score,
        "relevant": hit.relevant,
        "record_id": record.record_id,
        "document_id": record.document_id,
        "chunk_fingerprint": record.chunk_fingerprint,
        "chunk_index": record.chunk_index,
        "content_type": record.content_type,
        "section_path": list(record.section_path),
        "page_start": record.page_start,
        "page_end": record.page_end,
        "content": record.content,
        "provenance": record.provenance,
    }


def _query_markdown(evaluation: QueryEvaluation, mrr_k: int) -> list[str]:
    lines = [
        f"## {evaluation.case.query_id}",
        "",
        f"- Query: {evaluation.case.query}",
        f"- Type: `{evaluation.case.case_type.value}`",
    ]
    if evaluation.reciprocal_rank is None:
        lines.append("- Formal metrics: not applicable (diagnostic case)")
    else:
        for key, value in evaluation.recall_at_k:
            lines.append(f"- Recall@{key}: `{value:.6f}`")
        lines.extend(
            [
                f"- Reciprocal rank@{mrr_k}: `{evaluation.reciprocal_rank:.6f}`",
                f"- First relevant rank: `{evaluation.first_relevant_rank}`",
                f"- Missing relevant chunks: {len(evaluation.missing_relevant)}",
            ]
        )
    lines.extend(
        [
            "",
            "| Rank | Relevant | Score | Document | Pages | Type | Content preview |",
            "| ---: | --- | ---: | --- | --- | --- | --- |",
        ]
    )
    for hit in evaluation.hits:
        record = hit.result.record
        preview = " ".join(record.content.split())[:180].replace("|", "\\|")
        lines.append(
            f"| {hit.rank} | {hit.relevant} | {hit.result.score:.6f} | "
            f"`{record.document_id}` | {record.page_start}-{record.page_end} | "
            f"{record.content_type} | {preview} |"
        )
    lines.extend(["", "### Complete retrieved content", ""])
    for hit in evaluation.hits:
        record = hit.result.record
        lines.extend(
            [
                f"#### Rank {hit.rank} — relevant: {hit.relevant}",
                "",
                f"- Chunk: `{record.document_id}` / `{record.chunk_fingerprint}`",
                f"- Record: `{record.record_id}`",
                f"- Pages: {record.page_start}-{record.page_end}",
                "",
                *[f"> {line}" if line else ">" for line in record.content.split("\n")],
                "",
            ]
        )
    return lines


def _load_sample(value: object) -> CorpusSample:
    payload = _require_mapping(value, label="sample")
    return CorpusSample(
        sample_id=str(payload.get("sample_id", "")),
        path=str(payload.get("path", "")),
        document_id=str(payload.get("document_id", "")),
        file_sha256=str(payload.get("file_sha256", "")),
    )


def _load_case(value: object) -> RetrievalEvaluationCase:
    payload = _require_mapping(value, label="case")
    references = _require_sequence(payload.get("relevant_chunks"), label="relevant_chunks")
    return RetrievalEvaluationCase(
        query_id=str(payload.get("query_id", "")),
        query=str(payload.get("query", "")),
        case_type=EvaluationCaseType(str(payload.get("case_type", ""))),
        review_status=ReviewStatus(str(payload.get("review_status", ""))),
        relevant_chunks=tuple(_load_reference(item) for item in references),
        notes=str(payload.get("notes", "")),
    )


def _load_reference(value: object) -> RelevantChunkReference:
    payload = _require_mapping(value, label="relevant_chunk")
    return RelevantChunkReference(
        identity=ChunkIdentity(
            document_id=str(payload.get("document_id", "")),
            chunk_fingerprint=str(payload.get("chunk_fingerprint", "")),
        ),
        sample_id=str(payload.get("sample_id", "")),
        page_start=_require_int(payload.get("page_start"), label="page_start"),
        page_end=_require_int(payload.get("page_end"), label="page_end"),
        content_type=str(payload.get("content_type", "")),
        section_path=tuple(
            str(item)
            for item in _require_sequence(payload.get("section_path"), label="section_path")
        ),
        source_block_ids=tuple(
            str(item)
            for item in _require_sequence(payload.get("source_block_ids"), label="source_block_ids")
        ),
        content=str(payload.get("content", "")),
    )


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(serialized.encode()).hexdigest()


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _require_int(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return value


def _require_text(value: str, *, label: str) -> None:
    if not value.strip():
        raise ValueError(f"{label} must not be blank")


def _require_sha256(value: str, *, label: str) -> None:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must contain 64 hexadecimal characters")
