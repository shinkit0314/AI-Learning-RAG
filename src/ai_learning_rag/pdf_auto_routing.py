"""Page-level PDF parser orchestration before structure-aware chunking."""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, replace
from enum import StrEnum
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol

import pymupdf

from ai_learning_rag.docling_extraction import (
    create_docling_converter,
    parse_pdf_page_range_with_docling,
)
from ai_learning_rag.pdf_extraction import (
    parsed_document_from_pymupdf,
    pdf_file_sha256,
)
from ai_learning_rag.pdf_routing import (
    PageRouteDecision,
    PageRoutingSignals,
    PdfPageRoute,
    PdfRoutingConfig,
    TextQualityStatus,
    analyze_pymupdf_document,
    build_route_plan,
)
from ai_learning_rag.pdf_routing_cache import PdfRoutingCache, RoutingCacheEvent
from ai_learning_rag.pdf_schema import (
    ExtractionMethod,
    ExtractionProvenance,
    PageKind,
    ParsedDocument,
    ParsedPage,
    ParseStatus,
)


class ParseAttemptStatus(StrEnum):
    """Outcome of one parser attempt recorded for a routed page."""

    SUCCESS = "success"
    FAILED = "failed"


class OcrReviewChoice(StrEnum):
    """Explicit human resolution for one hidden-OCR conflict."""

    ACCEPT_EMBEDDED_OCR = "accept_embedded_ocr"
    ACCEPT_RERUN_OCR = "accept_rerun_ocr"
    REJECT_PAGE = "reject_page"


@dataclass(frozen=True, slots=True)
class OcrReviewDecision:
    """One auditable page decision loaded from the editable review file."""

    page_number: int
    conflict_id: str
    decision: OcrReviewChoice
    review_note: str

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("OCR review page_number must start at 1")
        if not self.conflict_id.strip():
            raise ValueError("OCR review conflict_id must not be blank")
        if not self.review_note.strip():
            raise ValueError("OCR review decision requires a non-blank review_note")


@dataclass(frozen=True, slots=True)
class OcrReviewDecisions:
    """Document-bound set of explicit page review decisions."""

    document_sha256: str
    routing_profile: str
    decisions: tuple[OcrReviewDecision, ...]

    def __post_init__(self) -> None:
        if len(self.document_sha256) != 64:
            raise ValueError("OCR review document_sha256 must contain 64 characters")
        try:
            int(self.document_sha256, 16)
        except ValueError as error:
            raise ValueError("OCR review document_sha256 must be hexadecimal") from error
        if not self.routing_profile.strip():
            raise ValueError("OCR review routing_profile must not be blank")
        page_numbers = [decision.page_number for decision in self.decisions]
        if len(page_numbers) != len(set(page_numbers)):
            raise ValueError("OCR review decisions must contain at most one decision per page")

    def by_page(self) -> dict[int, OcrReviewDecision]:
        """Return the decisions keyed by original one-based page number."""
        return {decision.page_number: decision for decision in self.decisions}


@dataclass(frozen=True, slots=True)
class OcrCandidateEvidence:
    """Immutable normalized candidate text shown to a human reviewer."""

    text_sha256: str
    normalized_text: str
    provenance: ExtractionProvenance

    def __post_init__(self) -> None:
        expected = sha256(self.normalized_text.encode()).hexdigest()
        if self.text_sha256 != expected:
            raise ValueError("OCR candidate text_sha256 does not match normalized_text")


@dataclass(frozen=True, slots=True)
class OcrConflictEvidence:
    """Two usable but different OCR candidates that require human review."""

    page_number: int
    conflict_id: str
    embedded_ocr: OcrCandidateEvidence
    rerun_ocr: OcrCandidateEvidence

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("OCR conflict page_number must start at 1")
        if not self.conflict_id.strip():
            raise ValueError("OCR conflict_id must not be blank")
        if self.embedded_ocr.text_sha256 == self.rerun_ocr.text_sha256:
            raise ValueError("OCR conflict candidates must contain different text")


@dataclass(frozen=True, slots=True)
class PageParseAttempt:
    """One visible parser attempt made while resolving a page route."""

    route: PdfPageRoute
    status: ParseAttemptStatus
    provenance: ExtractionProvenance | None = None
    warning_codes: tuple[str, ...] = ()
    failure_code: str | None = None

    def __post_init__(self) -> None:
        if self.status is ParseAttemptStatus.SUCCESS and self.failure_code is not None:
            raise ValueError("successful parse attempt cannot define failure_code")
        if self.status is ParseAttemptStatus.FAILED and not self.failure_code:
            raise ValueError("failed parse attempt must define failure_code")


@dataclass(frozen=True, slots=True)
class PageRoutingDiagnostic:
    """Decision, attempts, and final acceptance state for one page."""

    decision: PageRouteDecision
    attempts: tuple[PageParseAttempt, ...]
    final_status: ParseStatus
    final_provenance_id: str | None = None
    ocr_conflict: OcrConflictEvidence | None = None
    review_decision_id: str | None = None

    def __post_init__(self) -> None:
        if (
            self.final_status is ParseStatus.PARSED
            and not self.final_provenance_id
            and self.decision.selected_route is not PdfPageRoute.NONE
        ):
            raise ValueError("parsed routed page must identify its final provenance")
        if self.final_status is not ParseStatus.PARSED and self.final_provenance_id is not None:
            raise ValueError("unaccepted page must not define final provenance")
        if self.review_decision_id is not None and self.ocr_conflict is None:
            raise ValueError("review_decision_id requires OCR conflict evidence")


@dataclass(frozen=True, slots=True)
class PdfAutoRoutingResult:
    """Unified parsed document plus page-level routing evidence."""

    document: ParsedDocument
    page_diagnostics: tuple[PageRoutingDiagnostic, ...]
    page_signals: tuple[PageRoutingSignals, ...]
    routing_profile: str
    cache_events: tuple[RoutingCacheEvent, ...] = ()

    def __post_init__(self) -> None:
        page_numbers = [diagnostic.decision.page_number for diagnostic in self.page_diagnostics]
        if page_numbers != list(range(1, self.document.page_count + 1)):
            raise ValueError("routing diagnostics must cover every document page in order")
        if [signals.page_number for signals in self.page_signals] != page_numbers:
            raise ValueError("page signals and routing diagnostics must cover the same pages")
        if self.document.routing_profile != self.routing_profile:
            raise ValueError("result and document routing profiles must match")


@dataclass(frozen=True, slots=True)
class ParsedPageBatch:
    """Normalized pages returned by one page-range parser invocation."""

    pages: tuple[ParsedPage, ...]
    provenance: ExtractionProvenance


class PdfPageRangeBackend(Protocol):
    """Injectable expensive parser boundary used by deterministic tests."""

    def parse_range(
        self,
        source: Path,
        *,
        page_range: tuple[int, int],
        route: PdfPageRoute,
        file_sha256: str,
        config: PdfRoutingConfig,
    ) -> ParsedPageBatch: ...

    def provenance_for_route(
        self,
        route: PdfPageRoute,
        config: PdfRoutingConfig,
    ) -> ExtractionProvenance: ...


class LocalDoclingPageRangeBackend:
    """Lazy reusable Docling converters for layout and OCR page ranges."""

    def __init__(self) -> None:
        self._converters: dict[tuple[bool, bool, str], Any] = {}

    def parse_range(
        self,
        source: Path,
        *,
        page_range: tuple[int, int],
        route: PdfPageRoute,
        file_sha256: str,
        config: PdfRoutingConfig,
    ) -> ParsedPageBatch:
        do_ocr, force_full_page_ocr, _ = self._route_options(route)

        converter_key = (do_ocr, force_full_page_ocr, config.ocr_language)
        converter = self._converters.get(converter_key)
        if converter is None:
            converter = create_docling_converter(
                do_ocr=do_ocr,
                force_full_page_ocr=force_full_page_ocr,
                ocr_language=config.ocr_language,
            )
            self._converters[converter_key] = converter

        pages = parse_pdf_page_range_with_docling(
            source,
            page_range=page_range,
            do_ocr=do_ocr,
            force_full_page_ocr=force_full_page_ocr,
            ocr_language=config.ocr_language,
            converter=converter,
            file_sha256=file_sha256,
        )
        return ParsedPageBatch(
            pages=pages,
            provenance=self.provenance_for_route(route, config),
        )

    def provenance_for_route(
        self,
        route: PdfPageRoute,
        config: PdfRoutingConfig,
    ) -> ExtractionProvenance:
        do_ocr, force_full_page_ocr, extraction_method = self._route_options(route)
        return build_extraction_provenance(
            parser_name="docling",
            parser_version=version("docling"),
            extraction_method=extraction_method,
            configuration_profile=(
                f"route={route.value};ocr={str(do_ocr).lower()};"
                f"full_page={str(force_full_page_ocr).lower()};"
                f"language={config.ocr_language}"
            ),
            ocr_engine_name="rapidocr" if do_ocr else None,
            ocr_engine_version=version("rapidocr") if do_ocr else None,
        )

    @staticmethod
    def _route_options(
        route: PdfPageRoute,
    ) -> tuple[bool, bool, ExtractionMethod]:
        if route is PdfPageRoute.DOCLING:
            do_ocr = False
            force_full_page_ocr = False
            extraction_method = ExtractionMethod.LAYOUT
        elif route is PdfPageRoute.DOCLING_OCR:
            do_ocr = True
            force_full_page_ocr = False
            extraction_method = ExtractionMethod.LAYOUT
        elif route in {PdfPageRoute.OCR, PdfPageRoute.OCR_VERIFICATION}:
            do_ocr = True
            force_full_page_ocr = True
            extraction_method = ExtractionMethod.RERUN_OCR
        else:
            raise ValueError(f"route {route.value} does not use the Docling backend")
        return do_ocr, force_full_page_ocr, extraction_method


class CachingPageRangeBackend:
    """Read and write accepted normalized pages around an expensive backend."""

    def __init__(
        self,
        delegate: PdfPageRangeBackend,
        cache: PdfRoutingCache,
    ) -> None:
        self.delegate = delegate
        self.cache = cache
        self.events: list[RoutingCacheEvent] = []

    def provenance_for_route(
        self,
        route: PdfPageRoute,
        config: PdfRoutingConfig,
    ) -> ExtractionProvenance:
        return self.delegate.provenance_for_route(route, config)

    def parse_range(
        self,
        source: Path,
        *,
        page_range: tuple[int, int],
        route: PdfPageRoute,
        file_sha256: str,
        config: PdfRoutingConfig,
    ) -> ParsedPageBatch:
        provenance = self.provenance_for_route(route, config)
        cached_pages: list[ParsedPage] = []
        all_hit = True
        for page_number in range(page_range[0], page_range[1] + 1):
            page, event = self.cache.load_page(
                file_sha256=file_sha256,
                page_number=page_number,
                route=route,
                routing_profile=config.routing_profile,
                provenance=provenance,
            )
            self.events.append(event)
            if page is None:
                all_hit = False
            else:
                cached_pages.append(page)
        if all_hit:
            return ParsedPageBatch(pages=tuple(cached_pages), provenance=provenance)

        batch = self.delegate.parse_range(
            source,
            page_range=page_range,
            route=route,
            file_sha256=file_sha256,
            config=config,
        )
        for page in batch.pages:
            self.events.append(
                self.cache.save_page(
                    page,
                    file_sha256=file_sha256,
                    route=route,
                    routing_profile=config.routing_profile,
                    provenance=batch.provenance,
                )
            )
        return batch


def parse_pdf_auto(
    source: str | Path,
    config: PdfRoutingConfig | None = None,
    *,
    backend: PdfPageRangeBackend | None = None,
    cache: PdfRoutingCache | None = None,
    ocr_review_decisions: OcrReviewDecisions | None = None,
) -> PdfAutoRoutingResult:
    """Diagnose, route, parse, and merge one PDF before chunk construction."""
    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path.suffix.lower() != ".pdf":
        raise ValueError("source must be a PDF file")
    resolved_config = config or PdfRoutingConfig()
    file_sha256 = pdf_file_sha256(source_path)
    cache_events: list[RoutingCacheEvent] = []
    page_signals = None
    decisions = None
    signals_cache_hit = False
    if cache is not None:
        page_signals, event = cache.load_signals(
            file_sha256=file_sha256,
            extractor_version=resolved_config.signal_extractor_version,
            pymupdf_version=pymupdf.VersionBind,
        )
        cache_events.append(event)
        signals_cache_hit = page_signals is not None
    with pymupdf.open(source_path) as pymupdf_document:
        baseline = parsed_document_from_pymupdf(
            pymupdf_document,
            source_path,
            file_sha256,
        )
        if page_signals is None:
            page_signals = analyze_pymupdf_document(pymupdf_document)
    if cache is not None and not signals_cache_hit:
        cache_events.append(
            cache.save_signals(
                page_signals,
                file_sha256=baseline.file_sha256,
                extractor_version=resolved_config.signal_extractor_version,
                pymupdf_version=pymupdf.VersionBind,
            )
        )
    if cache is not None:
        decisions, event = cache.load_decisions(
            page_signals,
            routing_profile=resolved_config.routing_profile,
        )
        cache_events.append(event)
    if decisions is None:
        decisions = build_route_plan(page_signals, resolved_config)
        if cache is not None:
            cache_events.append(
                cache.save_decisions(
                    decisions,
                    page_signals,
                    routing_profile=resolved_config.routing_profile,
                )
            )

    resolved_backend = backend or LocalDoclingPageRangeBackend()
    caching_backend = (
        CachingPageRangeBackend(resolved_backend, cache) if cache is not None else None
    )
    result = route_pdf_document(
        source_path,
        baseline,
        page_signals,
        resolved_config,
        backend=caching_backend or resolved_backend,
        decisions=decisions,
        cache_events=tuple(cache_events),
        ocr_review_decisions=ocr_review_decisions,
    )
    if caching_backend is not None:
        result = replace(
            result,
            cache_events=result.cache_events + tuple(caching_backend.events),
        )
    return result


def route_pdf_document(
    source: str | Path,
    baseline: ParsedDocument,
    page_signals: tuple[PageRoutingSignals, ...],
    config: PdfRoutingConfig,
    *,
    backend: PdfPageRangeBackend,
    decisions: tuple[PageRouteDecision, ...] | None = None,
    cache_events: tuple[RoutingCacheEvent, ...] = (),
    ocr_review_decisions: OcrReviewDecisions | None = None,
) -> PdfAutoRoutingResult:
    """Resolve a preflighted document using injected expensive parser backends."""
    source_path = Path(source).resolve()
    decisions = decisions or build_route_plan(page_signals, config)
    if baseline.page_count != len(decisions):
        raise ValueError("baseline pages and route decisions must have the same length")
    review_decisions_by_page: dict[int, OcrReviewDecision] = {}
    if ocr_review_decisions is not None:
        if ocr_review_decisions.document_sha256 != baseline.file_sha256:
            raise ValueError("OCR review document_sha256 does not match the source PDF")
        if ocr_review_decisions.routing_profile != config.routing_profile:
            raise ValueError("OCR review routing_profile does not match the active configuration")
        review_decisions_by_page = ocr_review_decisions.by_page()

    pymupdf_native = build_extraction_provenance(
        parser_name="pymupdf",
        parser_version=pymupdf.VersionBind,
        extraction_method=ExtractionMethod.NATIVE,
        configuration_profile="pymupdf-native-fast-v1",
    )
    embedded_ocr = build_extraction_provenance(
        parser_name="pymupdf",
        parser_version=pymupdf.VersionBind,
        extraction_method=ExtractionMethod.EMBEDDED_OCR,
        configuration_profile="pymupdf-embedded-ocr-v1",
    )

    expensive_results: dict[int, ParsedPage] = {}
    expensive_provenance: dict[int, ExtractionProvenance] = {}
    range_failures: dict[int, tuple[PdfPageRoute, str]] = {}
    for route, page_range in _contiguous_expensive_ranges(decisions):
        try:
            batch = backend.parse_range(
                source_path,
                page_range=page_range,
                route=route,
                file_sha256=baseline.file_sha256,
                config=config,
            )
            expected_numbers = tuple(range(page_range[0], page_range[1] + 1))
            actual_numbers = tuple(page.page_number for page in batch.pages)
            if actual_numbers != expected_numbers:
                raise ValueError("backend page range output did not preserve original pages")
            for page in batch.pages:
                expensive_results[page.page_number] = page
                expensive_provenance[page.page_number] = batch.provenance
        except Exception as error:  # noqa: BLE001 - parser failures become typed diagnostics.
            failure_code = f"{type(error).__name__.upper()}_{str(error).strip() or 'PARSER_FAILED'}"
            for page_number in range(page_range[0], page_range[1] + 1):
                range_failures[page_number] = (route, failure_code)

    final_pages: list[ParsedPage] = []
    diagnostics: list[PageRoutingDiagnostic] = []
    accepted_provenances: dict[str, ExtractionProvenance] = {}
    for baseline_page, decision in zip(baseline.pages, decisions, strict=True):
        page, page_diagnostic, provenance = _resolve_page(
            baseline_page,
            decision,
            expensive_results.get(decision.page_number),
            expensive_provenance.get(decision.page_number),
            range_failures.get(decision.page_number),
            pymupdf_native,
            embedded_ocr,
            file_sha256=baseline.file_sha256,
            routing_profile=config.routing_profile,
            review_decision=review_decisions_by_page.get(decision.page_number),
        )
        final_pages.append(page)
        diagnostics.append(page_diagnostic)
        if provenance is not None:
            accepted_provenances[provenance.id] = provenance

    conflict_pages = {
        diagnostic.decision.page_number
        for diagnostic in diagnostics
        if diagnostic.ocr_conflict is not None
    }
    unused_review_pages = sorted(set(review_decisions_by_page) - conflict_pages)
    if unused_review_pages:
        pages = ", ".join(str(page_number) for page_number in unused_review_pages)
        raise ValueError(f"OCR review decisions do not match current conflicts on pages: {pages}")

    document_warnings = tuple(
        f"PAGE_{page.page_number}_{warning}" for page in final_pages for warning in page.warnings
    )
    document = ParsedDocument(
        document_id=baseline.document_id,
        source=baseline.source,
        file_sha256=baseline.file_sha256,
        parser_name="pdf-auto-router",
        parser_version=config.strategy_version,
        pages=tuple(final_pages),
        warnings=document_warnings,
        routing_profile=config.routing_profile,
        extraction_provenances=tuple(
            accepted_provenances[key] for key in sorted(accepted_provenances)
        ),
    )
    return PdfAutoRoutingResult(
        document=document,
        page_diagnostics=tuple(diagnostics),
        page_signals=page_signals,
        routing_profile=config.routing_profile,
        cache_events=cache_events,
    )


def build_extraction_provenance(
    *,
    parser_name: str,
    parser_version: str,
    extraction_method: ExtractionMethod,
    configuration_profile: str,
    ocr_engine_name: str | None = None,
    ocr_engine_version: str | None = None,
) -> ExtractionProvenance:
    """Create a deterministic compact provenance identity."""
    payload = json.dumps(
        {
            "parser_name": parser_name,
            "parser_version": parser_version,
            "extraction_method": extraction_method.value,
            "configuration_profile": configuration_profile,
            "ocr_engine_name": ocr_engine_name,
            "ocr_engine_version": ocr_engine_version,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return ExtractionProvenance(
        id=f"extract-{sha256(payload.encode()).hexdigest()[:16]}",
        parser_name=parser_name,
        parser_version=parser_version,
        extraction_method=extraction_method,
        configuration_profile=configuration_profile,
        ocr_engine_name=ocr_engine_name,
        ocr_engine_version=ocr_engine_version,
    )


def _contiguous_expensive_ranges(
    decisions: tuple[PageRouteDecision, ...],
) -> tuple[tuple[PdfPageRoute, tuple[int, int]], ...]:
    expensive_routes = {
        PdfPageRoute.DOCLING,
        PdfPageRoute.DOCLING_OCR,
        PdfPageRoute.OCR,
        PdfPageRoute.OCR_VERIFICATION,
    }
    ranges: list[tuple[PdfPageRoute, tuple[int, int]]] = []
    current_route: PdfPageRoute | None = None
    start_page = end_page = 0
    for decision in decisions:
        route = decision.selected_route
        if route not in expensive_routes:
            if current_route is not None:
                ranges.append((current_route, (start_page, end_page)))
                current_route = None
            continue
        if current_route is route and decision.page_number == end_page + 1:
            end_page = decision.page_number
            continue
        if current_route is not None:
            ranges.append((current_route, (start_page, end_page)))
        current_route = route
        start_page = end_page = decision.page_number
    if current_route is not None:
        ranges.append((current_route, (start_page, end_page)))
    return tuple(ranges)


def _resolve_page(
    baseline_page: ParsedPage,
    decision: PageRouteDecision,
    expensive_page: ParsedPage | None,
    expensive_page_provenance: ExtractionProvenance | None,
    range_failure: tuple[PdfPageRoute, str] | None,
    pymupdf_native: ExtractionProvenance,
    embedded_ocr: ExtractionProvenance,
    *,
    file_sha256: str,
    routing_profile: str,
    review_decision: OcrReviewDecision | None,
) -> tuple[ParsedPage, PageRoutingDiagnostic, ExtractionProvenance | None]:
    route = decision.selected_route
    if route is PdfPageRoute.NONE:
        page = replace(
            baseline_page,
            kind=PageKind.BLANK,
            status=ParseStatus.PARSED,
            blocks=(),
            pictures=(),
            warnings=("BLANK_PAGE",),
            extraction_provenance_id=None,
        )
        return (
            page,
            PageRoutingDiagnostic(
                decision=decision,
                attempts=(),
                final_status=ParseStatus.PARSED,
            ),
            None,
        )

    if route is PdfPageRoute.FAILED:
        failed = _failed_page(baseline_page, decision.reason_codes[0])
        return (
            failed,
            PageRoutingDiagnostic(
                decision=decision,
                attempts=(),
                final_status=ParseStatus.FAILED,
            ),
            None,
        )

    if route is PdfPageRoute.PYMUPDF:
        page = _attach_provenance(baseline_page, pymupdf_native)
        attempt = PageParseAttempt(
            route=route,
            status=ParseAttemptStatus.SUCCESS,
            provenance=pymupdf_native,
            warning_codes=page.warnings,
        )
        return (
            page,
            PageRoutingDiagnostic(
                decision=decision,
                attempts=(attempt,),
                final_status=page.status,
                final_provenance_id=pymupdf_native.id,
            ),
            pymupdf_native,
        )

    if range_failure is not None or expensive_page is None or expensive_page_provenance is None:
        failure_code = range_failure[1] if range_failure is not None else "PARSER_NO_PAGE_RESULT"
        failed = _failed_page(baseline_page, failure_code)
        attempt = PageParseAttempt(
            route=route,
            status=ParseAttemptStatus.FAILED,
            failure_code=failure_code,
        )
        return (
            failed,
            PageRoutingDiagnostic(
                decision=decision,
                attempts=(attempt,),
                final_status=ParseStatus.FAILED,
            ),
            None,
        )

    expensive_attempt = PageParseAttempt(
        route=route,
        status=(
            ParseAttemptStatus.SUCCESS
            if _page_is_usable(expensive_page)
            else ParseAttemptStatus.FAILED
        ),
        provenance=expensive_page_provenance,
        warning_codes=expensive_page.warnings,
        failure_code=None if _page_is_usable(expensive_page) else "PARSER_RESULT_UNUSABLE",
    )
    if not _page_is_usable(expensive_page):
        failed = _failed_page(baseline_page, "PARSER_RESULT_UNUSABLE")
        return (
            failed,
            PageRoutingDiagnostic(
                decision=decision,
                attempts=(expensive_attempt,),
                final_status=ParseStatus.FAILED,
            ),
            None,
        )

    if route is not PdfPageRoute.OCR_VERIFICATION:
        page = _attach_provenance(expensive_page, expensive_page_provenance)
        return (
            page,
            PageRoutingDiagnostic(
                decision=decision,
                attempts=(expensive_attempt,),
                final_status=ParseStatus.PARSED,
                final_provenance_id=expensive_page_provenance.id,
            ),
            expensive_page_provenance,
        )

    embedded_attempt = PageParseAttempt(
        route=PdfPageRoute.PYMUPDF,
        status=ParseAttemptStatus.SUCCESS,
        provenance=embedded_ocr,
        warning_codes=baseline_page.warnings,
    )
    baseline_text = _normalized_page_text(baseline_page)
    rerun_text = _normalized_page_text(expensive_page)
    attempts = (embedded_attempt, expensive_attempt)
    if baseline_text == rerun_text:
        page = _attach_provenance(
            replace(
                baseline_page,
                warnings=baseline_page.warnings + ("EMBEDDED_OCR_VERIFIED",),
            ),
            embedded_ocr,
        )
        return (
            page,
            PageRoutingDiagnostic(
                decision=decision,
                attempts=attempts,
                final_status=ParseStatus.PARSED,
                final_provenance_id=embedded_ocr.id,
            ),
            embedded_ocr,
        )
    if decision.text_quality is TextQualityStatus.UNUSABLE:
        page = _attach_provenance(
            replace(
                expensive_page,
                warnings=expensive_page.warnings + ("RERUN_OCR_REPLACED_BROKEN_TEXT_LAYER",),
            ),
            expensive_page_provenance,
        )
        return (
            page,
            PageRoutingDiagnostic(
                decision=decision,
                attempts=attempts,
                final_status=ParseStatus.PARSED,
                final_provenance_id=expensive_page_provenance.id,
            ),
            expensive_page_provenance,
        )

    conflict = _build_ocr_conflict_evidence(
        baseline_page,
        expensive_page,
        embedded_ocr,
        expensive_page_provenance,
        file_sha256=file_sha256,
        routing_profile=routing_profile,
    )
    if review_decision is not None:
        return _apply_ocr_review_decision(
            baseline_page,
            expensive_page,
            decision,
            attempts,
            conflict,
            review_decision,
            embedded_ocr,
            expensive_page_provenance,
        )

    failed = _failed_page(baseline_page, "OCR_RESULT_CONFLICT_REVIEW_REQUIRED")
    return (
        failed,
        PageRoutingDiagnostic(
            decision=decision,
            attempts=attempts,
            final_status=ParseStatus.FAILED,
            ocr_conflict=conflict,
        ),
        None,
    )


def _build_ocr_conflict_evidence(
    embedded_page: ParsedPage,
    rerun_page: ParsedPage,
    embedded_provenance: ExtractionProvenance,
    rerun_provenance: ExtractionProvenance,
    *,
    file_sha256: str,
    routing_profile: str,
) -> OcrConflictEvidence:
    embedded_text = _normalized_page_text(embedded_page)
    rerun_text = _normalized_page_text(rerun_page)
    embedded = OcrCandidateEvidence(
        text_sha256=sha256(embedded_text.encode()).hexdigest(),
        normalized_text=embedded_text,
        provenance=embedded_provenance,
    )
    rerun = OcrCandidateEvidence(
        text_sha256=sha256(rerun_text.encode()).hexdigest(),
        normalized_text=rerun_text,
        provenance=rerun_provenance,
    )
    identity_payload = json.dumps(
        {
            "schema": "pdf-ocr-conflict-v1",
            "document_sha256": file_sha256,
            "routing_profile": routing_profile,
            "page_number": embedded_page.page_number,
            "embedded_text_sha256": embedded.text_sha256,
            "embedded_provenance_id": embedded.provenance.id,
            "rerun_text_sha256": rerun.text_sha256,
            "rerun_provenance_id": rerun.provenance.id,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return OcrConflictEvidence(
        page_number=embedded_page.page_number,
        conflict_id=f"ocr-conflict-{sha256(identity_payload.encode()).hexdigest()[:20]}",
        embedded_ocr=embedded,
        rerun_ocr=rerun,
    )


def _apply_ocr_review_decision(
    embedded_page: ParsedPage,
    rerun_page: ParsedPage,
    route_decision: PageRouteDecision,
    attempts: tuple[PageParseAttempt, ...],
    conflict: OcrConflictEvidence,
    review_decision: OcrReviewDecision,
    embedded_provenance: ExtractionProvenance,
    rerun_provenance: ExtractionProvenance,
) -> tuple[ParsedPage, PageRoutingDiagnostic, ExtractionProvenance | None]:
    if review_decision.conflict_id != conflict.conflict_id:
        raise ValueError(
            f"OCR review conflict_id does not match current page {conflict.page_number} candidates"
        )
    review_decision_id = _review_decision_id(conflict, review_decision.decision)
    if review_decision.decision is OcrReviewChoice.REJECT_PAGE:
        failed = _failed_page(embedded_page, "OCR_RESULT_CONFLICT_REVIEW_REQUIRED")
        failed = replace(
            failed,
            warnings=failed.warnings + ("MANUAL_REVIEW_REJECTED_PAGE",),
        )
        return (
            failed,
            PageRoutingDiagnostic(
                decision=route_decision,
                attempts=attempts,
                final_status=ParseStatus.FAILED,
                ocr_conflict=conflict,
                review_decision_id=review_decision_id,
            ),
            None,
        )

    if review_decision.decision is OcrReviewChoice.ACCEPT_EMBEDDED_OCR:
        selected_page = embedded_page
        selected_provenance = embedded_provenance
        warning = "MANUAL_REVIEW_ACCEPTED_EMBEDDED_OCR"
    else:
        selected_page = rerun_page
        selected_provenance = rerun_provenance
        warning = "MANUAL_REVIEW_ACCEPTED_RERUN_OCR"
    reviewed_provenance = _reviewed_provenance(selected_provenance, review_decision_id)
    page = _attach_provenance(
        replace(selected_page, warnings=selected_page.warnings + (warning,)),
        reviewed_provenance,
    )
    return (
        page,
        PageRoutingDiagnostic(
            decision=route_decision,
            attempts=attempts,
            final_status=ParseStatus.PARSED,
            final_provenance_id=reviewed_provenance.id,
            ocr_conflict=conflict,
            review_decision_id=review_decision_id,
        ),
        reviewed_provenance,
    )


def _review_decision_id(
    conflict: OcrConflictEvidence,
    choice: OcrReviewChoice,
) -> str:
    payload = f"{conflict.conflict_id}\0{choice.value}"
    return f"ocr-review-{sha256(payload.encode()).hexdigest()[:20]}"


def _reviewed_provenance(
    provenance: ExtractionProvenance,
    review_decision_id: str,
) -> ExtractionProvenance:
    return build_extraction_provenance(
        parser_name=provenance.parser_name,
        parser_version=provenance.parser_version,
        extraction_method=provenance.extraction_method,
        configuration_profile=(
            f"{provenance.configuration_profile};manual_review={review_decision_id}"
        ),
        ocr_engine_name=provenance.ocr_engine_name,
        ocr_engine_version=provenance.ocr_engine_version,
    )


def _attach_provenance(page: ParsedPage, provenance: ExtractionProvenance) -> ParsedPage:
    blocks = tuple(replace(block, extraction_provenance_id=provenance.id) for block in page.blocks)
    return replace(page, blocks=blocks, extraction_provenance_id=provenance.id)


def _failed_page(page: ParsedPage, failure_code: str) -> ParsedPage:
    return replace(
        page,
        status=ParseStatus.FAILED,
        blocks=(),
        pictures=(),
        warnings=page.warnings + (failure_code,),
        extraction_provenance_id=None,
    )


def _page_is_usable(page: ParsedPage) -> bool:
    return (
        page.status is ParseStatus.PARSED
        and bool(page.blocks)
        and bool(_normalized_page_text(page))
    )


def _normalized_page_text(page: ParsedPage) -> str:
    text = "\n".join(
        block.text for block in sorted(page.blocks, key=lambda item: item.reading_order)
    )
    return " ".join(unicodedata.normalize("NFKC", text).split())
