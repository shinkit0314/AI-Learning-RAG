"""Deterministic page diagnostics and routing policy for PDF parser selection."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, fields
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any

import pymupdf

from ai_learning_rag.pdf_schema import PageKind


class SignalStatus(StrEnum):
    """Outcome of one independently evaluated quality signal."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class TextQualityStatus(StrEnum):
    """Structural usability of a PDF text layer, not semantic correctness."""

    USABLE = "usable"
    SUSPICIOUS = "suspicious"
    UNUSABLE = "unusable"


class LayoutComplexity(StrEnum):
    """Whether native PDF block order is safe enough for the fast path."""

    SIMPLE = "simple"
    UNCERTAIN = "uncertain"
    COMPLEX = "complex"


class PdfPageRoute(StrEnum):
    """Parser route selected before structure-aware chunking."""

    PYMUPDF = "pymupdf"
    DOCLING = "docling"
    DOCLING_OCR = "docling_ocr"
    OCR = "ocr"
    OCR_VERIFICATION = "ocr_verification"
    FAILED = "failed"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class PdfRoutingConfig:
    """Versioned thresholds used by the deterministic page router."""

    strategy_version: str = "pdf-page-routing-v1"
    signal_extractor_version: str = "pdf-page-signals-v1"
    min_useful_characters: int = 20
    full_page_image_union_ratio: float = 0.80
    dominant_image_ratio: float = 0.75
    unicode_anomaly_warn_ratio: float = 0.001
    unicode_anomaly_fail_ratio: float = 0.02
    private_use_warn_ratio: float = 0.001
    private_use_fail_ratio: float = 0.05
    invalid_bbox_warn_ratio: float = 0.01
    invalid_bbox_fail_ratio: float = 0.05
    duplicate_glyph_warn_ratio: float = 0.02
    duplicate_glyph_fail_ratio: float = 0.10
    latin_fragment_warn_ratio: float = 0.40
    latin_fragment_fail_ratio: float = 0.75
    mixed_script_warn_ratio: float = 0.0
    mixed_script_fail_ratio: float = 0.05
    mixed_script_affects_text_quality: bool = False
    column_min_blocks: int = 2
    column_min_gutter_ratio: float = 0.08
    column_vertical_overlap_ratio: float = 0.50
    many_text_blocks: int = 20
    many_drawings: int = 12
    weak_complex_signal_count: int = 2
    ocr_language: str = "ch"
    hidden_ocr_policy: str = "full-page-verify"
    fallback_policy: str = "route-specific-fail-closed-v1"

    def __post_init__(self) -> None:
        for field_name in (
            "strategy_version",
            "signal_extractor_version",
            "ocr_language",
            "hidden_ocr_policy",
            "fallback_policy",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must not be blank")
        if self.min_useful_characters < 1:
            raise ValueError("min_useful_characters must be positive")
        for field_name in (
            "full_page_image_union_ratio",
            "dominant_image_ratio",
            "unicode_anomaly_warn_ratio",
            "unicode_anomaly_fail_ratio",
            "private_use_warn_ratio",
            "private_use_fail_ratio",
            "invalid_bbox_warn_ratio",
            "invalid_bbox_fail_ratio",
            "duplicate_glyph_warn_ratio",
            "duplicate_glyph_fail_ratio",
            "latin_fragment_warn_ratio",
            "latin_fragment_fail_ratio",
            "mixed_script_warn_ratio",
            "mixed_script_fail_ratio",
            "column_min_gutter_ratio",
            "column_vertical_overlap_ratio",
        ):
            value = float(getattr(self, field_name))
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be between 0 and 1")
        for warn_name, fail_name in (
            ("unicode_anomaly_warn_ratio", "unicode_anomaly_fail_ratio"),
            ("private_use_warn_ratio", "private_use_fail_ratio"),
            ("invalid_bbox_warn_ratio", "invalid_bbox_fail_ratio"),
            ("duplicate_glyph_warn_ratio", "duplicate_glyph_fail_ratio"),
            ("latin_fragment_warn_ratio", "latin_fragment_fail_ratio"),
            ("mixed_script_warn_ratio", "mixed_script_fail_ratio"),
        ):
            if getattr(self, warn_name) > getattr(self, fail_name):
                raise ValueError(f"{warn_name} must not exceed {fail_name}")
        for field_name in (
            "column_min_blocks",
            "many_text_blocks",
            "many_drawings",
            "weak_complex_signal_count",
        ):
            if int(getattr(self, field_name)) < 1:
                raise ValueError(f"{field_name} must be positive")

    @property
    def profile_payload(self) -> dict[str, Any]:
        """Return the behavior-affecting configuration in canonical field order."""
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @property
    def routing_profile(self) -> str:
        """Return a stable compact identity for this routing policy."""
        payload = json.dumps(
            self.profile_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return f"{self.strategy_version}:{sha256(payload.encode()).hexdigest()}"

    @classmethod
    def from_json(cls, path: str | Path) -> PdfRoutingConfig:
        """Load a complete or partial candidate configuration without code changes."""
        config_path = Path(path)
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("routing configuration must be a JSON object")
        known_fields = {field.name for field in fields(cls)}
        unknown_fields = sorted(set(payload) - known_fields)
        if unknown_fields:
            raise ValueError(f"unknown routing configuration fields: {', '.join(unknown_fields)}")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class RoutingSignal:
    """One inspectable signal with the threshold result that used it."""

    name: str
    value: int | float | bool | str
    status: SignalStatus
    reason: str
    affects_text_quality: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("signal name must not be blank")
        if not self.reason.strip():
            raise ValueError("signal reason must not be blank")


@dataclass(frozen=True, slots=True)
class PageRoutingSignals:
    """Raw deterministic facts extracted from one PDF page."""

    page_number: int
    width: float
    height: float
    text_character_count: int
    text_block_count: int
    image_count: int
    image_union_coverage: float
    largest_image_coverage: float
    drawing_count: int
    unicode_anomaly_ratio: float
    private_use_ratio: float
    invalid_bbox_ratio: float
    duplicate_glyph_ratio: float
    latin_fragment_ratio: float
    mixed_script_token_ratio: float
    dominant_script: str
    column_count_candidate: int
    column_left_block_count: int
    column_right_block_count: int
    column_gutter_ratio: float
    column_vertical_overlap: float

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("page_number must start at 1")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("page dimensions must be positive")
        for field_name in (
            "text_character_count",
            "text_block_count",
            "image_count",
            "drawing_count",
            "column_count_candidate",
            "column_left_block_count",
            "column_right_block_count",
        ):
            if int(getattr(self, field_name)) < 0:
                raise ValueError(f"{field_name} must not be negative")
        if not self.dominant_script.strip():
            raise ValueError("dominant_script must not be blank")
        for field_name in (
            "image_union_coverage",
            "largest_image_coverage",
            "unicode_anomaly_ratio",
            "private_use_ratio",
            "invalid_bbox_ratio",
            "duplicate_glyph_ratio",
            "latin_fragment_ratio",
            "mixed_script_token_ratio",
            "column_gutter_ratio",
            "column_vertical_overlap",
        ):
            value = float(getattr(self, field_name))
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class PageRouteDecision:
    """Explainable parser route selected from raw page signals."""

    page_number: int
    source_kind: PageKind
    text_quality: TextQualityStatus
    layout_complexity: LayoutComplexity
    selected_route: PdfPageRoute
    reason_codes: tuple[str, ...]
    signal_results: tuple[RoutingSignal, ...]

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("page_number must start at 1")
        if not self.reason_codes or any(not code.strip() for code in self.reason_codes):
            raise ValueError("route decision must contain non-blank reason codes")


def analyze_pdf_pages(source: str | Path) -> tuple[PageRoutingSignals, ...]:
    """Extract raw page facts without loading Docling or an OCR model."""
    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path.suffix.lower() != ".pdf":
        raise ValueError("source must be a PDF file")

    with pymupdf.open(source_path) as document:
        return analyze_pymupdf_document(document)


def analyze_pymupdf_document(
    document: pymupdf.Document,
) -> tuple[PageRoutingSignals, ...]:
    """Extract routing signals from an already-open PyMuPDF document."""
    if document.needs_pass:
        raise ValueError("password-protected PDFs are not supported")
    return tuple(_analyze_page(page, page_index + 1) for page_index, page in enumerate(document))


def build_route_plan(
    page_signals: tuple[PageRoutingSignals, ...],
    config: PdfRoutingConfig | None = None,
) -> tuple[PageRouteDecision, ...]:
    """Apply a versioned policy to already extracted page facts."""
    resolved_config = config or PdfRoutingConfig()
    expected = list(range(1, len(page_signals) + 1))
    actual = [signals.page_number for signals in page_signals]
    if actual != expected:
        raise ValueError("page signals must be contiguous and ordered from 1")
    return tuple(_decide_page_route(signals, resolved_config) for signals in page_signals)


def _analyze_page(page: pymupdf.Page, page_number: int) -> PageRoutingSignals:
    page_rect = pymupdf.Rect(page.rect)
    page_area = page_rect.get_area()
    image_info = page.get_image_info()
    image_rects = [
        clipped
        for image in image_info
        if not (clipped := pymupdf.Rect(image["bbox"]) & page_rect).is_empty
        and not clipped.is_infinite
    ]
    image_union_coverage = _rectangle_union_area(image_rects) / page_area if page_area else 0.0
    largest_image_coverage = (
        max((rect.get_area() for rect in image_rects), default=0.0) / page_area
        if page_area
        else 0.0
    )

    text_blocks = [
        block
        for block in page.get_text("blocks", sort=True)
        if len(block) >= 7 and block[6] == 0 and block[4].strip()
    ]
    raw = page.get_text("rawdict", sort=True)
    spans = [
        span
        for block in raw.get("blocks", ())
        if block.get("type") == 0
        for line in block.get("lines", ())
        for span in line.get("spans", ())
    ]
    characters = [character for span in spans for character in span.get("chars", ())]
    character_values = [str(character.get("c", "")) for character in characters]
    non_whitespace = [character for character in character_values if not character.isspace()]
    character_count = len(non_whitespace)
    denominator = max(character_count, 1)
    unicode_anomalies = sum(_is_unicode_anomaly(character) for character in non_whitespace)
    private_use = sum(
        any(unicodedata.category(codepoint) == "Co" for codepoint in character)
        for character in non_whitespace
    )

    invalid_bbox_count = 0
    glyph_keys: list[tuple[str, float, float, float, float]] = []
    for character in characters:
        bbox = character.get("bbox")
        if not _valid_bbox(bbox, page_rect):
            invalid_bbox_count += 1
            continue
        glyph_keys.append(
            (
                str(character.get("c", "")),
                *(round(float(value), 2) for value in bbox),
            )
        )
    duplicate_count = len(glyph_keys) - len(set(glyph_keys))

    latin_spans = []
    for span in spans:
        text = "".join(str(character.get("c", "")) for character in span.get("chars", ()))
        latin_count = sum(character.isascii() and character.isalpha() for character in text)
        if latin_count:
            latin_spans.append((latin_count, len(text.strip())))
    latin_fragment_ratio = (
        sum(length <= 1 for _, length in latin_spans) / len(latin_spans) if latin_spans else 0.0
    )

    extracted_text = "\n".join(str(block[4]) for block in text_blocks)
    tokens = re.findall(r"\S+", extracted_text)
    mixed_script_tokens = sum(_has_mixed_latin_cjk(token) for token in tokens)
    mixed_script_ratio = mixed_script_tokens / len(tokens) if tokens else 0.0
    dominant_script = _dominant_script(non_whitespace)
    column_count, left_count, right_count, column_gutter, column_overlap = _column_candidate(
        text_blocks,
        page_rect,
    )

    try:
        drawing_count = len(page.get_drawings())
    except (RuntimeError, ValueError):
        drawing_count = 0

    return PageRoutingSignals(
        page_number=page_number,
        width=float(page_rect.width),
        height=float(page_rect.height),
        text_character_count=character_count,
        text_block_count=len(text_blocks),
        image_count=len(image_info),
        image_union_coverage=min(1.0, image_union_coverage),
        largest_image_coverage=min(1.0, largest_image_coverage),
        drawing_count=drawing_count,
        unicode_anomaly_ratio=unicode_anomalies / denominator,
        private_use_ratio=private_use / denominator,
        invalid_bbox_ratio=invalid_bbox_count / max(len(characters), 1),
        duplicate_glyph_ratio=duplicate_count / max(len(glyph_keys), 1),
        latin_fragment_ratio=latin_fragment_ratio,
        mixed_script_token_ratio=mixed_script_ratio,
        dominant_script=dominant_script,
        column_count_candidate=column_count,
        column_left_block_count=left_count,
        column_right_block_count=right_count,
        column_gutter_ratio=column_gutter,
        column_vertical_overlap=column_overlap,
    )


def _decide_page_route(
    signals: PageRoutingSignals,
    config: PdfRoutingConfig,
) -> PageRouteDecision:
    signal_results = _evaluate_text_signals(signals, config)
    text_quality = _text_quality(signal_results)
    source_kind = _source_kind(signals, text_quality, config)
    layout_complexity, layout_reasons = _layout_complexity(signals, config)

    if source_kind is PageKind.BLANK:
        route = PdfPageRoute.NONE
        reasons = ("BLANK_PAGE",)
    elif source_kind is PageKind.SCANNED_IMAGE:
        route = PdfPageRoute.OCR
        reasons = ("SCANNED_IMAGE_NO_TEXT",)
    elif source_kind is PageKind.SCANNED_WITH_OCR:
        route = PdfPageRoute.OCR_VERIFICATION
        reasons = ("HIDDEN_OCR_VERIFICATION_REQUIRED",)
    elif source_kind is PageKind.DAMAGED_TEXT_LAYER:
        route = PdfPageRoute.OCR
        reasons = ("DAMAGED_TEXT_LAYER",)
    elif source_kind is PageKind.MIXED:
        route = PdfPageRoute.DOCLING
        reasons = ("MIXED_PAGE_REQUIRES_DOCLING",)
    elif layout_complexity is LayoutComplexity.SIMPLE:
        route = PdfPageRoute.PYMUPDF
        reasons = ("NATIVE_TEXT_USABLE_SIMPLE_LAYOUT",)
    else:
        route = PdfPageRoute.DOCLING
        reasons = layout_reasons

    if route in {
        PdfPageRoute.DOCLING_OCR,
        PdfPageRoute.OCR,
        PdfPageRoute.OCR_VERIFICATION,
    } and not _ocr_language_supports(signals.dominant_script, config.ocr_language):
        route = PdfPageRoute.FAILED
        reasons = ("OCR_LANGUAGE_UNSUPPORTED",)

    return PageRouteDecision(
        page_number=signals.page_number,
        source_kind=source_kind,
        text_quality=text_quality,
        layout_complexity=layout_complexity,
        selected_route=route,
        reason_codes=reasons,
        signal_results=signal_results,
    )


def _evaluate_text_signals(
    signals: PageRoutingSignals,
    config: PdfRoutingConfig,
) -> tuple[RoutingSignal, ...]:
    results = [
        RoutingSignal(
            name="text_character_count",
            value=signals.text_character_count,
            status=(
                SignalStatus.PASS
                if signals.text_character_count >= config.min_useful_characters
                else SignalStatus.WARN
                if signals.text_character_count > 0
                else SignalStatus.FAIL
            ),
            reason=(f"Useful text requires at least {config.min_useful_characters} characters."),
        )
    ]
    for name, value, warn_threshold, fail_threshold, affects_text_quality in (
        (
            "unicode_anomaly_ratio",
            signals.unicode_anomaly_ratio,
            config.unicode_anomaly_warn_ratio,
            config.unicode_anomaly_fail_ratio,
            True,
        ),
        (
            "private_use_ratio",
            signals.private_use_ratio,
            config.private_use_warn_ratio,
            config.private_use_fail_ratio,
            True,
        ),
        (
            "invalid_bbox_ratio",
            signals.invalid_bbox_ratio,
            config.invalid_bbox_warn_ratio,
            config.invalid_bbox_fail_ratio,
            True,
        ),
        (
            "duplicate_glyph_ratio",
            signals.duplicate_glyph_ratio,
            config.duplicate_glyph_warn_ratio,
            config.duplicate_glyph_fail_ratio,
            True,
        ),
        (
            "latin_fragment_ratio",
            signals.latin_fragment_ratio,
            config.latin_fragment_warn_ratio,
            config.latin_fragment_fail_ratio,
            True,
        ),
        (
            "mixed_script_token_ratio",
            signals.mixed_script_token_ratio,
            config.mixed_script_warn_ratio,
            config.mixed_script_fail_ratio,
            config.mixed_script_affects_text_quality,
        ),
    ):
        status = (
            SignalStatus.FAIL
            if value > fail_threshold
            else SignalStatus.WARN
            if value > warn_threshold
            else SignalStatus.PASS
        )
        results.append(
            RoutingSignal(
                name=name,
                value=value,
                status=status,
                reason=f"warn>{warn_threshold:g}; fail>{fail_threshold:g}",
                affects_text_quality=affects_text_quality,
            )
        )
    return tuple(results)


def _text_quality(signal_results: tuple[RoutingSignal, ...]) -> TextQualityStatus:
    statuses = {signal.status for signal in signal_results if signal.affects_text_quality}
    if SignalStatus.FAIL in statuses:
        return TextQualityStatus.UNUSABLE
    if SignalStatus.WARN in statuses:
        return TextQualityStatus.SUSPICIOUS
    return TextQualityStatus.USABLE


def _source_kind(
    signals: PageRoutingSignals,
    text_quality: TextQualityStatus,
    config: PdfRoutingConfig,
) -> PageKind:
    has_text = signals.text_character_count > 0
    has_visual_content = signals.image_count > 0 or signals.drawing_count > 0
    scan_like = (
        signals.image_union_coverage >= config.full_page_image_union_ratio
        and signals.largest_image_coverage >= config.dominant_image_ratio
    )
    if not has_text and not has_visual_content:
        return PageKind.BLANK
    if scan_like and has_text:
        return PageKind.SCANNED_WITH_OCR
    if scan_like:
        return PageKind.SCANNED_IMAGE
    if has_text and text_quality is TextQualityStatus.UNUSABLE:
        return PageKind.DAMAGED_TEXT_LAYER
    if has_text and text_quality is TextQualityStatus.USABLE:
        return PageKind.NATIVE_TEXT
    return PageKind.MIXED


def _layout_complexity(
    signals: PageRoutingSignals,
    config: PdfRoutingConfig,
) -> tuple[LayoutComplexity, tuple[str, ...]]:
    strong_reasons = []
    if (
        signals.column_count_candidate >= 2
        and min(signals.column_left_block_count, signals.column_right_block_count)
        >= config.column_min_blocks
        and signals.column_gutter_ratio >= config.column_min_gutter_ratio
        and signals.column_vertical_overlap >= config.column_vertical_overlap_ratio
    ):
        strong_reasons.append("LAYOUT_MULTICOLUMN")

    weak_reasons = []
    if signals.text_block_count >= config.many_text_blocks:
        weak_reasons.append("LAYOUT_MANY_TEXT_BLOCKS")
    if signals.drawing_count >= config.many_drawings:
        weak_reasons.append("LAYOUT_MANY_DRAWINGS")
    if signals.image_count >= 2 and signals.text_block_count >= 4:
        weak_reasons.append("LAYOUT_IMAGES_INTERLEAVED_WITH_TEXT")

    if strong_reasons or len(weak_reasons) >= config.weak_complex_signal_count:
        return LayoutComplexity.COMPLEX, tuple(strong_reasons + weak_reasons)
    if weak_reasons:
        return LayoutComplexity.UNCERTAIN, tuple(weak_reasons)
    return LayoutComplexity.SIMPLE, ("LAYOUT_SIMPLE",)


def _rectangle_union_area(rectangles: list[pymupdf.Rect]) -> float:
    if not rectangles:
        return 0.0
    x_coordinates = sorted({coordinate for rect in rectangles for coordinate in (rect.x0, rect.x1)})
    area = 0.0
    for left, right in zip(x_coordinates, x_coordinates[1:], strict=False):
        if right <= left:
            continue
        intervals = sorted(
            (rect.y0, rect.y1) for rect in rectangles if rect.x0 < right and rect.x1 > left
        )
        covered_height = 0.0
        current_start = current_end = None
        for start, end in intervals:
            if current_start is None:
                current_start, current_end = start, end
            elif start <= current_end:
                current_end = max(current_end, end)
            else:
                covered_height += current_end - current_start
                current_start, current_end = start, end
        if current_start is not None:
            covered_height += current_end - current_start
        area += (right - left) * covered_height
    return area


def _valid_bbox(raw_bbox: Any, page_rect: pymupdf.Rect) -> bool:
    if not isinstance(raw_bbox, list | tuple) or len(raw_bbox) != 4:
        return False
    try:
        rect = pymupdf.Rect(*(float(value) for value in raw_bbox))
    except (TypeError, ValueError):
        return False
    tolerance = 1.0
    return (
        not rect.is_empty
        and not rect.is_infinite
        and rect.x0 >= page_rect.x0 - tolerance
        and rect.y0 >= page_rect.y0 - tolerance
        and rect.x1 <= page_rect.x1 + tolerance
        and rect.y1 <= page_rect.y1 + tolerance
    )


def _is_unicode_anomaly(character: str) -> bool:
    if not character:
        return True
    return any(
        codepoint == "\ufffd" or unicodedata.category(codepoint) in {"Cc", "Cs"}
        for codepoint in character
    )


def _has_mixed_latin_cjk(token: str) -> bool:
    has_latin = bool(re.search(r"[A-Za-z]", token))
    has_cjk = bool(re.search(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", token))
    return has_latin and has_cjk


def _dominant_script(characters: list[str]) -> str:
    counts = {"latin": 0, "han": 0, "kana": 0, "hangul": 0}
    for character in characters:
        for codepoint in character:
            value = ord(codepoint)
            if codepoint.isascii() and codepoint.isalpha():
                counts["latin"] += 1
            elif 0x3400 <= value <= 0x9FFF:
                counts["han"] += 1
            elif 0x3040 <= value <= 0x30FF:
                counts["kana"] += 1
            elif 0xAC00 <= value <= 0xD7AF:
                counts["hangul"] += 1
    highest = max(counts.values(), default=0)
    if highest == 0:
        return "unknown"
    leaders = [script for script, count in counts.items() if count == highest]
    return leaders[0] if len(leaders) == 1 else "mixed"


def _ocr_language_supports(dominant_script: str, ocr_language: str) -> bool:
    normalized_language = ocr_language.casefold()
    if dominant_script in {"unknown", "mixed"}:
        return True
    if normalized_language in {"ch", "zh", "zh-hans", "iso:zh-hans"}:
        return dominant_script in {"han", "latin"}
    if normalized_language in {"en", "english"}:
        return dominant_script == "latin"
    return True


def _column_candidate(
    text_blocks: list[tuple[Any, ...]],
    page_rect: pymupdf.Rect,
) -> tuple[int, int, int, float, float]:
    candidates = [
        pymupdf.Rect(block[:4])
        for block in text_blocks
        if 0 < float(block[2]) - float(block[0]) <= page_rect.width * 0.70
    ]
    if len(candidates) < 4:
        return (1 if candidates else 0), len(candidates), 0, 0.0, 0.0
    centers = sorted((rect.x0 + rect.x1) / 2 for rect in candidates)
    gaps = [
        (right - left, index)
        for index, (left, right) in enumerate(zip(centers, centers[1:], strict=False))
    ]
    gap, split_index = max(gaps)
    split_x = (centers[split_index] + centers[split_index + 1]) / 2
    left_column = [rect for rect in candidates if (rect.x0 + rect.x1) / 2 < split_x]
    right_column = [rect for rect in candidates if (rect.x0 + rect.x1) / 2 >= split_x]
    if not left_column or not right_column:
        return 1, len(candidates), 0, gap / page_rect.width, 0.0
    left_range = (min(rect.y0 for rect in left_column), max(rect.y1 for rect in left_column))
    right_range = (min(rect.y0 for rect in right_column), max(rect.y1 for rect in right_column))
    overlap = max(0.0, min(left_range[1], right_range[1]) - max(left_range[0], right_range[0]))
    shortest_height = min(left_range[1] - left_range[0], right_range[1] - right_range[0])
    return (
        2,
        len(left_column),
        len(right_column),
        gap / page_rect.width,
        overlap / shortest_height if shortest_height > 0 else 0.0,
    )
