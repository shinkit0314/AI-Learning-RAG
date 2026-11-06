"""Content-addressed JSON cache for deterministic PDF routing artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from ai_learning_rag.pdf_routing import (
    LayoutComplexity,
    PageRouteDecision,
    PageRoutingSignals,
    PdfPageRoute,
    RoutingSignal,
    SignalStatus,
    TextQualityStatus,
)
from ai_learning_rag.pdf_schema import (
    BlockType,
    BoundingBox,
    ExtractionMethod,
    ExtractionProvenance,
    PageKind,
    ParsedBlock,
    ParsedPage,
    ParsedPicture,
    ParseStatus,
)


class RoutingCacheMode(StrEnum):
    """Whether persistent routing artifacts may be read or written."""

    OFF = "off"
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"


class RoutingCacheStatus(StrEnum):
    """Observable cache event kept outside deterministic routing identity."""

    HIT = "hit"
    MISS = "miss"
    WRITTEN = "written"
    CORRUPT = "corrupt"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class RoutingCacheEvent:
    """Operational cache evidence excluded from routing fingerprints."""

    stage: str
    key: str
    status: RoutingCacheStatus
    message: str = ""


class PdfRoutingCache:
    """Versioned JSON cache that never deletes or silently repairs user files."""

    SIGNAL_SCHEMA = "pdf-routing-signals-v1"
    DECISION_SCHEMA = "pdf-route-decisions-v1"
    PAGE_SCHEMA = "pdf-routed-page-v1"

    def __init__(
        self,
        root: str | Path,
        mode: RoutingCacheMode = RoutingCacheMode.READ_WRITE,
    ) -> None:
        self.root = Path(root)
        self.mode = mode

    def load_signals(
        self,
        *,
        file_sha256: str,
        extractor_version: str,
        pymupdf_version: str,
    ) -> tuple[tuple[PageRoutingSignals, ...] | None, RoutingCacheEvent]:
        key = _cache_key(file_sha256, extractor_version, pymupdf_version, self.SIGNAL_SCHEMA)
        payload, event = self._read("signals", key)
        if payload is None:
            return None, event
        try:
            signals = tuple(PageRoutingSignals(**item) for item in payload["signals"])
        except (KeyError, TypeError, ValueError) as error:
            return None, RoutingCacheEvent("signals", key, RoutingCacheStatus.CORRUPT, str(error))
        return signals, event

    def save_signals(
        self,
        signals: tuple[PageRoutingSignals, ...],
        *,
        file_sha256: str,
        extractor_version: str,
        pymupdf_version: str,
    ) -> RoutingCacheEvent:
        key = _cache_key(file_sha256, extractor_version, pymupdf_version, self.SIGNAL_SCHEMA)
        return self._write(
            "signals",
            key,
            {"schema": self.SIGNAL_SCHEMA, "signals": [asdict(item) for item in signals]},
        )

    def load_decisions(
        self,
        signals: tuple[PageRoutingSignals, ...],
        *,
        routing_profile: str,
    ) -> tuple[tuple[PageRouteDecision, ...] | None, RoutingCacheEvent]:
        signals_fingerprint = _json_fingerprint([asdict(item) for item in signals])
        key = _cache_key(signals_fingerprint, routing_profile, self.DECISION_SCHEMA)
        payload, event = self._read("decisions", key)
        if payload is None:
            return None, event
        try:
            decisions = tuple(_decision_from_dict(item) for item in payload["decisions"])
        except (KeyError, TypeError, ValueError) as error:
            return None, RoutingCacheEvent("decisions", key, RoutingCacheStatus.CORRUPT, str(error))
        return decisions, event

    def save_decisions(
        self,
        decisions: tuple[PageRouteDecision, ...],
        signals: tuple[PageRoutingSignals, ...],
        *,
        routing_profile: str,
    ) -> RoutingCacheEvent:
        signals_fingerprint = _json_fingerprint([asdict(item) for item in signals])
        key = _cache_key(signals_fingerprint, routing_profile, self.DECISION_SCHEMA)
        return self._write(
            "decisions",
            key,
            {
                "schema": self.DECISION_SCHEMA,
                "decisions": [asdict(item) for item in decisions],
            },
        )

    def load_page(
        self,
        *,
        file_sha256: str,
        page_number: int,
        route: PdfPageRoute,
        routing_profile: str,
        provenance: ExtractionProvenance,
    ) -> tuple[ParsedPage | None, RoutingCacheEvent]:
        key = self.page_key(
            file_sha256=file_sha256,
            page_number=page_number,
            route=route,
            routing_profile=routing_profile,
            provenance=provenance,
        )
        payload, event = self._read("pages", key)
        if payload is None:
            return None, event
        try:
            cached_provenance = _provenance_from_dict(payload["provenance"])
            if cached_provenance != provenance:
                raise ValueError("cached extraction provenance does not match the cache key")
            page = _page_from_dict(payload["page"])
        except (KeyError, TypeError, ValueError) as error:
            return None, RoutingCacheEvent("pages", key, RoutingCacheStatus.CORRUPT, str(error))
        return page, event

    def save_page(
        self,
        page: ParsedPage,
        *,
        file_sha256: str,
        route: PdfPageRoute,
        routing_profile: str,
        provenance: ExtractionProvenance,
    ) -> RoutingCacheEvent:
        if page.status is not ParseStatus.PARSED or not page.blocks:
            return RoutingCacheEvent(
                "pages",
                "",
                RoutingCacheStatus.SKIPPED,
                "only accepted non-empty pages are cached",
            )
        key = self.page_key(
            file_sha256=file_sha256,
            page_number=page.page_number,
            route=route,
            routing_profile=routing_profile,
            provenance=provenance,
        )
        return self._write(
            "pages",
            key,
            {
                "schema": self.PAGE_SCHEMA,
                "provenance": asdict(provenance),
                "page": asdict(page),
            },
        )

    def page_key(
        self,
        *,
        file_sha256: str,
        page_number: int,
        route: PdfPageRoute,
        routing_profile: str,
        provenance: ExtractionProvenance,
    ) -> str:
        return _cache_key(
            file_sha256,
            str(page_number),
            route.value,
            routing_profile,
            provenance.id,
            self.PAGE_SCHEMA,
        )

    def _read(self, stage: str, key: str) -> tuple[dict[str, Any] | None, RoutingCacheEvent]:
        if self.mode is RoutingCacheMode.OFF:
            return None, RoutingCacheEvent(stage, key, RoutingCacheStatus.SKIPPED)
        path = self.root / stage / f"{key}.json"
        if not path.is_file():
            return None, RoutingCacheEvent(stage, key, RoutingCacheStatus.MISS)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("cache payload must be a JSON object")
        except (OSError, json.JSONDecodeError, ValueError) as error:
            return None, RoutingCacheEvent(stage, key, RoutingCacheStatus.CORRUPT, str(error))
        return payload, RoutingCacheEvent(stage, key, RoutingCacheStatus.HIT)

    def _write(self, stage: str, key: str, payload: dict[str, Any]) -> RoutingCacheEvent:
        if self.mode is not RoutingCacheMode.READ_WRITE:
            return RoutingCacheEvent(stage, key, RoutingCacheStatus.SKIPPED)
        directory = self.root / stage
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{key}.json"
        serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=directory,
                prefix=f".{key}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
                temporary_path = Path(stream.name)
            temporary_path.replace(target)
        except OSError as error:
            return RoutingCacheEvent(stage, key, RoutingCacheStatus.CORRUPT, str(error))
        return RoutingCacheEvent(stage, key, RoutingCacheStatus.WRITTEN)


def _cache_key(*parts: str) -> str:
    return sha256("\0".join(parts).encode()).hexdigest()


def _json_fingerprint(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(serialized.encode()).hexdigest()


def _decision_from_dict(payload: dict[str, Any]) -> PageRouteDecision:
    return PageRouteDecision(
        page_number=int(payload["page_number"]),
        source_kind=PageKind(payload["source_kind"]),
        text_quality=TextQualityStatus(payload["text_quality"]),
        layout_complexity=LayoutComplexity(payload["layout_complexity"]),
        selected_route=PdfPageRoute(payload["selected_route"]),
        reason_codes=tuple(payload["reason_codes"]),
        signal_results=tuple(
            RoutingSignal(
                name=item["name"],
                value=item["value"],
                status=SignalStatus(item["status"]),
                reason=item["reason"],
                affects_text_quality=bool(item.get("affects_text_quality", True)),
            )
            for item in payload["signal_results"]
        ),
    )


def _page_from_dict(payload: dict[str, Any]) -> ParsedPage:
    blocks = tuple(
        ParsedBlock(
            id=item["id"],
            page_number=int(item["page_number"]),
            block_type=BlockType(item["block_type"]),
            text=item["text"],
            bbox=BoundingBox(**item["bbox"]),
            reading_order=int(item["reading_order"]),
            extraction_method=ExtractionMethod(item["extraction_method"]),
            heading_level=item.get("heading_level"),
            parent_picture_id=item.get("parent_picture_id"),
            confidence=item.get("confidence"),
            warnings=tuple(item.get("warnings", ())),
            extraction_provenance_id=item.get("extraction_provenance_id"),
        )
        for item in payload.get("blocks", ())
    )
    pictures = tuple(
        ParsedPicture(
            id=item["id"],
            page_number=int(item["page_number"]),
            bbox=BoundingBox(**item["bbox"]),
            child_block_ids=tuple(item.get("child_block_ids", ())),
        )
        for item in payload.get("pictures", ())
    )
    return ParsedPage(
        page_number=int(payload["page_number"]),
        width=float(payload["width"]),
        height=float(payload["height"]),
        kind=PageKind(payload["kind"]),
        status=ParseStatus(payload["status"]),
        blocks=blocks,
        pictures=pictures,
        warnings=tuple(payload.get("warnings", ())),
        text_character_count=int(payload.get("text_character_count", 0)),
        image_count=int(payload.get("image_count", 0)),
        image_coverage_ratio=float(payload.get("image_coverage_ratio", 0.0)),
        extraction_provenance_id=payload.get("extraction_provenance_id"),
    )


def _provenance_from_dict(payload: dict[str, Any]) -> ExtractionProvenance:
    return ExtractionProvenance(
        id=payload["id"],
        parser_name=payload["parser_name"],
        parser_version=payload["parser_version"],
        extraction_method=ExtractionMethod(payload["extraction_method"]),
        configuration_profile=payload["configuration_profile"],
        ocr_engine_name=payload.get("ocr_engine_name"),
        ocr_engine_version=payload.get("ocr_engine_version"),
    )
