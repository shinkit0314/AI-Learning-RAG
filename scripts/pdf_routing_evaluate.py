"""Compare a candidate PDF routing configuration across the local sample manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from ai_learning_rag.pdf_routing import PdfRoutingConfig, analyze_pdf_pages, build_route_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pdf-routing-v1.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/PDF/sample_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/pdf-routing-evaluation.json"),
    )
    args = parser.parse_args()
    config = PdfRoutingConfig.from_json(args.config)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    samples = []
    route_counts: Counter[str] = Counter()
    for sample in manifest["samples"]:
        pdf_path = Path(sample["path"])
        if not pdf_path.is_file():
            samples.append({"sample_id": sample["sample_id"], "status": "missing"})
            continue
        signals = analyze_pdf_pages(pdf_path)
        decisions = build_route_plan(signals, config)
        route_counts.update(decision.selected_route.value for decision in decisions)
        samples.append(
            {
                "sample_id": sample["sample_id"],
                "path": sample["path"],
                "expected_type": sample["expected_type"],
                "pages": [
                    {"signals": asdict(page_signals), "decision": asdict(decision)}
                    for page_signals, decision in zip(signals, decisions, strict=True)
                ],
            }
        )
    payload = {
        "routing_profile": config.routing_profile,
        "routing_config": config.profile_payload,
        "route_counts": dict(sorted(route_counts.items())),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"samples={len(samples)} routes={dict(route_counts)} -> {args.output}")


if __name__ == "__main__":
    main()
