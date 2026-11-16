"""Promote approved relevance decisions into a complete evaluation dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_learning_rag.retrieval_annotation import build_approved_evaluation_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    preview = json.loads(args.preview.read_text(encoding="utf-8"))
    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    dataset = build_approved_evaluation_dataset(preview=preview, decisions=decisions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dataset.to_payload(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"approved dataset={dataset.dataset_id}@{dataset.dataset_version} -> {args.output}")


if __name__ == "__main__":
    main()
