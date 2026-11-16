"""Score a real PDF retrieval preview against an approved evaluation dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_learning_rag.retrieval_evaluation import (
    evaluate_retrieval,
    load_retrieval_evaluation_dataset,
)
from ai_learning_rag.retrieval_evaluation_preview import evaluation_inputs_from_preview


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset = load_retrieval_evaluation_dataset(args.dataset)
    preview = json.loads(args.preview.read_text(encoding="utf-8"))
    inventory, manifest, rankings = evaluation_inputs_from_preview(preview)
    report = evaluate_retrieval(
        dataset=dataset,
        inventory=inventory,
        results_by_query=rankings,
        index_manifest=manifest,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "retrieval-evaluation.json").write_text(
        report.to_json(),
        encoding="utf-8",
    )
    (args.output_dir / "retrieval-evaluation.md").write_text(
        report.to_markdown(),
        encoding="utf-8",
    )
    print(
        f"dataset={dataset.dataset_id}@{dataset.dataset_version} "
        f"queries={len(dataset.cases)} -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
