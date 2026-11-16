"""Turn a real retrieval preview into a human relevance-review worksheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_learning_rag.retrieval_annotation import build_annotation_review_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.preview.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        build_annotation_review_markdown(payload),
        encoding="utf-8",
    )
    print(f"annotation review -> {args.output}")


if __name__ == "__main__":
    main()
