"""
Enterprise RAG Evaluation & Benchmarking Script.

Validates the golden_dataset.json for correctness, detects placeholder
documents, and checks that expected source files exist in the corpus.

When a real retrieval backend is wired up, replace _mock_retrieval() in
run_eval.py to produce actual metrics.  This script serves as the
validation and report layer.

Usage:
  python tests/evaluation/evaluate.py
  python tests/evaluation/evaluate.py --dataset ./tests/evaluation/golden_dataset.json
  python tests/evaluation/evaluate.py --raw-dir ./data/raw
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    """Load and parse the golden dataset, skipping header/comment entries."""
    if not dataset_path.exists():
        print(f"[ERROR] Dataset file not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    # Skip entries that look like metadata/comment blocks (no 'question' key)
    entries = [e for e in raw if isinstance(e, dict) and "question" in e]
    return entries


def _validate_entries(
    entries: list[dict[str, Any]],
    raw_dir: Path,
) -> dict[str, Any]:
    """
    Validate all entries in the golden dataset.

    Checks:
    - Required fields present.
    - expected_document is not a PLACEHOLDER_.
    - expected_document file exists in raw_dir (if configured).

    Returns a validation report dict.
    """
    required_fields = {"question", "expected_document", "expected_evidence"}

    errors: list[dict[str, Any]] = []
    placeholder_count = 0
    missing_doc_count = 0
    field_error_count = 0

    for i, entry in enumerate(entries):
        test_id = entry.get("test_id", f"entry-{i + 1}")

        # 1. Required fields
        missing = required_fields - set(entry.keys())
        if missing:
            errors.append({
                "test_id": test_id,
                "type": "MISSING_FIELD",
                "detail": f"Required fields missing: {sorted(missing)}",
                "blocking": True,
            })
            field_error_count += 1
            continue

        doc = entry["expected_document"]

        # 2. Placeholder check
        if str(doc).startswith("PLACEHOLDER_"):
            errors.append({
                "test_id": test_id,
                "type": "PLACEHOLDER_DOCUMENT",
                "detail": (
                    f"expected_document={doc!r} is a placeholder.  "
                    "Upload the real document and update this field to enable evaluation."
                ),
                "blocking": True,
            })
            placeholder_count += 1
            continue

        # 3. Document existence check
        if raw_dir and not (raw_dir / doc).exists():
            errors.append({
                "test_id": test_id,
                "type": "MISSING_SOURCE_DOCUMENT",
                "detail": (
                    f"expected_document={doc!r} does not exist in {raw_dir}.  "
                    "Ingest the document before running evaluation."
                ),
                "blocking": True,
            })
            missing_doc_count += 1

    return {
        "total_entries": len(entries),
        "valid_entries": len(entries) - len([e for e in errors if e["blocking"]]),
        "placeholder_count": placeholder_count,
        "missing_doc_count": missing_doc_count,
        "field_error_count": field_error_count,
        "errors": errors,
        "can_evaluate": placeholder_count == 0 and missing_doc_count == 0 and field_error_count == 0,
    }


def evaluate_dataset(
    dataset_path: str = "./tests/evaluation/golden_dataset.json",
    raw_dir: str = "./data/raw",
) -> dict[str, Any]:
    """
    Validate the golden dataset and report blockers.

    Returns a machine-readable report dict.  Exit code is non-zero if
    evaluation is blocked by placeholders or missing documents.
    """
    path = Path(dataset_path)
    raw = Path(raw_dir)

    entries = _load_dataset(path)
    validation = _validate_entries(entries, raw)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(path.resolve()),
        "raw_dir": str(raw.resolve()),
        "validation": validation,
        "status": "READY" if validation["can_evaluate"] else "BLOCKED",
        "metrics": None,  # populated by run_eval.py when backend is wired
    }

    print(f"\n--- Enterprise-RAG Evaluation Dataset Validation ---")
    print(f"Dataset     : {path}")
    print(f"Total entries: {validation['total_entries']}")
    print(f"Valid entries: {validation['valid_entries']}")
    print(f"Placeholders : {validation['placeholder_count']}")
    print(f"Missing docs : {validation['missing_doc_count']}")
    print(f"Field errors : {validation['field_error_count']}")
    print(f"Status       : {report['status']}")

    if validation["errors"]:
        print("\n[Validation Errors]")
        for err in validation["errors"][:10]:
            print(f"  [{err['type']}] {err['test_id']}: {err['detail']}")
        if len(validation["errors"]) > 10:
            print(f"  ... and {len(validation['errors']) - 10} more errors.")

    if validation["can_evaluate"]:
        print("\n[OK] Dataset is valid.  Connect run_eval.py to the live backend to produce metrics.")
    else:
        print(
            "\n[BLOCKED] Evaluation requires real corpus documents.  "
            "See errors above and the golden_dataset.json _comment/_note fields for instructions."
        )

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enterprise-RAG Evaluation Dataset Validator")
    parser.add_argument(
        "--dataset",
        default="./tests/evaluation/golden_dataset.json",
        help="Path to the golden dataset JSON",
    )
    parser.add_argument(
        "--raw-dir",
        default="./data/raw",
        help="Path to the raw documents directory",
    )
    parser.add_argument(
        "--report",
        help="Optional path to save the JSON report",
    )
    args = parser.parse_args()

    report = evaluate_dataset(dataset_path=args.dataset, raw_dir=args.raw_dir)

    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport saved to: {args.report}")

    sys.exit(0 if report["validation"]["can_evaluate"] else 1)
