#!/usr/bin/env python3
"""
Evaluation-only runner (no training).

This is useful to:
  - Re-evaluate an existing run under a different thresholding policy (e.g., fpr_target)
  - Generate temporal metrics (TTD/coverage) without retraining

Example:
  ./.venv/bin/python scripts/evaluate_only.py --config configs/final/eval_operational_final_warmup_r30.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.evaluation.evaluator import Evaluator as NewEvaluator


def main() -> int:
    parser = argparse.ArgumentParser(description="Run evaluation only (no training).")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Minimal guardrails: avoid accidental reprocessing in eval-only runs
    config.setdefault("force_reprocess_data", False)

    print(f"Evaluation-only: {config.get('simulation_name', 'N/A')}")
    evaluator = NewEvaluator(config)
    evaluator.evaluate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

