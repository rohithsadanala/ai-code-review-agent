#!/usr/bin/env python3
"""Run the eval harness: ``python scripts/run_eval.py``.

Prints per-case precision / recall / F1 / false-positive rate /
citation accuracy and writes the full results to
``eval_data/eval_results.json``. Uses the configured LLM provider
(default: mock — fully offline).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

from cra.config import Settings
from cra.evaluation import run_eval, save_results

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main() -> int:
    load_dotenv(os.path.join(ROOT, ".env"))
    settings = Settings()
    results = run_eval(
        os.path.join(ROOT, "eval_data", "eval_set.json"), settings=settings
    )
    print(f"Eval results ({results['cases']} cases)")
    print(f"  mean precision:            {results['mean_precision']}")
    print(f"  mean recall:               {results['mean_recall']}")
    print(f"  mean F1:                   {results['mean_f1']}")
    print(f"  mean false-positive rate:  {results['mean_false_positive_rate']}")
    print(f"  mean citation accuracy:    {results['mean_citation_accuracy']}")
    for case in results["per_case"]:
        print(
            f"\n  {case['name']}: expected={case['expected']} "
            f"reported={case['reported']} matched={case['matched']} "
            f"P={case['precision']} R={case['recall']} F1={case['f1']}"
        )
        for missed in case["missed"]:
            print(
                f"    MISSED: {missed['file']}:{missed['line']} "
                f"[{missed['category']}/{missed['severity']}]"
            )
        for extra in case["extra"]:
            print(
                f"    EXTRA:  {extra['file']}:{extra['line']} "
                f"[{extra['category']}/{extra['severity']}] {extra['title']}"
            )
    out = os.path.join(ROOT, "eval_data", "eval_results.json")
    save_results(results, out)
    print(f"\nResults written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
