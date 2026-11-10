"""Command-line interface for the AI Code Review Agent.

Usage
-----
    cra review <diff-file> [--repo PATH] [--format markdown|json]
    cra review --pr owner/repo#number [--repo PATH]
    cra serve [--host 127.0.0.1] [--port 8000]
    cra eval [--eval-set eval_data/eval_set.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from .config import Settings
from .evaluation import run_eval, save_results
from .github_client import fetch_pr_diff, parse_pr_ref
from .pipeline import run_review


def _cmd_review(args: argparse.Namespace, settings: Settings) -> int:
    if args.pr:
        owner, repo, number = parse_pr_ref(args.pr)
        if not settings.github_token:
            print("error: GITHUB_TOKEN is not set — cannot fetch a PR diff.",
                  file=sys.stderr)
            return 2
        print(f"Fetching {args.pr} from GitHub ...", file=sys.stderr)
        diff_text = fetch_pr_diff(owner, repo, number, settings.github_token)
        pr = args.pr
    else:
        diff_text = Path(args.diff_file).read_text(encoding="utf-8")
        pr = None

    report = run_review(
        diff_text, settings=settings, repo_path=args.repo or "", pr=pr
    )
    if args.format == "json":
        print(report.model_dump_json(indent=2))
    else:
        print(report.markdown)
    return 0


def _cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    print(
        f"AI Code Review Agent — UI at "
        f"http://{args.host}:{args.port} (docs at /docs)",
        file=sys.stderr,
    )
    uvicorn.run(
        "cra.api:app",
        host=args.host,
        port=args.port,
        reload=settings.reload,
    )
    return 0


def _cmd_eval(args: argparse.Namespace, settings: Settings) -> int:
    results = run_eval(args.eval_set, settings=settings)
    print(f"\nEval results ({results['cases']} cases)")
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
            print(f"    MISSED: {missed['file']}:{missed['line']} "
                  f"[{missed['category']}/{missed['severity']}]")
        for extra in case["extra"]:
            print(f"    EXTRA:  {extra['file']}:{extra['line']} "
                  f"[{extra['category']}/{extra['severity']}] {extra['title']}")
    save_results(results, args.out)
    print(f"\nResults written to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cra",
        description="AI Code Review Agent — review diffs and PRs with an LLM agent.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_review = sub.add_parser("review", help="Review a diff file or a GitHub PR.")
    p_review.add_argument("diff_file", nargs="?", help="Path to a unified diff file.")
    p_review.add_argument("--pr", help="Pull request ref: owner/repo#number.")
    p_review.add_argument("--repo", default="", help="Local repo path for RAG context.")
    p_review.add_argument(
        "--format", choices=["markdown", "json"], default="markdown",
        help="Output format (default: markdown).",
    )
    p_review.set_defaults(func=_cmd_review)

    p_serve = sub.add_parser("serve", help="Start the API server + web UI.")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=_cmd_serve)

    p_eval = sub.add_parser("eval", help="Run the eval harness.")
    p_eval.add_argument(
        "--eval-set", default="eval_data/eval_set.json",
        help="Path to the eval-set JSON file.",
    )
    p_eval.add_argument(
        "--out", default="eval_data/eval_results.json",
        help="Where to write the results JSON.",
    )
    p_eval.set_defaults(func=_cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    settings = Settings()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "review" and not args.diff_file and not args.pr:
        parser.error("review needs a diff file or --pr owner/repo#number")
    return args.func(args, settings)


if __name__ == "__main__":
    raise SystemExit(main())
