# Contributing to AI Code Review Agent

Thanks for your interest! This is a small, focused project — contributions
that keep it that way are welcome.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest
```

## Rules of the road

- **No secrets in code, ever.** Keys and tokens come from environment
  variables (see `.env.example`). If a test needs a token, use a fake
  value and mock the HTTP call.
- **The offline path must keep working.** Every new feature has to run
  with `CRA_LLM_PROVIDER=mock` and `CRA_EMBEDDING_PROVIDER=fake`
  (no API key, no downloads). Real-model providers are opt-in.
- **Findings must cite real lines.** A finding's `line`/`end_line`
  must exist in the reviewed hunk. The reviewer coerces and drops
  anything else — do not weaken that check.
- **Tests for new rules.** A new rule in `src/cra/rules.py` needs a
  case in `eval_data/eval_set.json` (or a unit test) that shows it
  firing — and a clean-code case proving it does not fire on good code.
- **Keep the pipeline explicit.** New review stages go in as LangGraph
  nodes in `src/cra/pipeline.py`, not as hidden calls inside other
  nodes.
- **Style:** type hints everywhere, docstrings on public functions,
  line length 100 (ruff: `E, F, I, W`).

## Running the checks

```bash
pytest                    # unit + API + eval tests
python scripts/run_eval.py  # eval harness (writes eval_data/eval_results.json)
```
