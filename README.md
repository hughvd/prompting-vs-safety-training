# Spec Compliance Evaluation

This repo hosts a lightweight pipeline for measuring how instruction-tuned language models respond to different "spec" (system prompt) variants.

## Experiment at a Glance
- **Goal:** Compare adherence, over-refusal, and under-refusal rates across specs ranging from no instructions to detailed/contradictory rule sets.
- **Models:** Task model runs the prompts; judge model scores the responses via JSON rubric (OpenRouter-hosted by default).
- **Dataset:** Small JSONL prompt set labelled by domain (casual, legal, STEM, creative) and safety class (benign, unsafe).

## Key Components
- `configs/evaluation/` – YAML configs describing datasets, specs, model endpoints, and runtime flags.
- `specs/` – System prompt variants (minimal, principles, rules, conflict, over-strict, baseline).
- `src/` – Python modules implementing config loading (`utils.py`), the evaluator (`evaluation.py`), and the CLI (`scripts/run_evaluation.py`).
- `scripts/evaluation/run_spec_eval.sh` – Entry point that calls the evaluator with `uv run`.
- `data/` – Auto-created, gitignored directory where results land (JSONL + room for figures/logs).
- `scratch/` – Notebook playground (e.g., `analysis.ipynb` for quick plots).

## Quick Start
1. `cp .env.example .env` and add your `OPENROUTER_API_KEY` plus real `HTTP-Referer`/`X-Title` values in the config.
2. `uv sync` to install dependencies (`openai`, `tqdm`, `pandas`, …).
3. (Optional) Dry run with `run.dry_run: true` to validate plumbing.
4. Set `run.dry_run: false` and execute:
   ```bash
   uv run bash scripts/evaluation/run_spec_eval.sh configs/evaluation/spec_adherence.yaml
   ```
5. Inspect `data/spec_adherence_baseline/results/*.jsonl` and use the notebook or a pandas script for metrics/visuals.

## Notes
- Change `task_model`/`judge_model` IDs in the config to try other OpenRouter or OpenAI-compatible models (e.g., GPT-4o as judge).
- Add new specs by dropping files in `specs/` and referencing them in the config list.
- Disable the `tqdm` progress bar by setting `run.show_progress: false` if needed.
