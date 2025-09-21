# Hosted Inference Setup (OpenRouter)

Follow these steps to run the spec-adherence experiment using OpenRouter's hosted inference API—no GPU provisioning required.

## 1. Create an OpenRouter account
- Visit https://openrouter.ai, create an account, and add billing (prepaid credits or a card).
- Generate an API key (`sk-...`) from the **API Keys** page.

## 2. Save the API key in `.env`
Copy `.env.example` to `.env` and drop in your key:
```bash
cp .env.example .env
```
Edit `.env` so it contains:
```
OPENROUTER_API_KEY=sk-your-key
```
The runner automatically loads `.env` before executing, so nothing else is required.

## 3. (Optional) Customize request headers
OpenRouter appreciates `HTTP-Referer` and `X-Title` headers. Update both entries under `task_model.headers` and `judge_model.headers` in `configs/evaluation/spec_adherence.yaml` with a URL/email and short title tied to you or your project. The provided placeholder (`https://example.com/spec-compliance`) is a stub—replace it before long-running jobs.

## 4. Install dependencies
This project assumes `uv` for dependency management. From the repo root:
```bash
uv sync
```
This creates/updates `.venv` and installs packages from `pyproject.toml` (`httpx`, `pyyaml`, `openai`, `tqdm`, etc.).

## 5. (Optional) Dry run locally
Keep `run.dry_run: true` in `configs/evaluation/spec_adherence.yaml` to verify the loop without network calls:
```bash
uv run python src/scripts/run_evaluation.py
```
You should see log output ending with a dry-run results file under `data/spec_adherence_baseline/results/`.

## 6. Run against OpenRouter
Flip `run.dry_run` to `false` (or omit the key) in the config, then execute:
```bash
uv run bash scripts/evaluation/run_spec_eval.sh configs/evaluation/spec_adherence.yaml
```
The evaluator uses the official OpenAI Python client configured for OpenRouter and will query two models:
- Task model: `qwen/qwen-2.5-7b-instruct`
- Judge model: `qwen/qwen-2.5-3b-instruct`

Results are appended to `data/spec_adherence_baseline/results/spec_adherence.sample.jsonl` (adjust `output_filename` in the config as needed).

By default the runner shows a `tqdm` progress bar; add `show_progress: false` under the `run:` block in the config if you prefer plain logging.

## 7. Inspect outputs
Use any JSONL tooling to review judgments, e.g.:
```bash
head data/spec_adherence_baseline/results/spec_adherence.sample.jsonl
```
Each record includes the spec id, prompt metadata, raw response, and judge verdict.

## 8. Tuning options
- Swap model IDs under `task_model.name` or `judge_model.name` to test alternatives available on OpenRouter.
- Adjust generation params (temperature, max tokens) or judge instructions in the config.
- Use `http.timeout` at the top level of the config if you need to raise/lower the 60s default request timeout.

## 9. (Optional) Self-hosting
If you later need full control (custom weights, local GPUs), restore the previous `execution.md` instructions from version control or adapt the config back to point at your own endpoints. The rest of the pipeline remains unchanged.
