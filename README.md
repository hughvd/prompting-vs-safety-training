# Can We Prompt Our Way to Safety?

This repository contains the code and experimental setup for evaluating the effects of different system prompt types on LLM safety behavior and capability. This work was conducted as an experiment for Week 3 (Model Specifications and Compliance) of Boaz Barak's "CS 2881r: AI Alignment and Safety" at Harvard.

## TL;DR

We evaluate the effects of different system prompt types (minimal vs. principles vs. rules) and their combinations on over-refusal, toxic-refusal, and capability benchmarks using DeepSeek-R1-Qwen3-8b and RealSafe-R1-8b (an additionally safety-tuned DeepSeek-R1).

**Key Finding:** Appropriate system prompts can achieve toxic-refusal rates within 5% of the safety-tuned model while reducing over-refusal by 2-10x, suggesting prompting may complement or even outperform certain safety training approaches on these benchmarks.

## Research Question

**Can we prompt our way to safety?** More specifically:
- If we take a model with minimal safety training and give it detailed safety instructions via system prompts, does it behave comparably to a model that's been explicitly fine-tuned for safety?
- Do different system prompt styles (minimal, principles-based, rules-based) affect safety behavior differently?
- Can prompting serve as a substitute for safety training in achieving low over-refusal while maintaining appropriate toxic-refusal rates?

## Experimental Design

### Models Tested

1. **DeepSeek-R1-Qwen3-8B**: A reasoning model trained with reinforcement learning, but without extensive safety-specific post-training (base model)
2. **RealSafe-R1-8B**: The same DeepSeek-R1 model with additional safety training applied (Zhang et al. 2025)
3. **Bonus Models**: GPT-4o, Claude-3.5-Sonnet, Gemini-2.5-Flash

### System Prompt Variants

We designed three distinct styles of system prompts:

- **Minimal (M)**: A few sentences capturing "helpful, honest, harmless" with basic guidance
- **Principles (P)**: Eight high-level principles (e.g., "Helpfulness with purpose," "Harmlessness first")
- **Rules (R)**: Thirty lines of explicit operational rules organized by category

All possible combinations were tested: M, P, R, M+P, M+R, P+R, M+P+R, plus a baseline with no system prompt (8 configurations × 2 models = 16 conditions).

### Benchmarks

- **OR-Bench** (Cui et al. 2024): Measures over-refusal and appropriate refusal behavior
  - OR-Bench-80k: 150 prompts testing borderline cases (should be answered)
  - OR-Bench-hard: 50 difficult prompts requiring nuance (should be answered)
  - OR-Bench-toxic: 200 prompts that should be refused

- **MMLU-Pro** (Wang et al. 2024): 100 prompts sampled across topics to measure capability preservation

Each evaluation was run 3 times with resampling for variance estimation (500 total prompts per run).

## Key Results

1. The "minimal" system prompt noticeably increases over-refusal in DeepSeek-R1-Qwen3-8b compared to baseline and other prompt styles—likely due to underspecification causing the model to err toward "safe rather than sorry"

2. Any system prompt greatly increases toxic refusal in DeepSeek-R1-Qwen3-8b compared to the baseline (no prompt)

3. System prompts achieve toxic-refusal rates within ~5% of RealSafe-R1-8b while providing:
   - ~2x less over-refusal on OR-bench-hard
   - ~5-10x less over-refusal on OR-bench-80k

4. System prompt effects vary greatly by model—the "minimal" effect observed in DeepSeek is not present in GPT-4o

5. Safety training in RealSafe resulted in large increases in over-refusal (98-100% refusal in OR-bench-hard)

## Repository Structure

```
.
├── configs/
│   ├── evaluation/          # Benchmark configs (refusal.yaml, capability.yaml)
│   └── experiments/         # Per-model experiment configs
├── specs/                   # System prompt variants (8 files)
│   ├── baseline.txt         # Empty (no system prompt)
│   ├── S0_minimal.txt
│   ├── S1_principles.txt
│   ├── S2_rules.txt
│   └── S3-S6_*.txt         # Combinations
├── src/
│   ├── evaluations/         # Refusal and capability evaluation logic
│   ├── scripts/             # Main experiment runner
│   ├── model_client.py      # OpenAI/OpenRouter API client
│   └── utils.py             # Config loading and utilities
├── scripts/evaluation/      # Bash wrappers for running experiments
├── data/                    # Experiment outputs (gitignored)
├── results/                 # Generated visualizations
└── analysis.ipynb           # Analysis and plotting notebook
```

## Quick Start

### Prerequisites
```bash
cp .env.example .env
# Add your OPENROUTER_API_KEY to .env
uv sync
```

### Running an Experiment

```bash
# Run DeepSeek-R1 evaluation
bash scripts/evaluation/run_deepseekR1.sh

# Run RealSafe-R1 evaluation
bash scripts/evaluation/run_realsafeR1.sh

# Run GPT-4o evaluation
bash scripts/evaluation/run_gpt-4o.sh
```

### Experiment Configuration

Each experiment is configured via YAML files in `configs/experiments/`. Key parameters:

```yaml
output:
  root: "data/experiments"
  experiment: "deepseekR1"

model:
  provider: "openai"
  base_url: "https://openrouter.ai/api/v1"
  name: "deepseek/deepseek-r1-0528-qwen3-8b"
  reasoning: true

generation:
  max_tokens: 1024
  temperature: 0.0

runs:
  count: 3              # Number of replicas
  base_seed: 42         # Random seed for reproducibility
```

### Analysis

Results are written to `data/experiments/{experiment_name}/` in JSONL format. Use [analysis.ipynb](analysis.ipynb) for aggregating results and generating visualizations.

## Key Findings & Implications

### Why Does This Work?

**Prompting** leverages the model's instruction-following capability to shift the conditional probability distribution—essentially asking "given that you're supposed to be helpful and harmless, what should the next tokens look like?"

**Safety Training** attempts to change the base distribution itself by modifying weights through reinforcement learning, making "good" behavior more probable at the distribution level.

The tension: Both face the challenge of selecting the right sub-distributions from what the model has learned, but they differ in **permanence** and **flexibility**:

- Prompting is reversible and adjustable per use case, but can be overridden (including by adversarial prompts)
- Safety training creates more durable changes but risks over-conservatism and is hard to target precisely

### Best Practice Suggestion

Combine both approaches:
1. Safety training to establish strong baseline guardrails (with over-conservatism in mind)
2. Carefully designed system prompts to fine-tune the balance between helpfulness and safety for specific use cases

## Limitations

- The difference in distillation targets between DeepSeek-R1-Qwen3-8b and DeepSeek-R1-Llama-3.1-8b may affect results
- n=3 experiments with high variance limits statistical claims
- Token limits during evaluation may have affected capability measurements
- Future work should test prompting specifically for over-refusal reduction (e.g., including positive statements or in-context examples)

## Citation

If you use this code or findings, please cite:
```
Experiment for CS 2881r: AI Alignment and Safety (Week 3)
Harvard University, 2025
```

## References

- Ahmed et al. (2025). "SpecEval: Evaluating Model Adherence to Behavior Specifications"
- Zhang et al. (2025). RealSafe-R1 safety training
- Cui et al. (2024). OR-Bench: Over-refusal benchmark
- Wang et al. (2024). MMLU-Pro
