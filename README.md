# Anchoring Bias in LLM-as-a-Judge Systems: Prior Scores Compromise Evaluation Independence

Code, public data, and aggregate results for "Anchoring Bias in LLM-as-a-Judge Systems: Prior Scores Compromise Evaluation Independence," accepted at CIKM 2026.

Paper: https://doi.org/10.1145/3799682.3840879

## What is included

- the 20 public text tasks used in the paper
- sanitized trial-level data for the 192,000-call main experiment
- sanitized token-probability data for the Llama mechanism probes
- sanitized trial-level data for the numerical mitigation experiment
- scripts for the task-aware analyses and paper figures
- text-only experiment runners for OpenAI-compatible APIs and local Hugging Face models
- aggregate industry results and the code used for the industry experiments

The main data omit prompts and raw model responses because the task definitions already contain the public inputs and those fields are not needed for analysis.

The industrial dataset cannot be disclosed. It contains proprietary campaign descriptions and messages. This repository therefore includes only aggregate industry results. `run_industry.py` and `scripts/industry.py` document the required columns and can be used by readers with their own compatible dataset.

## Setup

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync
```

Local Hugging Face inference has large optional dependencies:

```bash
uv sync --group local
```

For an OpenAI-compatible endpoint, copy `.env.example` to `.env` and set:

```text
LITELLM_ENDPOINT=https://your-endpoint
LITELLM_API_KEY=your-key
```

The model names in `src/llm_judge_bias/config.py` match the paper's environment. Most readers will need to replace API model names or adjust the provider client. Local Hugging Face model names work with Transformers, subject to hardware and model-access requirements.

## Reproduce the analyses

The committed data are enough to regenerate all public statistical outputs and figures without calling a model:

```bash
uv run python scripts/reproduce.py
```

Outputs are written to `results/` and `figures/`. The command uses 10,000 bootstrap replicates and may take a few minutes.

Run the tests and linter with:

```bash
uv run pytest
uv run ruff check .
```

## Run experiments

A small API run:

```bash
uv run python run_experiment.py \
  --models infobip-gpt-4-1 \
  --tasks sum_001_relativity \
  --trials 1
```

The full main experiment uses 20 tasks, eight models, three conditions, four temperatures, and 100 trials per cell. It is expensive. Review the configuration before running it.

Targeted experiment runners:

```bash
uv run python run_mechanism.py --model infobip-gpt-4-1
uv run python run_mitigation.py --model infobip-gpt-4-1
uv run python run_industry.py --input private/campaign_drift.csv --output private/industry.parquet
uv run python scripts/industry.py --input private/industry.parquet --output-dir results
```

`run_mechanism.py` requests token log probabilities, which not every provider supports. The industry runner expects `Id`, `CampaignId`, `CampaignDescription`, `SampleMessage`, `LLMClassification`, and `ReviewerClassification` columns in a semicolon-delimited CSV. Keep its input and row-level output under the ignored `private/` directory.

## Data files

- `data/main_experiment.parquet`: main numerical experiment, 192,000 attempted evaluations
- `data/mechanism_llama.parquet`: selected Llama token-probability probes
- `data/mechanism_gpt41_summary.json`: recovered aggregate GPT-4.1 mechanism values
- `data/mitigation_trials.parquet`: sanitized GPT-4.1 mitigation calls
Industry aggregates are under `results/industry_group_a_aggregates.csv` and `results/industry_group_b_*.csv`.

## Citation

> Ante Kapetanovic, Kemal Altwlkany, Andro Mercep, Tomislav Duricic, and Emanuel Lacic. 2026. Anchoring Bias in LLM-as-a-Judge Systems: Prior Scores Compromise Evaluation Independence. In Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM '26), November 7-11, 2026, Rome, Italy. ACM, New York, NY, USA. https://doi.org/10.1145/3799682.3840879

```bibtex
@inproceedings{kapetanovic2026anchoring,
  author = {Kapetanovic, Ante and Altwlkany, Kemal and Mercep, Andro and Duricic, Tomislav and Lacic, Emanuel},
  title = {Anchoring Bias in LLM-as-a-Judge Systems: Prior Scores Compromise Evaluation Independence},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
  year = {2026},
  publisher = {Association for Computing Machinery},
  doi = {10.1145/3799682.3840879}
}
```

## License

MIT
