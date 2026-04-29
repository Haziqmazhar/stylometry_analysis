# IJC319 Stylometry Project: Human vs LLM Academic Abstracts

This repository contains the code, data structure, and report outputs for an IJC319 project on stylometric differences between human-written academic abstracts and LLM-generated rewrites of the same source abstracts.

The project uses a config-driven pipeline to generate matched LLM rewrites, extract stylometric feature families, evaluate binary and multiclass attribution tasks, and produce semantic-overlap summaries and figures for the report.

## Project Aim

The main research question is whether feature-based stylometry can distinguish human-written academic abstracts from LLM-generated rewrites when both are based on the same source papers.

The current analysed corpus is a balanced pilot dataset with 1,000 texts:

- 200 human abstracts
- 200 Llama 3.1 rewrites
- 200 Gemma 2 rewrites
- 200 Qwen 2.5 rewrites
- 200 Mistral 7B rewrites

The source-matched design uses `source_id` so that each human abstract and its generated rewrites can be kept together during grouped cross-validation. This reduces source leakage and makes the results more defensible than a naive random split.

## Main Files

- `config.yaml`: controls generation and analysis settings
- `run_generation.py`: generates LLM rewrites from the human abstract CSV
- `run_analysis.py`: builds the feature table, runs classification, and writes result summaries
- `plot_result.py`: creates report figures from saved result files
- `src/stylometry.py`: base stylometric feature extraction helpers
- `src/feature_engine_v2.py`: feature-family construction for the v2 pipeline
- `prompts/factual_prompt.txt`: factual rewrite prompt used for generation
- `reports/IJC319_report_second_draft.tex`: current LaTeX report draft
- `reports/results/v2/`: main analysis outputs used by the report

## Repository Structure

```text
data/
  interim/
  processed/
  schema.md
prompts/
  factual_prompt.txt
reports/
  results/
    factual/
    v2/
src/
  __init__.py
  feature_engine_v2.py
  stylometry.py
config.yaml
plot_result.py
requirements.txt
run_analysis.py
run_generation.py
```

Generated LaTeX files such as `.aux`, `.log`, `.lof`, `.lot`, `.out`, `.toc`, and `.synctex.gz` are compilation artefacts rather than core project files.

## Configured Models

The current configuration uses four open-weight instruction-tuned model families through the `transformers` backend:

- `meta-llama/Llama-3.1-8B-Instruct`
- `google/gemma-2-9b-it`
- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

Shared generation settings are currently:

- temperature: `0.4`
- top-p: `0.9`
- max new tokens: `450`
- prompt type: `factual`
- sample size: `200` source abstracts

## Feature Families

The analysis compares six feature families:

- `lexical`: word count, vocabulary size, average word length, type-token ratio, hapax ratio
- `syntactic`: sentence length, sentence-length variation, punctuation rates, function-word rates, hedge and citation markers
- `combined`: lexical and syntactic features together
- `writeprints`: lightweight Writeprints-style proxy features, including character and punctuation ratios plus character trigram frequencies
- `stylometrix`: lightweight Stylometrix-style proxy features, including long-word, modal, and pronoun rates
- `stanza`: lightweight Stanza-style proxy features based on selected surface suffix distributions

The `writeprints`, `stylometrix`, and `stanza` sets are proxy feature families. They are not full reproductions of the original external toolkits and should be interpreted as engineered approximations for this pilot study.

## Evaluation Design

The main analysis uses:

- grouped 5-fold cross-validation with `source_id`
- Logistic Regression with standardisation
- Random Forest with class balancing
- binary human-vs-model classification for each LLM family
- five-class multiclass attribution across `human`, Llama, Gemma, Qwen, and Mistral
- semantic-overlap checks between human abstracts and generated rewrites

Binary metrics include accuracy, precision, recall, F1, ROC-AUC, Matthews correlation coefficient, and macro-F1. Multiclass outputs use the same grouped validation logic and report macro-averaged metrics.

Semantic overlap is evaluated using ROUGE-style scores and a simplified in-project METEOR-style score. These metrics are used as supporting validity checks only; they do not prove full semantic equivalence.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The Python dependencies are listed in `requirements.txt`. Generation with `transformers` models may also require model access, sufficient storage, and GPU/HPC resources depending on the local environment.

## Configure The Experiment

Edit `config.yaml` to control:

- input and output paths
- prompt file and prompt type
- source and text columns
- sample size
- enabled model entries
- generation settings
- cross-validation folds
- random seed
- feature sets
- binary and multiclass analysis toggles

Current default paths:

- human input: `data/processed/human_abstracts_min80.csv`
- LLM input/output: `data/processed/llm_abstracts_v2.csv`
- generation JSONL log: `data/interim/llm_abstracts_v2.jsonl`
- analysis output: `reports/results/v2`

## Run Generation

```powershell
python run_generation.py --config config.yaml
```

This writes:

- `data/processed/llm_abstracts_v2.csv`
- `data/interim/llm_abstracts_v2.jsonl`
- per-model CSV and JSONL files under `data/processed/per_model/` and `data/interim/per_model/`

Generation supports resume behaviour, so existing outputs can be reused instead of regenerated from scratch.

## Run Analysis

```powershell
python run_analysis.py --config config.yaml
```

This writes the main result files to `reports/results/v2/`, including:

- `master_with_features.csv`
- `summary_binary.csv`
- `summary_multiclass.csv`
- `summary_semantic.csv`
- `semantic_pair_scores.csv`
- binary fold files for each model, feature family, and estimator
- multiclass fold files for each feature family and estimator
- `run_manifest.json`

## Generate Figures

```powershell
python plot_result.py --results-dir reports/results/v2
```

This writes report-ready figures to `reports/results/v2/figures/`, including class balance, binary F1 summaries, multiclass summaries, fold distributions, and model-specific diagnostic plots.

## Dataset Requirements

The human input CSV must contain:

- `source_id`
- `abstract`

Recommended human metadata columns:

- `title`
- `year`
- `doi`

The LLM abstract CSV must contain:

- `source_id`
- `model_name`
- `prompt_type`
- `generated_abstract`

Recommended generation metadata columns:

- `temperature`
- `top_p`

See `data/schema.md` for the broader data schema.

## Current Report Outputs

The current report uses the v2 result set. The key interpretation is that feature-based stylometry shows measurable separation between human and LLM-generated academic abstracts in this controlled, source-matched setting. Combined and syntactic feature sets are generally strongest, while lexical-only and weaker proxy feature sets are less reliable.

The results should be interpreted cautiously. The project is a pilot study limited by domain, prompt condition, sample size, and proxy feature-family implementations. The outputs support stylistic separability in this dataset, not universal LLM detection or forensic-level certainty.

## Notes

- Keep `source_id` grouped during validation to avoid source leakage.
- Do not interpret proxy feature families as full toolkit reproductions.
- Do not treat ROUGE or simplified METEOR as proof of deep semantic equivalence.
- The LaTeX report and generated figures are under `reports/`.
- The main reproducibility record is `reports/results/v2/run_manifest.json`.
