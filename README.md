# Stylometry Study: Human vs LLM Abstracts

This project provides a reproducible pipeline for stylometric analysis of:
- Human-written academic abstracts (Scopus export)
- LLM-generated abstracts (Llama 3.1), matched by source paper

The default main experiment is:
- `human` vs `llm_factual` (balanced, grouped split by `source_id`)

The opinion prompt condition can be added later and analyzed separately.

## Research Design (Implemented)
- Domain: Computer Science / Machine Learning abstracts
- Language: English only
- Unit: Abstract text (target 220-300 words for LLM generations)
- Split: Group-aware (`source_id`) to prevent leakage
- Priorities: interpretability, accuracy, forensic explainability

## Project Structure
```text
data/
  raw/
  interim/
  processed/
  schema.md
prompts/
  factual_prompt.txt
reports/
  methodology_template.md
  results/
scripts/
  01_prepare_human_dataset.py
  02_build_generation_prompts.py
  03_extract_features.py
  04_train_evaluate.py
  05_explain_shap.py
src/
  stylometry.py
requirements.txt
```

## Setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Step 1: Prepare Human Dataset from Scopus CSV
```powershell
python scripts/01_prepare_human_dataset.py `
  --input "C:\Users\Amir\Downloads\scopus_export_Feb 16-2026_482bfa38-ccb8-499b-ad82-0da1381dcbb5.csv" `
  --output data/processed/human_abstracts.csv `
  --limit 1000
```

## Step 2: Build Llama 3.1 Generation Prompts (Factual)
```powershell
python scripts/02_build_generation_prompts.py `
  --human data/processed/human_abstracts.csv `
  --output data/interim/factual_prompts.jsonl
```

## Step 2.5: Generate with Ollama (Local Llama 3.1)
Install Ollama and pull model once:
```powershell
ollama pull llama3.1:8b
```

Run generation (writes JSONL outputs):
```powershell
python scripts/02c_generate_with_ollama.py `
  --input-jsonl data/interim/factual_prompts.jsonl `
  --output-jsonl data/interim/factual_outputs.jsonl `
  --model llama3.1:8b `
  --resume
```

Then convert to analysis CSV format:
```powershell
python scripts/02b_build_llm_dataset.py `
  --input-jsonl data/interim/factual_outputs.jsonl `
  --output-csv data/processed/llm_abstracts.csv `
  --model-name llama-3.1 `
  --prompt-type factual
```

See `data/schema.md` for the required columns.

## Step 3: Extract Stylometric Features
```powershell
python scripts/03_extract_features.py `
  --human data/processed/human_abstracts.csv `
  --llm data/processed/llm_abstracts.csv `
  --prompt-type factual `
  --output data/processed/features_factual.csv
```

## Step 4: Train + Evaluate
```powershell
python scripts/04_train_evaluate.py `
  --features data/processed/features_factual.csv `
  --results-dir reports/results/factual
```

## Step 5: Explainability (SHAP)
```powershell
python scripts/05_explain_shap.py `
  --features data/processed/features_factual.csv `
  --results-dir reports/results/factual
```

## Refined Factual Prompt
Stored in `prompts/factual_prompt.txt`.

## Notes
- Keep factual and opinion prompt experiments separate in reporting.
- Do not mix `llm_factual` and `llm_opinion` into one LLM class for the main stylometry claim.
