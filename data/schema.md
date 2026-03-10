# Dataset Schema

## `human_abstracts.csv`
Required columns:
- `source_id`: unique stable id for the source paper
- `title`: paper title
- `abstract`: human abstract text
- `year`: publication year (optional but recommended)
- `doi`: DOI (optional)

## `llm_abstracts.csv`
Required columns:
- `source_id`: must match `human_abstracts.source_id`
- `model_name`: e.g., `llama-3.1-8b-instruct`
- `prompt_type`: `factual` or `opinion`
- `generated_abstract`: LLM abstract text
- `temperature`: generation temperature (recommended)
- `top_p`: generation top-p (recommended)

## Modeling Input
The scripts derive a combined feature file with:
- `source_id`
- `label`: `human` or `llm`
- `prompt_type`
- `text`
- numeric stylometric features

## Quality Rules
- English only
- Non-empty abstracts
- Remove duplicates (exact-text duplicates)
- Keep LLM output near target length (220-300 words recommended)
