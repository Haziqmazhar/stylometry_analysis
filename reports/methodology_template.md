# Methodology and Results Template

## 1. Research Objective
- Primary objective: Compare stylometric differences between human and LLM-generated abstracts in machine learning literature.
- Main experiment: Human vs LLM factual prompt outputs.

## 2. Data
- Human abstracts: `N=____` (Scopus, English only, deduplicated)
- LLM abstracts: `N=____` (Llama 3.1, factual prompt, matched by `source_id`)
- Unit of analysis: abstract text

## 3. Experimental Controls
- Prompt type analyzed: factual only
- Target length: 220-300 words
- Split strategy: grouped split by `source_id` (no source leakage)
- Class balance: 1:1

## 4. Stylometric Features
- Lexical richness: type-token ratio, hapax ratio, average word length
- Sentence structure: average/variance of sentence length
- Function-word profile (normalized rates)
- Punctuation profile (per 100 words)
- Hedging and citation marker rates

## 5. Models
- Logistic Regression (interpretable baseline)
- Random Forest (non-linear baseline)

## 6. Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Insert from: `reports/results/factual/metrics.json`

## 7. Explainability and Forensic Analysis
- Global importance:
  - Model coefficients / feature_importances
  - Permutation importance
  - SHAP mean absolute values
- Local evidence:
  - SHAP per-sample values

Files:
- `reports/results/factual/feature_importance_global.csv`
- `reports/results/factual/feature_importance_permutation.csv`
- `reports/results/factual/shap_global_importance.csv`
- `reports/results/factual/shap_local_values.csv`

## 8. Findings (Write-up)
- Which features best separate human and LLM writing?
- Are distinctions stable and linguistically plausible?
- Potential confounds and limitations.

## 9. Limitations
- Single domain (machine learning abstracts)
- Single language (English)
- Single model family (Llama 3.1) in first phase

## 10. Next Phase
- Add second condition: opinion prompt
- Compare factual vs opinion stylometric drift
- Add additional open-source models for robustness
