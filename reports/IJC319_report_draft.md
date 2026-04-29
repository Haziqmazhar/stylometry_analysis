# Stylometric Analysis of Human and Large Language Model Academic Abstracts: A Config-Driven Pilot Study

## Structured Abstract

This project investigates whether stylometric analysis can distinguish human-written academic abstracts from large language model (LLM) generated rewrites in a reproducible machine learning pipeline. The study is motivated by two linked problems: the rapid adoption of generative AI in academic writing workflows, and the need for technically defensible methods that examine writing style rather than topic content alone. The primary aim is to evaluate whether feature-based stylometry can separate human abstracts from LLM outputs and, in principle, support broader authorship attribution across multiple model families.

The data pipeline uses English-language academic abstracts collected from Scopus-derived records and matched LLM-generated rewrites built from the same source items. The implemented `v2` pipeline is config-driven and supports multiple open-weight models, grouped cross-validation by `source_id`, and multiple stylometric feature families. The analysed `master_with_features.csv` file contains 1,000 texts: 200 human abstracts and 200 generated abstracts from each of four models (`Llama 3.1`, `Gemma 2`, `Qwen 2.5`, and `Mistral 7B`). However, the currently verified fold-level evaluation outputs in the saved `v2` results directory cover the binary `human` versus `Gemma 2` task.

The strongest verified results were obtained using combined and syntactic feature sets. Random Forest with combined features achieved mean accuracy of `0.8400`, F1 of `0.8385`, and ROC-AUC of `0.9249`. Lexical-only features were materially weaker, with mean accuracy around `0.69`. These results suggest that stylistic separation is possible, but that higher-value signals lie in sentence structure, function-word behaviour, punctuation, and related proxy stylistic markers rather than in lexical richness alone.

The main limitation is that the current report is based on a partial `v2` pilot rather than a fully completed large-scale experiment with validated multiclass outputs for all four models. The findings are therefore promising but should be interpreted as pilot evidence rather than definitive proof of robust cross-model stylometric attribution.

## 1. Introduction

Generative language models are increasingly capable of producing text that is fluent, coherent, and superficially similar to formal academic prose. This creates a practical and scholarly problem. In educational and research contexts, it is no longer sufficient to ask whether generated text is readable or factually plausible; it is also necessary to examine whether machine-generated writing leaves stylistic traces that remain detectable after content is constrained by the same source material. Stylometry offers a relevant framework for this task because it focuses on measurable aspects of writing style such as sentence structure, lexical variation, function-word usage, punctuation behaviour, and other low-level linguistic regularities [REF: stylometry foundations].

The present project studies this problem through academic abstracts. Abstracts are a useful unit of analysis because they are short enough to support consistent preprocessing while still containing formal discourse markers, information density, and authorial choices about structure and emphasis. They also matter in practice: abstracts are high-visibility research artifacts and a plausible target for AI-assisted drafting. If LLM-generated abstracts differ systematically from human-written ones at the stylistic level, this has implications for authorship analysis, educational integrity, computational linguistics, and the broader study of human–AI writing interaction.

At the same time, the problem should not be oversimplified. Modern detectors often overfit to superficial characteristics, domain artifacts, or generation settings rather than capturing stable stylistic signatures [REF: AI detection critique]. Moreover, academic writing is itself a constrained genre. Authors tend to follow disciplinary norms, which can reduce the variance that stylometry depends on. A defensible project in this area therefore needs to do more than report classifier accuracy. It must describe the corpus carefully, justify feature design, prevent train–test leakage, explain what is and is not evidenced by the data, and interpret results in relation to both technical and methodological limitations.

This report presents a config-driven pilot study of human versus LLM-generated academic abstracts. The implemented `v2` pipeline supports generation from multiple LLMs, extraction of multiple stylometric feature families, and group-aware cross-validation for both binary and multiclass classification. The current saved results directory provides strongest verifiable evidence for the binary task distinguishing `human` from `gemma-2-9b-it`, while the feature table confirms that the broader multi-model corpus has been constructed at the feature level.

The report addresses three research questions:

1. Can feature-based stylometry distinguish human-written academic abstracts from LLM-generated rewrites of the same source items?
2. Which stylometric feature families appear most informative in this setting?
3. What conclusions can be drawn from the current `v2` pipeline outputs, and what limitations prevent stronger generalisation?

The remainder of the report is structured as follows. Section 2 reviews relevant literature on stylometry, AI-generated text detection, and feature-based attribution. Section 3 describes the methodology and implementation of the `v2` pipeline. Section 4 presents and interprets the available results. Section 5 concludes by answering the research questions and outlining the most important next steps.

## 2. Literature Review

### 2.1 Stylometry and Authorship Analysis Foundations

Stylometry is traditionally concerned with identifying authorship or stylistic similarity through quantifiable linguistic patterns. Classical work in authorship attribution emphasises that function words, sentence-level measures, and distributional regularities can reveal stable stylistic tendencies that are less topic-dependent than content vocabulary [REF: stylometry foundations]. The core idea is that style is partly habitual: even when subject matter changes, writers tend to exhibit recurring structural preferences, such as typical sentence length, punctuation density, or the use of connectives and hedging expressions.

This is attractive for the present project because academic abstracts are highly topic-constrained. If topic-driven lexical overlap dominates, pure lexical cues become less reliable as indicators of authorship class. A stylometric approach instead seeks signals in how information is packaged rather than what content is mentioned. However, stylometry has never been a magic solution. It works best when corpus construction, genre control, and evaluation design are rigorous. Where datasets are small, imbalanced, or confounded by topic, estimated stylistic patterns may reflect corpus artifacts instead of genuine authorial behaviour [REF: stylometry evaluation risks].

### 2.2 AI-Generated Text Detection

The recent literature on AI-generated text detection spans watermarking, probabilistic detection, classifier-based discrimination, and forensic linguistic approaches [REF: AI text detection benchmark]. Many contemporary systems achieve high performance under controlled settings, but published results often degrade when prompts, domains, or model families change. This has two implications. First, content-sensitive detectors may exploit model-specific lexical or semantic tendencies that do not generalise. Second, reported accuracy can overstate practical robustness when evaluation does not adequately separate generation conditions or prevent leakage [REF: robustness in machine text detection].

For academic writing in particular, a stylometric approach offers a narrower but arguably more interpretable target. Instead of trying to infer “machine-ness” from global semantics, it evaluates whether generated text systematically differs from human text in stylistic execution. This is conceptually closer to forensic authorship work. The tradeoff is that style-based detection may be less powerful if models learn to imitate genre conventions well. A rigorous project therefore needs both quantitative results and careful explanation of what the features actually measure.

### 2.3 Linguistic Feature Families

Feature design is central to stylometric work. Lexical richness metrics such as type-token ratio, average word length, or hapax ratio have long been used because they offer simple summaries of vocabulary behaviour [REF: lexical richness methods]. Yet lexical features often mix style with topic and corpus size effects. In short texts, their stability is limited, and in domain-constrained genres they may not be sufficiently discriminative on their own.

Syntactic and structural features provide an alternative. Sentence count, average sentence length, variation in sentence length, and distributional patterns in function words can capture how writers organise and pace information [REF: syntax in stylometry]. Punctuation behaviour can also be informative because it reflects stylistic decisions about clause management, emphasis, and formal structure. In academic prose, citation markers and hedging terms are especially relevant because they relate to epistemic stance and disciplinary convention [REF: academic stance markers].

The current project also includes proxy feature families labelled `writeprints`, `stylometrix`, and `stanza`. These should be treated carefully. In the present implementation they are Python-only approximations rather than full native outputs from the original external toolkits. Methodologically, this is acceptable for a pilot so long as it is disclosed, but it weakens any claim that the analysis reproduces those frameworks in a strict sense. For a stronger final study, either the proxies should be defended explicitly as bespoke engineered features or the external toolchains should be integrated directly.

### 2.4 Supervised Classification for Text Attribution

Supervised learning is the standard evaluation framework for stylometric attribution tasks. Logistic Regression is widely used as an interpretable baseline because it performs well on moderately sized tabular feature sets and allows direct inspection of coefficients [REF: interpretable classification for stylometry]. Random Forest is useful as a complementary non-linear baseline because it can capture feature interactions and threshold effects without strong parametric assumptions [REF: random forest authorship attribution].

However, classifier choice is less important than evaluation design. If texts derived from the same source appear across training and test folds, performance can be inflated by leakage. The present project addresses this by using grouped cross-validation keyed by `source_id`. This is a defensible choice because each human abstract and its generated rewrites are linked to the same source item. Preventing source leakage is necessary if the goal is to learn stylistic separation rather than document-specific overlap.

### 2.5 Limitations and Risks in the Current Research Area

The literature suggests several risks that are directly relevant here. First, detector performance often depends heavily on prompt style and model family [REF: prompt sensitivity in AI detection]. Second, academic writing is a constrained genre, which can reduce stylistic variability and amplify the effect of preprocessing choices. Third, results may be difficult to compare across studies because “human” and “machine” corpora are frequently assembled under different conditions. Fourth, evaluation metrics alone do not resolve questions of validity. A high ROC-AUC is useful, but it does not prove that the model is capturing stable stylistic behaviour rather than artifacts of prompt design, source filtering, or corpus composition.

These issues inform the design and interpretation of the present project. The study should therefore be read as a reproducible pilot with technically credible controls, but not as a finished large-scale benchmark. Its value lies in the implemented pipeline, the transparency of the dataset construction, and the early evidence about which feature families matter most in this specific human-versus-LLM abstract setting.

## 3. Methodology and Implementation

### 3.1 Research Design

The project uses a supervised stylometric classification design. Human-written academic abstracts are treated as one class, and LLM-generated rewrites of matched source abstracts are treated as comparison classes. The overall `v2` design supports both binary classification (`human` versus one model) and multiclass classification (`human` plus several model labels simultaneously).

The current report is explicitly based on a partial `v2` pilot. This matters. The feature table confirms that a five-class corpus was assembled for the analysed subset, but the currently verified fold-level result files in `reports/results/v2` correspond only to the binary `human` versus `gemma-2-9b-it` task. The report therefore distinguishes between implemented system capability and saved evaluation evidence.

### 3.2 Data Pipeline

The human data source is `data/processed/human_abstracts_min80.csv`, which contains Scopus-derived abstracts with metadata such as `source_id`, `title`, `abstract`, `year`, and `doi`. The analysis configuration indicates that abstracts are processed using `source_id` as the grouping key and `abstract` as the main text field. The broader processed human file contains `997` rows, while the analysed `master_with_features.csv` contains a balanced subset of `200` human abstracts matched to generated counterparts.

The LLM dataset is stored in `data/processed/llm_abstracts_v2.csv`. It contains `800` rows in total: `200` generated abstracts from each of four models:

- `llama-3.1-8b-instruct`
- `gemma-2-9b-it`
- `qwen-2.5-7b-instruct`
- `mistral-7b-instruct-v0.3`

All saved rows are labelled with the `factual` prompt type. The generation pipeline uses the original abstract as the source text and asks the LLM to produce a rewritten abstract. The study therefore controls topic and source content more tightly than a free-generation benchmark would. This design is appropriate for stylometry because it reduces the chance that simple topic drift dominates the classification outcome.

### 3.3 LLM Generation Workflow

The generation stage is implemented in `run_generation.py` and controlled through `config.yaml`. The configuration specifies:

- the human input path
- output CSV and JSONL paths
- the factual prompt file
- `sample_size: 300`
- `resume: true`
- `strict_matrix: true`
- the list of configured models and generation parameters

The code supports both `transformers` and `ollama` backends, though the present `v2` setup uses `transformers`. For each source abstract, the script constructs a prompt using the factual prompt template and records generated outputs with metadata such as model name, model ID, temperature, and top-p. The pipeline is designed to append JSONL records during execution and persist CSV outputs, which is good practice for long-running jobs on HPC systems where failures or quota interruptions can occur.

From a research perspective, the important point is that generation is source-matched. Each model receives the same source abstract, which makes later comparisons more defensible because content provenance is shared across human and generated instances.

### 3.4 Feature Engineering

Feature extraction is implemented in `src/stylometry.py` and `src/feature_engine_v2.py`. The base stylometric set includes:

- lexical features: word count, type count, average word length, type-token ratio, hapax ratio
- sentence-level features: sentence count, mean sentence length, standard deviation of sentence length
- punctuation rates
- function-word frequencies
- hedge markers
- citation markers

The `v2` engine also adds three proxy feature families:

- `writeprints`-like character and punctuation ratios plus top trigram frequencies
- `stylometrix`-like proxy rates for long words, modal verbs, and pronouns
- `stanza`-like POS-style proxies based on surface word endings

This is a defensible pilot design because it broadens stylistic coverage without requiring heavy external NLP infrastructure. However, the proxies are not full toolkit replications. The report must therefore treat them as engineered approximations, not as direct outputs from canonical `Writeprints`, `Stylometrix`, or `stanza` pipelines.

The active feature sets listed in `config.yaml` are:

- `lexical`
- `syntactic`
- `combined`
- `writeprints`
- `stylometrix`
- `stanza`

The currently saved fold-level `v2` results only cover `lexical`, `syntactic`, and `combined` on the Gemma binary task, so the present discussion is limited to those verified outputs.

### 3.5 Classification and Evaluation

The analysis stage is implemented in `run_analysis.py`. The script first builds a common master dataset by intersecting `source_id` values across the human data and all required model outputs. It then extracts features and runs classification with `StratifiedGroupKFold`, using `source_id` as the grouping variable. This is the correct design choice for the project because each source paper can generate multiple related texts. Group-aware splitting reduces leakage and produces more credible estimates than naive random splitting.

Two classifiers are implemented:

- Logistic Regression with standardisation
- Random Forest with class balancing

The configured number of folds is `5`, and the random seed is `42`. Binary evaluation uses accuracy, precision, recall, F1-score, ROC-AUC, Matthews correlation coefficient, and macro-F1. Multiclass evaluation is also implemented in the code, though verified multiclass result files are not currently present in the saved `v2` results directory.

### 3.6 Reproducibility

The reproducibility strategy is one of the project’s stronger design decisions. The workflow is driven by `config.yaml`, with separate entry points for generation and analysis:

- `run_generation.py`
- `run_analysis.py`

This design makes it easier to:

- change model lists without rewriting scripts
- rerun experiments with different sample sizes or feature sets
- preserve a clear separation between data generation and analysis

The project is therefore reproducible at the pipeline level, even though the current saved results represent only a partial pilot execution of the full intended experiment.

## 4. Results and Discussion

### 4.1 Available `v2` Evidence

The strongest verified `v2` evidence comes from the `reports/results/v2` directory. The master feature table contains `1,000` rows in total:

- `200` human abstracts
- `200` `Llama 3.1` abstracts
- `200` `Gemma 2` abstracts
- `200` `Qwen 2.5` abstracts
- `200` `Mistral 7B` abstracts

This confirms that the feature-engineered corpus for the balanced pilot subset exists. However, the saved fold-level evaluation outputs presently cover only binary classification between `human` and `gemma-2-9b-it`, across three verified feature sets:

- `lexical`
- `syntactic`
- `combined`

No saved multiclass fold files are present in the current results directory. This is a substantive limitation because it means the report can only make direct metric-based claims about the Gemma binary task, not about the full four-model attribution problem.

### 4.2 Binary Human vs Gemma Results

Table 1 summarises mean performance across the five folds.

| Feature set | Estimator | Accuracy | Precision | Recall | F1 | ROC-AUC | MCC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Lexical | Logistic Regression | 0.6875 | 0.6891 | 0.6950 | 0.6889 | 0.7861 | 0.3786 |
| Lexical | Random Forest | 0.6925 | 0.6759 | 0.7450 | 0.7073 | 0.7601 | 0.3890 |
| Syntactic | Logistic Regression | 0.8175 | 0.8185 | 0.8300 | 0.8195 | 0.9095 | 0.6435 |
| Syntactic | Random Forest | 0.8225 | 0.8324 | 0.8100 | 0.8202 | 0.9166 | 0.6465 |
| Combined | Logistic Regression | 0.8100 | 0.8127 | 0.8200 | 0.8118 | 0.9187 | 0.6279 |
| Combined | Random Forest | **0.8400** | **0.8487** | **0.8300** | **0.8385** | **0.9249** | **0.6814** |

The headline result is that combined and syntactic feature sets materially outperform lexical-only features. The best system is Random Forest with combined features, achieving mean accuracy of `0.8400` and mean ROC-AUC of `0.9249`. Even the syntactic feature sets alone are competitive, reaching `0.8175` to `0.8225` mean accuracy with ROC-AUC values above `0.90`. By contrast, lexical-only performance is noticeably lower, with accuracy close to `0.69`.

This pattern is important. It suggests that the most useful distinctions between human and Gemma-generated abstracts in this setting are not driven primarily by lexical richness. Instead, the better signal appears to come from structural and distributional properties: sentence length behaviour, function-word profiles, punctuation rates, hedging, citation-style markers, and related combined proxies.

### 4.3 Interpretation of Feature Effects

The relative weakness of lexical-only features is unsurprising. Because the LLM rewrites are based on the same source abstracts, strong lexical overlap is expected. Both human and generated texts operate in the same academic domain and address the same underlying research topic. This reduces the discriminative power of simple vocabulary-based measures. In other words, lexical features are probably constrained by the source text itself.

Syntactic and combined features perform better because they capture how ideas are arranged rather than which content words are selected. The sample rows in `llm_abstracts_v2.csv` support this interpretation. The generated abstracts are coherent and topical, but they often exhibit smoother sentence planning, regularised exposition, and more standardised discourse packaging than the corresponding human texts. A classifier that can exploit sentence-length distribution, punctuation density, or function-word patterns therefore has a better chance of separating the classes.

The Random Forest advantage over Logistic Regression is modest but consistent in the combined setting. This suggests that at least part of the decision boundary is non-linear. That is plausible for stylometric data, where multiple weak cues can interact. For example, moderate differences in sentence length may become more informative when combined with punctuation ratios or stance markers. A linear model remains valuable as a baseline, but the stronger Random Forest result implies that the class signal is not fully captured by simple additive effects.

### 4.4 What the Results Do and Do Not Show

The current results support a cautious but positive answer to the first research question. Yes, feature-based stylometry can separate human abstracts from at least one LLM output class in this source-matched academic setting with reasonably strong performance. The ROC-AUC values above `0.90` for combined and syntactic models are especially encouraging.

However, the current evidence does **not** justify stronger claims such as:

- stylometry can robustly distinguish all four model families from human writing under the current pipeline
- multiclass attribution among `human`, `Llama`, `Gemma`, `Qwen`, and `Mistral` has been fully demonstrated
- the findings will necessarily generalise across prompts, domains, or longer-form academic writing

Those claims would require verified saved outputs for the full multiclass evaluation and ideally stronger interpretability evidence about which features drive separation across different models.

### 4.5 Validity, Limitations, and Threats to Interpretation

The report’s main methodological strength is the use of grouped cross-validation by `source_id`. This reduces an important source of leakage and makes the binary results more believable. The balanced pilot subset is another strength because it prevents simple majority-class effects from distorting the metrics.

The main limitations are as follows.

First, the study is a partial `v2` pilot. Although the feature table includes all five classes, the saved fold-level evidence in the current results directory is limited to the Gemma binary task. This sharply narrows the scope of defensible claims.

Second, the proxy feature families (`writeprints`, `stylometrix`, `stanza`) have been engineered as stand-ins for toolkit outputs. That is acceptable for pipeline development, but a stronger research contribution would either validate those proxies or replace them with genuine toolkit extraction.

Third, the generation process itself may introduce systematic regularities that are model- and prompt-specific. The factual prompt constrains content and probably improves comparability, but it also means the study is not testing free-form authorship style. It is testing stylometric behaviour under prompt-mediated rewriting conditions.

Fourth, the HPC and storage issues encountered during model generation are relevant only indirectly. They do not invalidate the verified Gemma binary results, but they do help explain why the broader intended `v2` evaluation is currently incomplete. This should be mentioned briefly as an execution constraint rather than used as an excuse for missing evidence.

Fifth, the unit of analysis is the abstract. This is defensible, but abstracts are short texts. Short-text stylometry is inherently more difficult than long-form authorship analysis because individual documents provide fewer stylistic observations [REF: short text stylometry].

### 4.6 Overall Discussion

Taken together, the results indicate that stylometric separation between human and LLM-generated academic abstracts is feasible in this pipeline, but not all feature families contribute equally. Lexical richness alone is too weak to support a strong detector in this setting. The more reliable signal lies in structural and functional aspects of writing. That finding is methodologically coherent and aligns with the broader stylometric literature, which generally places higher value on style markers that are less topic-sensitive [REF: stylometry foundations].

At the same time, this is not yet a finished stylometric benchmark. The project is stronger as a reproducible pilot than as a final definitive experiment. Its main contribution at this stage is the construction of a transparent config-driven workflow, a balanced multi-model feature table, and early evidence that combined and syntactic features can distinguish at least one LLM class from human abstracts with good performance.

## 5. Conclusions

This project set out to test whether stylometric analysis can distinguish human-written academic abstracts from LLM-generated rewrites in a reproducible, feature-based classification pipeline. Based on the currently verified `v2` outputs, the answer is yes in the binary `human` versus `gemma-2-9b-it` setting. The strongest performance was achieved by Random Forest with combined features, reaching mean accuracy of `0.8400`, F1 of `0.8385`, and ROC-AUC of `0.9249`.

The most important substantive finding is that lexical-only features are not sufficient on their own. Combined and syntactic feature sets perform substantially better, suggesting that stylistic separation is driven more by structural and distributional behaviour than by vocabulary richness. This is a plausible result for source-matched academic abstracts, where topic overlap constrains lexical variation.

The main limitation is that the report is based on a partial `v2` pilot rather than a fully completed final experiment. Although the feature table includes all four LLM classes plus human texts, the currently saved fold-level evaluation evidence is limited to the Gemma binary task. The present conclusions should therefore be treated as credible pilot findings rather than broad claims about universal LLM stylometric detectability.

The most important next steps are clear. First, the multi-model `v2` experiment should be completed and saved with verified binary and multiclass outputs for all intended models. Second, the feature layer should be strengthened either by validating the current proxy features or by integrating canonical external toolchains. Third, the analysis should be extended with more explicit interpretability outputs, including per-feature importance summaries and, ideally, error analysis across models and feature families. Those steps would convert the current pipeline from a strong pilot into a more robust research contribution.

## References

Replace the placeholders below with real APA-formatted references before submission.

- [REF: stylometry foundations]
- [REF: stylometry evaluation risks]
- [REF: AI text detection benchmark]
- [REF: robustness in machine text detection]
- [REF: lexical richness methods]
- [REF: syntax in stylometry]
- [REF: academic stance markers]
- [REF: interpretable classification for stylometry]
- [REF: random forest authorship attribution]
- [REF: prompt sensitivity in AI detection]
- [REF: short text stylometry]
- [REF: AI detection critique]
