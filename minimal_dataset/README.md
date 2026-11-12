# Minimal dataset: LLM benchmark of JARUS SORA 2.5 drone risk assessment in English and Turkish

Version 1.0, 8 November 2026

Author: Salih Sarp, Department of Computer Engineering, Gebze Technical University, Kocaeli, Türkiye
(ORCID 0000-0001-6674-2590)

Related article: Sarp, S. Do large language models rate drone operations as safer than they are? A deterministic
benchmark of JARUS SORA 2.5 risk assessment in English and Turkish. Manuscript under review.

## What this is

This is the minimal dataset behind the article. It contains every graded model answer and the exact answer it was
graded against, so that every number, table and figure in the article can be recomputed. Each SORA 2.5 operation
(scenario) was described in English and in Turkish. Eight models assessed it in three set-ups:

- **closed book:** task rules only;
- **rule card:** a one-page restatement of SORA 2.5 Steps 2–7 added to the prompt;
- **extract + calculator:** the model only fills a parameter schema and a rule engine computes the classes.

A rule engine that encodes SORA 2.5 Steps 2–7 gives the exact intrinsic and final ground risk class (GRC), initial and
residual air risk class (ARC), tactical mitigation performance requirement (TMPR) and specific assurance and integrity
level (SAIL). The primary outcome is SAIL under-classification (`sail_under` in `assessments.csv`).

| Model | Access | Scenarios (per language) | Assessments |
|---|---|---|---|
| claude-haiku-4-5 | Anthropic API, Message Batches, 31 Oct 2026 | S00000–S00399 (400) | 2400 |
| claude-haiku-5-5 | same | S00000–S01499 (1500) | 9000 |
| claude-sonnet-5-5 | same | S00000–S00499 (500) | 3000 |
| claude-opus-5-5 | same | S00000–S00199 (200) | 1200 |
| gpt-oss:20b | Ollama 0.40.13, one RTX 5090, 3–5 Nov 2026 | S00000–S00499 (500) | 3000 |
| mistral-small3.2:24b | same | S00000–S00499 (500) | 3000 |
| gemma4:31b | same | S00000–S00499 (500) | 3000 |
| qwen3.8:27b | same | S00000–S00499 (500) | 3000 |
| **Total** | | | **27,600** |

Every model saw the first *n* scenarios, so the subsets are nested and paired: S00000–S00199 were assessed by all eight
models, and S00000–S00499 by Haiku 5.5, Sonnet 5.5 and the four open-weight models. Settings: Claude models with effort
medium and max_tokens 32,000. Open-weight models as 4-bit builds (MXFP4 or Q4_K_M) with num_ctx 40,960, seed 2026,
max 32,000 output tokens and the model's default sampling.

## Files

| File | Rows | Content |
|---|---|---|
| `assessments.csv` | 27,600 | One row per graded assessment: model answers (`pred_*`), exact answers (`true_*`), correctness (`ok_*`), primary and secondary outcomes, source of each wrong SAIL, SORA 2.0 markers, extraction accuracy (`ex_*`), tokens, latency and cost |
| `extractions.csv` | 9200 | The parameter values each model extracted in the extract + calculator set-up |
| `repeatability_haiku55.csv` | 300 | Repeatability run: Haiku 5.5, closed book, English, S00000–S00099, three runs each (same columns as `assessments.csv` plus `rep`) |
| `scenarios.csv` | 1500 | Facts of each operation, every intermediate value of the exact answer (iGRC table column, density band, mitigation credits, M1 floor, airspace encounter category, VLOS reduction) and 19 trap flags |
| `scenario_descriptions.jsonl` | 1500 | The English and Turkish operation descriptions given to the models (same facts in both) |
| `data_dictionary.csv` | 224 | Type and meaning of every column |
| `check_key_results.py` | | Recomputes the main results from the CSV files (Python 3.8+, standard library only) |
| `expected_output.txt` | | Output of `check_key_results.py` |
| `SHA256SUMS.txt` | | Checksums of the files above |

Conventions: UTF-8, comma-separated, header row. Yes/no fields are 1/0. A blank cell means not applicable or not given:
for example, `ex_*` outside the extract set-up, a missing model answer, or `true_final_grc` when the iGRC is outside
SORA. Join files on `sid` (scenario) or `key` (assessment).

## Reproducing the results

```
python3 check_key_results.py
```

This prints the primary outcome with Wilson 95% intervals for every model, set-up and language (Table 2
of the article). It also prints the sources of wrong SAILs (Figure 2), extraction accuracy, repeatability
(Fleiss' κ = 0.347) and cost per correct assessment. The paired tests (exact McNemar with Holm correction), the
bootstrap and the GEE models in the article use the same rows. Join `assessments.csv` with the trap flags in
`scenarios.csv` on `sid` for the GEE models.

## Not included

- **Full response texts and model reasoning** (about 170 MB). They are available from the author on request. The code
  (scenario generator, rule engine, prompts and analysis scripts) is at https://github.com/salihsarp/sora25-llm-benchmark.
  The fields derived from the text (`mentions_*`) are included here.
- **The SORA 2.5 Main Body.** It is a JARUS publication and is not redistributed; obtain it from JARUS
  (JAR-DEL-SRM-SORA-MB-2.5, 13 May 2024).
- **Pilot runs.** They are not pooled with the main results (see the article, Section 3.7); available on request.
- **Scenarios S01500–S02999.** The generator produced 3000 operations, but no model assessed these. The generator
  recreates them from seed 2026.

The scenarios are synthetic. Operator names are invented, and place names serve only as settings. No personal data are
included.

## Licence

Creative Commons Attribution 4.0 International (CC BY 4.0). Please cite the related article when using the data.
