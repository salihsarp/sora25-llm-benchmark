# Do LLMs rate drone operations as safer than they are? A SORA 2.5 benchmark in English and Turkish

Code and data for the paper **"Do Large Language Models Rate Drone Operations as Safer than They Are? A Deterministic
Benchmark of JARUS SORA 2.5 Risk Assessment in English and Turkish"** (S. Sarp, Gebze Technical University; manuscript
under review).

Steps 2–7 of the JARUS Specific Operations Risk Assessment (SORA 2.5) are deterministic once the facts of an operation are
fixed. A rule engine therefore gives the exact intrinsic and final ground risk class (GRC), initial and residual air risk class
(ARC), tactical mitigation performance requirement (TMPR) and specific assurance and integrity level (SAIL) for any number of
synthetic operations, with no human labelling. The benchmark measures how often large language models get the SAIL wrong,
and above all how often they set it **too low** (the operation is rated safer than SORA allows). That is the primary
outcome.

- **Scenarios:** 3000 synthetic UAS operations (seed 2026), each written in English and in Turkish with the same facts and
  built-in traps. The first 1500 were assessed.
- **Set-ups:**
  - **closed book:** task rules only;
  - **rule card:** a one-page restatement of the Steps 2–7 rules added to the prompt;
  - **extract + calculator:** the model only extracts the operation's facts and the rule engine computes the classes.
- **Models:**
  - Claude Haiku 4.5, Haiku 5.5, Sonnet 5.5 and Opus 5.5 (Anthropic Message Batches API, 31 Oct 2026);
  - gpt-oss 20B, Mistral Small 3.2 24B, Gemma 4 31B and Qwen3.8 27B (Ollama 0.40.13, 4-bit builds, one RTX 5090, 3–5 Nov 2026).
- **Size:** 27,600 graded assessments, plus a repeatability run.

This repository holds the code and a minimal dataset (`minimal_dataset/`): every graded assessment with the model's
and the exact answer, the assessed scenarios with their English and Turkish descriptions, the extracted parameters and
the repeatability run. The full model responses and reasoning texts (about 170 MB) and the pilot runs are available from
the author on request.

## Main result

SAIL too low (% of assessments), English · Turkish:

| Model (scenarios per language) | Closed book | Rule card | Extract + calculator |
|---|---|---|---|
| Claude Haiku 4.5 (400) | 70.5 · 76.0 | 27.2 · 20.5 | 0.2 · 0.0 |
| Claude Haiku 5.5 (1500) | 37.9 · 39.3 | 1.9 · 0.9 | 0.0 · 0.0 |
| Claude Sonnet 5.5 (500) | 27.2 · 37.0 | 2.6 · 1.4 | 0.0 · 0.0 |
| Claude Opus 5.5 (200) | 3.5 · 7.0 | 0.0 · 0.0 | 0.0 · 0.0 |
| gpt-oss 20B (500) | 71.2 · 68.2 | 15.8 · 16.6 | 6.4 · 10.8 |
| Mistral Small 3.2 24B (500) | 53.6 · 74.0 | 45.8 · 57.6 | 0.8 · 2.4 |
| Gemma 4 31B (500) | 78.6 · 85.2 | 4.2 · 5.6 | 0.6 · 0.0 |
| Qwen3.8 27B (500) | 79.6 · 76.4 | 4.2 · 4.0 | 0.8 · 5.2 |

Wilson intervals and exact SAIL accuracy are printed by `minimal_dataset/check_key_results.py`; every graded
assessment is in `minimal_dataset/assessments.csv`.

## Repository layout

| Path | Content |
|---|---|
| `soraeval/sora25.py` | Ground-truth engine: SORA 2.5 Main Body Tables 2, 5, 6, 7 and 14, Figure 6 and the Step 5 VLOS rule |
| `soraeval/generator.py` | Scenario sampler and English/Turkish renderer (same facts in both), trap flags |
| `soraeval/prompts.py` | System prompts, rule card and extraction schema in English and Turkish; the three set-ups |
| `soraeval/parse.py` | JSON extraction, answer normalisation, extraction → engine ("calculator") |
| `soraeval/providers.py` | Anthropic API (sync and Message Batches), OpenAI-compatible endpoints and native Ollama; standard library only |
| `soraeval/runner.py` | Command line: `generate`, `run`, `batch-submit`, `batch-collect`; resumable JSONL output, budget cap |
| `soraeval/analyze.py` | Grading, Wilson intervals, exact McNemar with Holm correction, trap tables, SORA 2.0 markers → Markdown and graded CSV |
| `soraeval/stats_extra.py` | GEE models, paired bootstrap, Fleiss' κ and figures (needs pandas, statsmodels, matplotlib) |
| `tests/` | Offline tests (`test_all.py`) and a dry run of the whole pipeline with simulated models (`simulate.py`) |
| `docs/` | Pre-specified analysis plan with Amendments 1 and 2 (it also lists result files that are not in this repository) |
| `minimal_dataset/` | Compact CSV version of everything the paper reports, with data dictionary and a standard-library check script (see its README) |
| `scripts/` | Rebuild the minimal dataset from the raw model outputs |
| `run_all.sh`, `run_open_*.sh`, `serve_ollama_gpu1.sh` | The exact commands used for the Claude runs, the open-weight pilot and main runs, and the dedicated Ollama server |

## Requirements

- Python ≥ 3.9. The engine, generator, runner and `analyze.py` use only the standard library.
- `stats_extra.py` and the figures need the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Reproduce

```bash
python3 tests/test_all.py                                    # offline tests
python3 tests/simulate.py                                    # pipeline dry run with simulated models
python3 -m soraeval.runner generate --n 3000 --seed 2026     # writes data/scenarios.jsonl
sha256sum data/scenarios.jsonl                               # 1af09badfe8acb3d1af1cbab92f30d45e36e4609d0e0dfcfeafefb7b5f4b7325
python3 minimal_dataset/check_key_results.py                 # headline results from the minimal dataset (standard library only)
```

With the full model outputs in `results/` (available on request), the paper's tables, statistics and figures are
rebuilt with:

```bash
python3 -m soraeval.analyze --results results/main_*.jsonl --out results/summary_main.md
python3 -m soraeval.stats_extra --results results/main_*.jsonl \
  --stability results/stability_haiku55.jsonl --outdir results/main_extra
python3 scripts/build_minimal_dataset.py && python3 scripts/make_dictionary.py   # rebuilds minimal_dataset/
```

**Running models again.**
- **Claude:** needs `ANTHROPIC_API_KEY` in the environment or in a `.env` file. The commands used are in `run_all.sh`.
  Claude 5.5 models take no `temperature`; the runs used `--effort medium` and a 32,000-token output limit.
  `batch-submit` / `batch-collect` use the Message Batches API at half price.
- **Open-weight models:** start `serve_ollama_gpu1.sh`, then run `run_open_main.sh`. These runs use
  `--provider ollama` (native `/api/chat` with `num_ctx` 40,960 and seed 2026). Ollama's OpenAI-compatible endpoint
  silently uses a shorter context.

## Ground-truth encoding choices

From the SORA 2.5 Main Body (JAR-DEL-SRM-SORA-MB-2.5, 13 May 2024):

- **iGRC:**
  - The column is the left-most one that covers both the maximum dimension and the maximum speed.
  - The row is the highest population density in the footprint.
  - ≤ 250 g and ≤ 25 m/s gives iGRC 1.
  - Grey cells give "certified".
- **Mitigations:**
  - Robustness is the lower of integrity and assurance; credits follow Table 5, and N/A gives no credit.
  - The M1 credit is floored at the column's controlled-area value, and M2 is applied after the floor.
  - A final GRC above 7 gives "certified".
- **Initial ARC:** Figure 6, using the top of the operational volume (flight geography plus contingency volume).
- **Residual ARC:** VLOS, or BVLOS with an airspace observer next to the pilot, lowers the class by one, but never to ARC-a. No Annex C claims.
- **TMPR:** none under VLOS, otherwise Table 6.
- **Out of scope:** containment (Step 8).
- **Excluded cases:** the generator never produces boundary values or combinations that SORA forbids or leaves open. See `docs/analysis_plan.md`.

The SORA 2.5 Main Body is a JARUS publication and is **not** included; obtain it from JARUS. The rule card in
`soraeval/prompts.py` restates the rules and table values in our own words and cites JARUS.

## Licence

- **Code:** MIT (`LICENSE`).
- **Data:** CC BY 4.0 (`LICENSE-DATA.md`) for the minimal dataset. The model answers in it were generated by the models
  named above.

## Citation

See `CITATION.cff`. Please cite the paper once it is published.

## Use of generative AI

Claude (Anthropic) was used as an assistive tool for code development, data visualization, reference verification and
language editing. Four of the evaluated models are also Claude models. The author designed the study, ran the experiments
and checked all results.
