# Amendment 2 (approved 2 Nov 2026): open-weight models on local GPUs

Approved by the author on 2 Nov 2026 before any open-model run; summary in `docs/analysis_plan.md` (Amendments).


## 1. Why
All models in the main study are from one vendor (Anthropic). Reviewers will ask whether the findings (SAIL
under-classification, the effect of the rule card, the near-zero error of extract + calculator) hold for other model
families. Open-weight models also matter for practice: a drone operator or a small authority can run them locally
without sending operational data to a third party. This amendment adds open-weight models from four other developers.
Commercial APIs (GPT, Gemini) will be a separate Amendment 3.

## 2. What stays the same (frozen by Amendment 1)
- Scenarios: `data/scenarios.jsonl` (sha256 `1af09bad…7325`), ground truth and encoding choices unchanged.
- Prompts: system prompts, rule card and extraction schema in `soraeval/prompts.py` unchanged, word for word, in both
  languages. No model-specific prompt changes, even if a model formats its answers badly.
- Set-ups: closed book, rule card (`reference`), extract + calculator; both languages; 1 run per item.
- Grading, parsing and outcomes: `soraeval/parse.py` and `soraeval/analyze.py` unchanged. Primary outcome = SAIL too
  low; secondary = SAIL exact, too high, unparsed, per-step accuracy, trap analysis.
- Output budget: max 32,000 output tokens (thinking + answer), as in the main study.

## 3. Models
One model per developer, the strongest one that runs on a single RTX 5090 (32 GB) with a 40k-token context, using the
default Ollama build (4-bit, Q4_K_M or the developer's own 4-bit release):

| Model (Ollama tag) | Developer | Size on disk | Reasoning set-up |
|---|---|---|---|
| `qwen3.8:27b` | Alibaba (Qwen) | 18 GB | thinking on |
| `gemma4:31b` | Google | 19–20 GB | thinking on |
| `gpt-oss:20b` | OpenAI | 14 GB | reasoning effort "medium" |
| `mistral-small3.2:24b` | Mistral AI | 15 GB | no thinking mode (instruction-tuned only) |

Optional, if time allows after the four above: `gpt-oss:120b` (65 GB, split over both GPUs) as a larger model from the
same family as `gpt-oss:20b`, to show the effect of model size inside one family (as Haiku → Sonnet → Opus does for
Claude). Exact model digests (sha256) are recorded in each result file at run time.

## 4. Settings
- Server: Ollama 0.40.13, native `/api/chat` endpoint (needed to set the context size, the thinking mode and the seed;
  the OpenAI-compatible endpoint silently uses a short default context and truncates long prompts).
- Context `num_ctx` = 40,960 tokens; `num_predict` (max output) = 32,000. Prompt ≈ 2–3k tokens, so output is never cut
  by the context limit.
- Sampling: each model's own default sampling settings as shipped in its Ollama build (temperature, top-p, top-k),
  with a fixed seed (2026). Reason: the Claude 5.5 models also ran with their default sampling (they reject the
  `temperature` setting), and greedy decoding (temperature 0) is known to make reasoning models repeat themselves
  until the token limit. The settings used are written into every result record.
- One model at a time per GPU, one request at a time (`OLLAMA_NUM_PARALLEL=1`) so results do not depend on batching.
- Our own Ollama server on port 11437 (separate from the servers used by other projects on this machine).

## 5. Sample
- Main: the first **200 scenarios** (S00000–S00199), all 3 set-ups × 2 languages = 1,200 graded assessments per model.
  These 200 scenarios were also answered by all four Claude models, so every comparison is paired.
- Extension to the first 500 scenarios (matching Sonnet 5.5) for all models **only if** the infrastructure pilot shows
  that 500 × 6 × 4 models can finish within 7 days of GPU time. This is decided from the pilot's speed alone, before any
  main result is seen, and the decision is written below this amendment.

## 6. Infrastructure pilot (not pooled)
- Scenarios S02990–S02999 (outside every main subset), 10 × 3 set-ups × 2 languages = 60 calls per model.
- Purpose only: check that the server works, that answers are not cut off, that thinking is separated from the answer,
  and measure speed. **No prompt or grading change may follow from the pilot.** Allowed changes: server settings, the
  model tag (e.g. if a model does not load), and fixing bugs in our client code.
- Pilot results are kept in `results/pilot_open_*.jsonl` and are not pooled.

## 7. Analysis
- The same tables as the main study for each open model (`analyze.py`), plus the answers that were cut off at 32,000
  tokens (`stop_reason` = length) reported separately; they count as unparsed, as in the main study.
- Cross-model comparison on the 200 shared scenarios: SAIL too low and SAIL exact for all 8 models (4 Claude + 4 open)
  with Wilson CIs; GEE (logit, exchangeable, clustered by scenario) with model, language and set-up; Holm-corrected
  McNemar tests for EN vs TR within each model and set-up, as in the main study.
- Trap analysis and SORA 2.0 carry-over markers (e.g. "kinetic energy", M3) for each open model, as in the main study.
- Exploratory (labelled as such): closed-book error versus model size, and versus whether the model thinks.

## 8. Limitations to report
- 4-bit quantised versions; results may differ from the full-precision models served by the developers.
- Defaults differ between models (sampling, thinking length); we compare models as shipped, not under identical
  decoding.
- Model knowledge cut-offs differ; some models may never have seen SORA 2.5 (published May 2024).

## 9. Cost and time
$0 API cost. Electricity only. Expected speed and total GPU time to be filled in from the pilot.
