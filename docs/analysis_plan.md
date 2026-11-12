# Pre-specified analysis plan — SORA-LLM (v1, 31 Oct 2026; Amendments 1–2)

Author: Salih Sarp, Gebze Technical University. Written on 31 Oct 2026 before the main runs. Amendments are dated and
were written before the runs they cover; later dated entries only record what happened.
Working title: "Do LLMs rate drone operations as safer than they are? A deterministic benchmark of JARUS SORA 2.5 risk
assessment in English and Turkish".

## Novelty (checked 31 Oct 2026)
- No published accuracy benchmark of LLMs on SORA. Closest: arXiv 2603.09999 (Immordino et al. 2026; RAG assistant for UAS
  safety assessment and compliance; no SAIL accuracy metrics); arXiv 2303.02053 (Habibi et al., ICUAS 2023; automating SORA,
  pre-LLM); Iyenghar 2025 Drones 9(9):657 (UAVThreatBench, LLM cyber-threat identification — different task, no exact truth).
  Deterministic calculators exist (Norwegian CAA training calculator, updated 22 Oct 2025; Dronedesk) — tools, not LLM
  evaluations.

## Ground truth (no human labels)
Engine `soraeval/sora25.py` encodes JARUS SORA Main Body 2.5 (JAR-DEL-SRM-SORA-MB-2.5, 13 May 2024; page numbers = "x of 57"
footers of the copy mirrored by the Czech CAA):
- Step 2: Table 2 iGRC (p.34); left-most column matching BOTH dimension and speed (p.35); highest density in the footprint
  (p.35); 250 g / 25 m/s rule (p.34); Table 3 qualitative descriptors (p.36); grey cells = not part of SORA.
- Step 3: Table 5 credits (p.38); robustness = lower of integrity and assurance (p.26, Table 1); M1 floor = column's
  controlled-ground-area value (p.39 (f)); M2 applied after the floor (as in the Norwegian calculator); final GRC > 7 =
  certified (p.40). Annex B 2.5: M1(A) medium cannot be combined with M1(B) (Norwegian calculator enforces the same).
- Step 4: Figure 6 decision tree (p.41; read from the rendered figure); vertical limit = flight geography + contingency volume.
- Step 5: VLOS, or BVLOS with an airspace observer alongside the pilot, lowers the initial ARC by one class, never to ARC-a
  (4.5.4, p.43-44). Annex C local-density reductions are out of scope (the task prompt says no such claim is made).
- Step 6: Table 6 TMPR (p.45); VLOS needs no TMPR (4.6.4 (a),(c)). Step 7: Table 7 SAIL (p.47). Step 9 Table 14 OSOs kept in
  the engine (not asked of the models). Step 8 containment out of scope.
- Verification: 24 offline tests; simulated "oracle"/"naive" dry run; blind check by a second Claude agent on 30
  trap-stratified scenarios worked from the source PDF — 30/30 agreement on every field (results/verification/).

## Scenarios
- 3,000 operations, seed 2026 (sha256 of data/scenarios.jsonl = 1af09bad…7325, identical on Python 3.10 and 3.13). Same facts
  in English and Turkish (Turkish number format, SHGM context). ~195 words each.
- Never generated (ambiguous or forbidden in SORA): values on a column/band edge, operational-volume top 135–170 m, M1(A) high,
  M1(C) above low, M1(A) medium with M1(B), M1(A) over open-air assemblies or for UA > 25 kg, final GRC < 1, EVLOS.
- Trap flags: speed drives column; multi-segment route; qualitative density; controlled area; cruise distractor;
  integrity≠assurance; credit at N/A robustness; M1 floor binds; M2 below column floor (sensitivity); contingency pushes above
  500 ft; VLOS reduction; VLOS at ARC-b (no ARC-a); airport precedence; airport E/G; 250 g rule; near-250 g; ERP distractor;
  out of scope.

## Conditions (soraeval/prompts.py)
1. closed_book — task rules only (credit only claimed mitigations, stated robustness taken as accepted; no collision-risk map,
   no Annex C claim; operator claims any reduction the Main Body itself grants for the way the operation is flown, no other
   strategic mitigation or agreement; out of scope -> "certified").
2. reference — same + a rule card restating Steps 2-7 (our wording; table numbers as facts).
3. extract — model only extracts parameters into a schema; the engine computes ("LLM + calculator").

## Hypotheses
H1 closed-book under-classification ≥ 5% for every model and leaning unsafe; H2 rule card and then calculator reduce it;
H3 each trap raises the error of the step it targets (+ SORA 2.0 carry-over in closed book); H4 Turkish no better than English
closed book. RQ5 compares models.

## Outcomes and statistics
- Primary: SAIL under-classification (predicted < true on I<…<VI<certified); unparsed = not-under (sensitivity: under).
- Secondary: SAIL exact; step accuracy; GRC/ARC under; missed "certified"; self-consistency; error source; extraction
  accuracy; cost per correct assessment.
- Tests: Wilson 95% CIs; exact McNemar (paired) with Holm within families; exact binomial under vs over; logistic GEE with
  scenario clusters on trap flags + language; bootstrap (2,000) for between-model differences on the 200 shared scenarios;
  Fleiss κ for repeats.

## Runs (done)
Pilots ($3.18) -> Amendment 1 -> Message Batches 31 Oct 2026: Haiku 5.5 n=1,500; Sonnet 5.5 n=500; Opus 5.5 n=200; Haiku 4.5
n=400 (EN+TR × 3 conditions = 15,600) + repeatability Haiku 5.5 100×3 closed-book EN. Total $50.17.

## Amendments
- 31 Oct 2026, AMENDMENT 1 (after pilots, before main runs): max_tokens 6,000 -> 32,000 (58/80 closed-book answers were cut
  off by adaptive thinking) and effort fixed at "medium"; prompt rewrite of two ambiguous phrases ("assume ... any ATC
  agreement this requires is in place" made the model invent ATC-based ARC reductions; "the robustness that follows from the
  information given" made it downgrade claimed robustness). Pilot data not pooled. 5.5 models reject `temperature`.
- 2 Nov 2026, AMENDMENT 2 (approved by the author before any open-model run; full text `docs/amendment2_open_models.md`):
  open-weight models on gtu-dsa (RTX 5090, 32 GB), one per developer, default Ollama 4-bit builds: qwen3.8:27b (thinking
  on), gemma4:31b (thinking on), gpt-oss:20b (reasoning "medium"), mistral-small3.2:24b (no thinking mode); optional
  gpt-oss:120b over both GPUs (size effect within one family). Model digests recorded per record.
  - Frozen: scenarios, prompts (word for word, EN+TR, no model-specific changes), 3 set-ups, grading, outcomes, 32,000 output
    tokens.
  - Settings: Ollama 0.40.13 native /api/chat, num_ctx 40,960, num_predict 32,000, seed 2026, each model's default
    sampling (as for the 5.5 models; greedy decoding makes reasoning models loop), one request at a time, own server on
    GPU 1 port 11437.
  - Sample: first 200 scenarios (shared with all 4 Claude models; paired) × 3 set-ups × 2 languages = 1,200 per model.
    Extend all models to the first 500 only if the pilot speed alone shows 500 × 6 × 4 models fits in 7 GPU-days; the
    decision is written here before any main result is seen.
  - Infrastructure pilot (not pooled): S02990–S02999 × 3 × 2 = 60 calls per model; may change server settings, model tag
    or client bugs only — never prompts or grading.
  - Analysis: same tables per model + truncations (cut off at 32,000 = unparsed); 8-model comparison on the 200 shared
    scenarios (Wilson CIs; GEE with model, language, set-up; Holm-McNemar EN vs TR); traps and SORA 2.0 markers.
    Exploratory: closed-book error vs model size and vs thinking.
  - Limitations: 4-bit quantisation; decoding defaults differ between models; knowledge cut-offs differ.
  - 2 Nov 2026, PILOT OUTCOME AND SAMPLE DECISION (before any main open-model run): 4 × 60 calls, 0 errors, 0 cut off at
    32,000 tokens (longest output 12,307), thinking returned separately for gpt-oss/Qwen/Gemma; 2 unparsed (extraction
    gave values the engine rejects; model errors, not client bugs). No setting changed. Projected GPU time on one RTX 5090:
    500 scenarios × 6 = 45 h for all four models (gpt-oss 9, Mistral 3, Qwen 22, Gemma 12) ≤ 7 days → **all four models
    run on the first 500 scenarios** (S00000–S00499; the first 200 remain the 8-model comparison set).
    Files: `results/pilot_open_*.jsonl`, `results/summary_pilot_open.md` (not pooled).
  - 5 Nov 2026, MAIN OPEN-MODEL RUNS DONE (record only; nothing changed from the settings above): S00000–S00499 × 3 set-ups
    × 2 languages = 3,000 per model, 12,000 in all; 0 errors; 0 cut off (all stop_reason "stop"; longest output 18,036
    tokens); Ollama 0.40.13, one model at a time on GPU 1 (port 11437). Times Europe/Istanbul, GPU hours = summed latency:
    gpt-oss:20b (reasoning "medium", digest f38aa0c53da5…) 3 Nov 04:27–12:39, 8.2 h; mistral-small3.2:24b (no thinking,
    5a408ab55df5…) 3 Nov 12:40–15:13, 2.6 h; gemma4:31b (thinking on, 17ba34c06c80…) 3 Nov 15:13–4 Nov 02:55, 11.7 h;
    qwen3.8:27b (thinking on, aaee06c39dcf…) 4 Nov 02:56–5 Nov 00:28, 21.5 h. One digest per model across all its records
    (full digests in each record's `settings.digest`). Files `results/main_open_*.jsonl`; analysed with the 4 Claude files
    in `results/summary_main.md` (Claude-only version kept as `results/summary_main_api_only.md`).
    Note for reporting: in the 8-model summary each Holm family covers all 8 models, so the Holm p-values of the Claude
    comparisons are larger than in the Claude-only summary (raw p unchanged; e.g. Sonnet 5.5 closed-book TR vs EN under:
    Holm p 7e-5 → 1.3e-4). The paper must say which family it uses.
    Likewise the bootstrap draws all model pairs from one random stream (seed 1), so adding models shifts the Claude-pair
    intervals by up to 1.0 point (point estimates unchanged; no interval changes whether it excludes zero; e.g. Haiku 5.5 vs
    Sonnet 5.5 closed-book TR under: −5.5 to 10.5 → −5.0 to 11.0). GEE odds ratios and Fleiss κ for Claude are unchanged.
- Planned next (needs Amendment 3 before running): other vendors via API (GPT, Gemini); realistic documents; RAG over the
  official SORA PDF; robustness checks (effort, paraphrases, Firth); optional human/expert baseline.

## Ethics / copyright
No human subjects so far (a human baseline would need ethics approval). JARUS PDF says "no copying without JARUS permission":
paper and code reproduce no JARUS text; tables restated as numbers with citation. Decide before release whether to ask JARUS
for permission to publish the rule card.
