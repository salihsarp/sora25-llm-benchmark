#!/usr/bin/env bash
# SORA-LLM study - run order actually used (31 Oct 2026). Resumable: re-run any line.
# Needs ANTHROPIC_API_KEY in .env (or the environment). Claude 5.5 models reject `temperature`; effort fixed at medium.
set -euo pipefail
cd "$(dirname "$0")"
python3 tests/test_all.py
python3 -m soraeval.runner generate --n 3000 --seed 2026          # sha256 1af09bad...7325
# pilots (see Amendment 1 in the analysis plan)
python3 -m soraeval.runner run --model claude-haiku-5-5 --n 40 --out results/pilot3_haiku55.jsonl --budget 2 --workers 10 --effort medium
# main grid via Message Batches (50% price)
python3 -m soraeval.runner batch-submit --model claude-haiku-5-5  --n 1500 --out results/main_haiku55.jsonl  --budget 10 --est-cost-per-call 0.001  --effort medium
python3 -m soraeval.runner batch-submit --model claude-sonnet-5-5 --n 500  --out results/main_sonnet55.jsonl --budget 20 --est-cost-per-call 0.0055 --effort medium
python3 -m soraeval.runner batch-submit --model claude-opus-5-5   --n 200  --out results/main_opus55.jsonl   --budget 20 --est-cost-per-call 0.014  --effort medium
python3 -m soraeval.runner batch-submit --model claude-haiku-4-5  --n 400  --out results/main_haiku45.jsonl  --budget 10 --est-cost-per-call 0.0035 --effort ""
python3 -m soraeval.runner batch-submit --model claude-haiku-5-5  --n 100 --reps 3 --langs en --conditions closed_book --out results/stability_haiku55.jsonl --budget 1 --est-cost-per-call 0.0015 --effort medium
# collect (repeat until all ended)
for f in main_haiku55 main_sonnet55 main_opus55 main_haiku45 stability_haiku55; do
  python3 -m soraeval.runner batch-collect --out results/$f.jsonl --wait
done
python3 -m soraeval.analyze --results results/main_haiku55.jsonl results/main_sonnet55.jsonl results/main_opus55.jsonl results/main_haiku45.jsonl --out results/summary_main.md
