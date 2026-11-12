#!/usr/bin/env bash
# Amendment 2 main runs: first 500 scenarios x 3 set-ups x 2 languages = 3,000 calls per model (~45 h in all on GPU 1).
# Needs ./serve_ollama_gpu1.sh running. Resumable: rerunning skips finished keys.
set -u
cd "$(dirname "$0")"

run() {  # model, think, out
  echo "== $1 (think=$2) $(date '+%F %T')"
  python3 -m soraeval.runner run --provider ollama --model "$1" --think "$2" \
    --offset 0 --n 500 --workers 1 --budget 1 --out "results/$3"
}

run gpt-oss:20b          medium main_open_gptoss20b.jsonl
run mistral-small3.2:24b ""     main_open_mistralsmall32.jsonl
run gemma4:31b           true   main_open_gemma4_31b.jsonl
run qwen3.8:27b          true   main_open_qwen38_27b.jsonl
echo "MAIN OPEN DONE $(date '+%F %T')"
