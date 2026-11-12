#!/usr/bin/env bash
# Amendment 2 infrastructure pilot: S02990-S02999 x 3 set-ups x 2 languages = 60 calls per model. Not pooled.
# Needs ./serve_ollama_gpu1.sh running. Each model starts once its pull has finished.
set -u
cd "$(dirname "$0")"
URL=http://127.0.0.1:11437

run() {  # model, think, out
  until curl -s $URL/api/tags | grep -q "\"name\":\"$1\""; do sleep 60; done
  echo "== $1 (think=$2) $(date +%T)"
  python3 -m soraeval.runner run --provider ollama --model "$1" --think "$2" \
    --offset 2990 --n 10 --workers 1 --budget 1 --out "results/$3"
}

run gpt-oss:20b          medium pilot_open_gptoss20b.jsonl
run mistral-small3.2:24b ""     pilot_open_mistralsmall32.jsonl
run qwen3.8:27b          true   pilot_open_qwen38_27b.jsonl
run gemma4:31b           true   pilot_open_gemma4_31b.jsonl
echo "PILOT DONE $(date +%T)"
