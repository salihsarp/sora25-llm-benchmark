#!/usr/bin/env bash
# Ollama server for this study only (Amendment 2): GPU 1, port 11437, one request at a time.
# Vulkan is switched off because it ignores CUDA_VISIBLE_DEVICES and would also see GPU 0.
cd "$(dirname "$0")"
mkdir -p logs
export CUDA_VISIBLE_DEVICES=1 GGML_VK_VISIBLE_DEVICES=-1 OLLAMA_VULKAN=0
export OLLAMA_HOST=127.0.0.1:11437 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_CONTEXT_LENGTH=40960 OLLAMA_KEEP_ALIVE=30m
exec ollama serve >> logs/ollama_serve_11437.log 2>&1
