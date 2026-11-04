#!/bin/bash
# llm/serve.sh — Start the LLM server.
# Choose between Ollama (no GPU) and vLLM (GPU).
# Usage: bash llm/serve.sh [ollama|vllm] [model]

set -euo pipefail

MODE=${1:-ollama}
MODEL=${2:-llama3.2:3b}

if [ "$MODE" = "ollama" ]; then
    echo "Starting Ollama with model: $MODEL"
    which ollama || { echo "Install Ollama: https://ollama.com/download"; exit 1; }
    ollama serve &
    sleep 3
    ollama pull "$MODEL"
    echo "Ollama ready at http://localhost:11434"
    echo "OpenAI-compat endpoint: http://localhost:11434/v1/chat/completions"

elif [ "$MODE" = "vllm" ]; then
    MODEL=${2:-mistralai/Mistral-7B-Instruct-v0.3}
    echo "Starting vLLM with model: $MODEL"
    which python || { echo "Python not found"; exit 1; }
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --quantization awq \
        --max-model-len 8192 \
        --port 8080 \
        --host 0.0.0.0 &
    echo "vLLM ready at http://localhost:8080"
    echo "Update LLM_BASE_URL=http://localhost:8080/v1 in .env"

elif [ "$MODE" = "llamacpp" ]; then
    GGUF=${2:-models/mistral-7b-instruct-v0.2.Q4_K_M.gguf}
    echo "Starting llama.cpp with model: $GGUF"
    which llama-server || { echo "Build llama.cpp first: https://github.com/ggerganov/llama.cpp"; exit 1; }
    llama-server \
        --model "$GGUF" \
        --n-gpu-layers 0 \
        --ctx-size 4096 \
        --port 8080 \
        --host 0.0.0.0 &
    echo "llama.cpp ready at http://localhost:8080"

else
    echo "Usage: $0 [ollama|vllm|llamacpp] [model]"
    exit 1
fi

wait
