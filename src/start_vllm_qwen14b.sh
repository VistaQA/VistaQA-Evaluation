#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

conda run -n vistaqa python -m vllm.entrypoints.openai.api_server \
  --model /path/Qwen2.5-14B-Instruct \
  --served-model-name Qwen2.5-14B-Instruct \
  --host 0.0.0.0 \
  --port 8001 \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9
