#!/usr/bin/env bash
# Re-run every engine and re-score. Each step is resumable; comment out what you don't need.
# Cloud steps cost money (about $16 on OpenRouter for both test sets, Sep-Oct 2026 prices).
set -euo pipefail
cd "$(dirname "$0")"
IV=data/ivrit/manifest.json; ID=data/id100/manifest.json
P=results/predictions

# 0. data (gated: accept the terms on the Hub and `huggingface-cli login` first)
[ -f "$IV" ] || python scripts/prepare_ivrit.py

# 1. cloud and hosted open models via OpenRouter
for m in google/gemini-3.8-flash google/gemini-3.1-pro-preview openai/gpt-5.6-sol anthropic/claude-opus-5.5 \
         qwen/qwen3.8-27b qwen/qwen3-vl-32b-instruct qwen/qwen3-vl-8b-instruct z-ai/glm-4.6v \
         mistralai/mistral-small-3.2-24b-instruct meta-llama/llama-4-maverick; do
  f="or__$(echo "$m" | tr '/:' '__').jsonl"
  python runners/openai_compat.py "$m" $IV $P/ivrit/$f
  TASK=id python runners/openai_compat.py "$m" $ID $P/id100/$f
done
# their reasoning mode returned empty answers, so these two run with reasoning off
for m in google/gemma-4-31b-it qwen/qwen3.6-35b-a3b; do
  f="or__$(echo "$m" | tr '/:' '__').jsonl"
  EFFORT=off python runners/openai_compat.py "$m" $IV $P/ivrit/$f
  EFFORT=off TASK=id python runners/openai_compat.py "$m" $ID $P/id100/$f
done
python runners/mistral_ocr.py $IV $P/ivrit/or__mistral-ocr.jsonl
python runners/mistral_ocr.py $ID $P/id100/or__mistral-ocr.jsonl

# 2. local CPU engines
python runners/mishkefet.py $IV $P/ivrit/local__mishkefet-v1.jsonl
CLEAN=1 python runners/mishkefet.py $ID $P/id100/local__mishkefet-v1.jsonl
python runners/digit_readers.py $ID $P/id100/local__our-digit-readers.jsonl
python runners/hf_local.py trocr $IV $P/ivrit/local__trocr-hebrew.jsonl
python runners/hf_local.py glmocr $IV $P/ivrit/local__glm-ocr.jsonl
# HebHTR needs its own Python 3.6 / TensorFlow 1.12 environment, see runners/hebhtr.py
# LANG=C.UTF-8 py36/bin/python runners/hebhtr.py word_beam $IV $P/ivrit/local__hebhtr.jsonl

# 3. own GPU (RunPod + vLLM): see README, "Own-GPU models"

# 4. score
python scoring/score_ivrit.py
python scoring/score_id100.py
