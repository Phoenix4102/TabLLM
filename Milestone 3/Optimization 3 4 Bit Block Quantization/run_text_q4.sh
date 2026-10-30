#!/system/bin/sh
cd /data/local/tmp/tabllm
export LD_LIBRARY_PATH=/data/local/tmp/tabllm

PROMPT="$1"
MAX_TOKENS="${2:-32}"
TEMPERATURE="${3:-0.7}"
TOP_K="${4:-40}"
EOS=$(cat eos.txt)

if [ -z "$PROMPT" ]; then
    PROMPT="What is the capital of France?"
fi

OMP_NUM_THREADS=4 TLLM_CHAT=1 ./tabllm \
  smollm2-135m-q4.tllm \
  "$PROMPT" \
  "$MAX_TOKENS" \
  "$TEMPERATURE" \
  "$TOP_K" \
  out_tokens.bin \
  "$EOS" \
  tokenizer.json