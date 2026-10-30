#!/system/bin/sh
cd /data/local/tmp/tabllm
export LD_LIBRARY_PATH=/data/local/tmp/tabllm
EOS=$(cat eos.txt)
OMP_NUM_THREADS=${1:-4}
./tabllm smollm2-135m.tllm @prompt_tokens.bin ${2:-32} ${3:-0.7} ${4:-40} out_tokens.bin $EOS